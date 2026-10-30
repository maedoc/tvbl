// Host process: spawns the sandboxed Pyodide worker, owns the op registry and
// the JSONL op log, and executes guest-submitted Python. Credentials/API
// clients (mocked here) live only on this side of the boundary.

import { Worker } from "node:worker_threads";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createOps } from "./ops.js";
import { createOpLog } from "./oplog.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const log = createOpLog(path.resolve(here, "..", "runs", `run-${Date.now()}.jsonl`));
log.append({ t: Date.now(), op: "__run_start__", tier: "meta", note: "pyodide-wasm sandbox run" });

const { ops } = createOps({ log });

// Handles cross the worker boundary as plain data; the worker wraps them in
// callable proxies on its side. Functions cannot cross postMessage.
const handleSpec = Object.fromEntries(
  Object.entries(ops).map(([name, h]) => [name, { __handle: h.__handle }])
);

const worker = new Worker(path.resolve(here, "worker.js"), {
  workerData: { handleSpec },
  // No env passthrough of secrets: the worker gets an empty environment.
  env: {},
});

let runId = 0;

// Host-side execution of guest op calls: this is the only place op
// implementations, the log, and (mocked) credentials exist.
worker.on("message", (msg) => {
  if (msg.type === "op-call") {
    const op = ops[msg.name];
    if (!op) {
      worker.postMessage({ type: "op-result", id: msg.id, ok: false, error: `unknown op '${msg.name}'` });
      return;
    }
    try {
      const value = op.call(msg.args);
      worker.postMessage({ type: "op-result", id: msg.id, ok: true, result: JSON.parse(JSON.stringify(value)) });
    } catch (err) {
      worker.postMessage({ type: "op-result", id: msg.id, ok: false, error: String(err && err.message || err) });
    }
  }
});

function run(code) {
  return new Promise((resolve, reject) => {
    const id = ++runId;
    const onMsg = (msg) => {
      if (msg.type === "ready") {
        console.log(`[host] worker ready: pyodide ${msg.pyodide}, boot ${JSON.stringify(msg.boot)}`);
        worker.postMessage({ type: "run", id, code });
      } else if (msg.type === "result" && msg.id === id) {
        worker.off("message", onMsg);
        msg.ok ? resolve(msg) : reject(new Error(msg.error));
      } else if (msg.type === "fatal") {
        reject(new Error(msg.error));
      }
    };
    worker.on("message", onMsg);
    worker.on("error", reject);
  });
}

const guestCode = `
import numpy as np

async def main():
    # 1. open op: echo through a handle
    ack = await host.echo({"msg": "hello from wasm", "n": 41})

    # 2. real op with mocked backend: host-side "simulation"
    sim = await host.run_mock_simulation({"coupling": 1.2, "steps": 200, "seed": 7})

    # 3. numerical work stays in the guest
    x = np.linspace(0, 2*np.pi, 1000)
    spectrum = float(np.abs(np.fft.rfft(np.sin(x))).max())

    return {
        "echo": ack,
        "sim_final_state": sim["final_state"],
        "sim_trace_tail_len": len(sim["trace_tail"]),
        "fft_peak": spectrum,
    }

await main()
`;

run(guestCode)
  .then(async (res) => {
    console.log(`[host] guest run ok in ${res.ms} ms`);
    console.log("[host] result:", JSON.stringify(res.result, null, 2));
    log.append({ t: Date.now(), op: "__run_end__", tier: "meta", ok: true, ms: res.ms });
    await log.close();
    console.log(`[host] op log: ${log.path} (${log.count()} entries)`);
    worker.terminate();
  })
  .catch(async (err) => {
    console.error("[host] guest run failed:", err.message);
    log.append({ t: Date.now(), op: "__run_end__", tier: "meta", ok: false, error: err.message });
    await log.close();
    worker.terminate();
    process.exit(1);
  });
