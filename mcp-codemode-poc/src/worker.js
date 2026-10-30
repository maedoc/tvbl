// Guest sandbox: Pyodide (wasm CPython) running inside a Node worker thread.
// The worker has no network and no fs access to the host beyond the vendored
// Pyodide assets it was handed. Guest Python reaches the host only through
// opaque handle proxies injected below — it cannot see the ops registry, the
// log, or any credentials.

import { parentPort, workerData } from "node:worker_threads";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const pyodideDir = path.resolve(here, "..", "vendor", "pyodide");

// pyodide.asm.js is CJS and expects node globals; provide them before import.
globalThis.require = require;
globalThis.__dirname = pyodideDir;
globalThis.__filename = path.join(pyodideDir, "pyodide.asm.js");

const { loadPyodide } = await import(
  new URL(`file://${path.join(pyodideDir, "pyodide.mjs")}`).href
);

async function main() {
  const t0 = performance.now();
  const pyodide = await loadPyodide({
    indexURL: pyodideDir + path.sep,
    lockFileURL: path.join(pyodideDir, "pyodide-lock.json"),
  });
  const t1 = performance.now();
  await pyodide.loadPackage(["numpy", "scipy"]);
  const t2 = performance.now();

  // Handle bridge: the host side owns the real op implementations. The worker
  // only gets the registry of handle names; each guest call is posted back to
  // the host, executed there, logged, and the capped result returned. This is
  // the postMessage boundary — the guest never holds a host function.
  const pending = new Map();
  let callSeq = 0;
  parentPort.on("message", (msg) => {
    if (msg.type === "op-result" && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.ok ? resolve(msg.result) : reject(new Error(msg.error));
    }
  });
  function hostCall(name, args) {
    return new Promise((resolve, reject) => {
      const id = ++callSeq;
      pending.set(id, { resolve, reject });
      // structured clone requires plain JSON-safe data at the boundary; a
      // Python dict arrives here as a Map-backed proxy, so serialize explicitly
      let plain = {};
      if (args != null) {
        if (typeof args.toJs === "function") plain = args.toJs({ dict_converter: Object.fromEntries });
        else plain = args;
      }
      parentPort.postMessage({ type: "op-call", id, name, args: JSON.parse(JSON.stringify(plain)) });
    });
  }
  // Pyodide proxies await returned promises, so async host ops work from Python.
  const handleProxies = {};
  for (const name of Object.keys(workerData.handleSpec)) {
    handleProxies[name] = (args) => hostCall(name, args);
  }

  pyodide.globals.set("_handles", pyodide.toPy(handleProxies));

pyodide.runPython(`
_handles = globals()["_handles"]

class HostAPI:
    """Opaque handle-based API surface exposed to guest code."""
    def __getattr__(self, name):
        try:
            h = _handles[name]
        except Exception:
            raise AttributeError(f"op '{name}' is not available (policy by omission)")
        async def call(args=None):
            import json
            payload = {} if args is None else json.loads(json.dumps(args, default=lambda o: o.to_py() if hasattr(o, "to_py") else str(o)))
            return (await h(payload)).to_py()
        return call

host = HostAPI()
`);

  parentPort.postMessage({
    type: "ready",
    boot: { pyodide_ms: Math.round(t1 - t0), packages_ms: Math.round(t2 - t1) },
    pyodide: pyodide.version,
  });

  parentPort.on("message", async (msg) => {
    if (msg.type !== "run") return;
    const t0 = performance.now();
    try {
      // runPythonAsync: guest code may await host op promises (they resolve
      // via postMessage round-trips) without JSPI stack switching.
      const result = await pyodide.runPythonAsync(msg.code);
      let out = null;
      if (result !== undefined && result !== null) {
        if (typeof result.toJs === "function") {
          out = JSON.parse(JSON.stringify(result.toJs({ dict_converter: Object.fromEntries })));
        } else {
          out = result;
        }
      }
      parentPort.postMessage({
        type: "result",
        id: msg.id,
        ok: true,
        ms: Math.round((performance.now() - t0) * 100) / 100,
        result: out,
      });
    } catch (err) {
      parentPort.postMessage({ type: "result", id: msg.id, ok: false, ms: Math.round((performance.now() - t0) * 100) / 100, error: String(err && err.message || err) });
    }
  });
}

main().catch((err) => {
  parentPort.postMessage({ type: "fatal", error: String(err && err.stack || err) });
});
