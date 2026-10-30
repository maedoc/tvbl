// Host-side op registry. Tiered: "open" ops run without approval, "gated" ops
// would suspend for approval (not wired in this POC), anything not listed here
// does not exist for the guest (policy by omission).
// Every op appends one line to the per-run JSONL log, which doubles as the
// replay log for deterministic re-execution.

export const MAX_RESULT_BYTES = 256 * 1024;

export function createOps({ log }) {
  const handles = new Map();
  let nextId = 1;

  function record(entry) {
    log.append(entry);
  }

  function makeHandle(name, tier, fn) {
    const id = `h${nextId++}`;
    handles.set(id, { name, tier });
    return {
      __handle: id,
      call(args) {
        const seq = record({ t: Date.now(), op: name, tier, args: summarize(args) });
        let result;
        try {
          result = fn(args);
        } catch (err) {
          record({ t: Date.now(), op: name, seq, ok: false, error: String(err && err.message || err) });
          throw err;
        }
        const capped = cap(result);
        record({ t: Date.now(), op: name, seq, ok: true, result_bytes: capped.__bytes });
        return capped.value;
      },
    };
  }

  function summarize(args) {
    const s = JSON.stringify(args);
    return s.length > 512 ? s.slice(0, 512) + `...<${s.length}B>` : args;
  }

  function cap(result) {
    const s = JSON.stringify(result);
    const bytes = Buffer.byteLength(s);
    if (bytes <= MAX_RESULT_BYTES) return { value: result, __bytes: bytes };
    return { value: { __truncated: true, bytes, preview: s.slice(0, 1024) }, __bytes: bytes };
  }

  const ops = {
    echo: makeHandle("echo", "open", (args) => ({ echoed: args })),
    // Mocked "real" op: a fake TVB-like simulation backend living only in the
    // host. The guest never sees credentials or transport, just a handle.
    run_mock_simulation: makeHandle("run_mock_simulation", "open", (args) => {
      const { coupling = 1.0, steps = 100, seed = 0 } = args || {};
      let x = 0.5 + seed * 1e-6;
      const trace = [];
      for (let i = 0; i < steps; i++) {
        x = x + coupling * 0.01 * Math.sin(x * 3.0 + i * 0.1);
        trace.push(x);
      }
      return {
        backend: "mock-tvb-wongwang",
        params: { coupling, steps, seed },
        final_state: x,
        trace_tail: trace.slice(-8),
      };
    }),
  };

  return { ops, handles };
}
