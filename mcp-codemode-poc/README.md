# tvbl-mcp-codemode-poc

Code Mode MCP-style sandboxed Python execution harness for tvbl scientific
workflows: the model writes Python, the code runs in a Pyodide (wasm CPython)
worker, and the only way out is an enumerated set of host-side ops invoked
through opaque handles across a postMessage boundary. See [SCOPE.md](SCOPE.md)
for the full decision record with measured numbers.

## Layout

```
mcp-codemode-poc/
├── src/
│   ├── host.js    # host process: spawns worker, owns ops + JSONL op log
│   ├── worker.js  # guest sandbox: offline Pyodide, handle bridge, runs guest code
│   ├── ops.js     # tiered op registry (open / gated / absent-by-omission), result caps
│   └── oplog.js   # per-run append-only JSONL log (= replay log)
├── vendor/pyodide/  # vendored Pyodide 0.27.8 core + numpy/scipy/sklearn/matplotlib/h5py wheels (53 MB)
├── runs/          # per-run op logs land here
└── SCOPE.md
```

## Run

```sh
cd mcp-codemode-poc
npm install        # only needed to fetch the pyodide npm package once; the runtime uses vendor/
npm start          # = node src/host.js
```

Expected output: worker boots Pyodide offline from `vendor/pyodide/` in ~2.5 s,
runs guest code that calls `host.echo(...)` and `host.run_mock_simulation(...)`
through handles, and prints the result plus the path of the JSONL op log.

## Residual security assumptions (read before trusting this)

1. **wasm is not a syscall sandbox by itself.** Pyodide's emscripten runtime
   only performs syscalls the JS host shims allow; in this Node worker the
   guest has no fs/net/process access because Node worker_threads don't
   expose them — but the isolation guarantee comes from the *host runtime
   configuration*, not from wasm memory safety alone. A bug in the host (e.g.
   passing a host function the guest can call with crafted args) breaks it.
2. **The handle boundary is policy by omission**: only ops explicitly
   registered in `src/ops.js` exist for the guest. Adding an op is a host-side
   code change; anything destructive (fs writes, network, process spawn) must
   stay absent or approval-gated (gates are designed but not wired in this POC).
3. **Result caps and the JSONL log are host-side conveniences**, not a
   security boundary: a guest can exfiltrate data *through approved ops* (e.g.
   embed secrets in echo args) — the log records it, it doesn't prevent it.
4. **No resource limits inside wasm**: Pyodide has no built-in CPU/memory cap
   below the 4 GB wasm linear-memory ceiling. The host must enforce wall-clock
   and terminate workers (`worker.terminate()`); a guest can spin forever or
   allocate up to 4 GB. The POC does neither.
5. **Deterministic replay assumes ops are pure functions of their logged
   args.** `run_mock_simulation` is; real API-backed ops won't be (side effects,
   changing remote state) — replay of those must be mocked from the log.
6. **The worker is spawned with `env: {}`** so no host environment (including
   secrets) crosses into the sandbox; keep it that way.
7. **Vendored assets must be integrity-checked** (hashes in a lockfile) in any
   deployment; this POC trusts `vendor/pyodide/` blindly.
8. **Node worker_threads are not a hard security boundary** the way
   Deno `permissions: "none"` + blob workers (or a real container/seccomp)
   would be. For production, run the host with no network grants, or move the
   worker into Deno with scoped read of only the vendored directory (the
   working bootstrap path is documented in SCOPE.md decision 1).
