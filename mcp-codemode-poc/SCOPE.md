# Scope: Code Mode MCP-style sandboxed simulation harness (POC)

Architecture transposed from a Deno-based Code Mode MCP server: the model writes
Python; the code runs in a wasm sandbox (Pyodide in a worker); the sandbox talks
to the outside world only through opaque, enumerated, tiered host ops over a
postMessage boundary. Credentials and API clients live only in the host.

Everything below was measured in this sandbox on Node v22.23.4 / Deno 2.9.6 /
Python 3.12.15, against vendored Pyodide 0.27.8 (CPython 3.12.7, emscripten
3.1.58, wasm32).

## Decision 1 — Offline Pyodide in a permissions-none worker

**Q: Can Pyodide load fully offline in a permissions-none worker?**

Yes, with one caveat on Deno. Vendored assets (53 MB total in
`vendor/pyodide/`: `pyodide.asm.wasm` 9.7 MB, `python_stdlib.zip` 2.3 MB,
`pyodide.asm.js` 1.2 MB, lock file, plus wheels) load with **zero network
fetches** when `indexURL`/`lockFileURL` point at the local directory — measured
core boot ~1.1–1.2 s on Node. Wheels must be pre-downloaded into the same
directory (Pyodide's `loadPackage` falls back to the jsdelivr CDN otherwise);
we vendored numpy/scipy/scikit-learn/matplotlib/h5py wheels (see Q2).

Deno 2.x path, measured with Deno 2.9.6:

- `new Worker(url, { permissions: "none" })` in Deno **cannot even read its own
  module file** ("Requires read access to .../worker.ts") — `permissions:
  "none"` also strips the read needed to load the worker script itself.
- Workaround that works: the host (run with scoped `--allow-read=<vendor dir>`)
  reads the worker source and spawns it from a **Blob URL** with
  `permissions: "inherit"` (inheriting only the vendored-asset read) — worker
  booted and ran numpy in **2.1 s**. With `permissions: "none"` the blob worker
  fails at `import("file://...pyodide.mjs")` with "Requires read access".
- The CJS `pyodide.asm.js` additionally needs `require`/`__dirname`/`__filename`
  globals shimmed (2-line `createRequire` shim) before importing
  `pyodide.mjs` — painful but mechanical.

**Fallback host verdict: Node 22 `worker_threads` is simpler for the
prototype.** Same wasm isolation, `workerData`/`postMessage` semantics are
identical, no permission dance, and the worker can be spawned with `env: {}`
(no secret leakage) plus `resourceLimits`. Deno's advantage (deny-by-default
syscalls, `permissions: "none"`) is real for production, but the blob-worker +
CJS-shim friction makes it the wrong first prototype host. Plain browser is
viable too (Pyodide's native habitat, real `Worker` + `postMessage`), but then
the "host" is the page and there is no fs for the JSONL op log without an
OPFS layer — Node wins for a headless MCP-style server.

## Decision 2 — Package reality check (Pyodide 0.27.8, pyodide-lock.json, 351 pkgs)

All five target packages ship as wasm32 wheels and load offline from the
vendored directory:

| package | wheel | wheel size | worker RSS after load (cumulative) |
|---|---|---|---|
| numpy 2.0.2 | cp312 pyodide_2024_0 wasm32 | 3.0 MB | 238 MB (core alone: 180 MB) |
| scipy 1.14.1 | + openblas 0.3.26 | 13 MB | 645 MB with numpy |
| scikit-learn 1.6.1 | + joblib, threadpoolctl | 6.0 MB | — |
| matplotlib 3.8.4 | + contourpy, pillow, fonttools… | 6.5 MB | — |
| h5py 3.12.1 | + pkgconfig | 1.7 MB | — |
| **all five** | | ~28 MB wheels | **1371 MB RSS**, package load 4.0 s |

Notes: core Pyodide boot ~1.15 s / 180 MB RSS; numpy adds ~0.4 s; scipy
(openblas) adds ~1.3 s; all five ~4 s and ~1.4 GB RSS per worker. That is the
per-worker memory footprint to budget (wasm linear memory caps at 4 GB;
`--max-old-space-size` is irrelevant — the ceiling is wasm memory). For tvbl
itself: `tvb-library` is pure Python and already runs in this repo's
emscripten-forge pyjs build, but stock Pyodide has **no numba/llvmlite/tvb-gdist** —
Pyodide ≠ the pyjs kernel this repo ships in JupyterLite; bridging that gap is
future work, not this POC.

## Decision 3 — Handle marshalling across the Pyodide/JS boundary

Measured on Node 22, Pyodide 0.27.8:

- Small JSON op through a `registerJs`/globals proxy: **~0.18 ms/call**
  (1000 calls in 183 ms, sync); through the worker postMessage round-trip:
  **~0.06 ms/call** when already warm (200 sequential awaited calls) — i.e.
  the boundary is cheap for control-plane ops; batching matters only above
  ~10³ calls/run.
- ndarray → JS: `arr.toJs()` on an 8 MB float64 array = **~6–11 ms** (a memcpy
  out of wasm linear memory into a JS TypedArray; `createTypedArray: true` is
  marginally faster). JS → Python re-wrap of the same 8 MB costs ~245 ms when
  it round-trips through `np.asarray` conversion — so **send buffers, not
  JSON**: cap JSON payloads (~256 KB in the POC) and move ndarrays as
  TypedArrays/transferables. postMessage structured clone copies ArrayBuffers
  efficiently; there is no zero-copy transfer across worker_threads, but
  `transferList` still avoids a second copy on the receiving side.
- Guest compute vs native: 1e6-element float64 square-sum = **1.4 ms in wasm**
  vs 0.94 ms native CPython+OpenBLAS — wasm is ~1.5× slower on this BLAS-bound
  microbench, entirely acceptable for simulation control flow; the heavy TVB
  numerics are what would need the pyjs/numba route anyway.
- Async ops from Python need `runPythonAsync` + `await` on the proxy promise;
  Pyodide 0.27.8 without JSPI throws "WebAssembly stack switching not
  supported" for `run_sync` on a pending promise. All args/results at the
  postMessage boundary must be structured-cloneable (plain JSON — a Python
  dict arrives as a Map-backed proxy and must be serialized explicitly).

## Decision 4 — wasm vs CPython subprocess + seccomp, head-to-head

| axis | Pyodide wasm worker | CPython subprocess (seccomp/systemd) |
|---|---|---|
| isolation | memory-safe wasm, no syscalls except via host shims; worker has no fs/net by construction | process boundary only; a plain subprocess **can open sockets** (verified) — needs seccomp-bpf + dropped caps + no-new-privs, or a container; systemd sandboxing is Linux-only and leaks fd/env by default |
| cold start | ~1.2 s core, ~2.5 s with numpy+scipy | ~30–80 ms interpreter start |
| memory/worker | 180 MB core, 645 MB w/ numpy+scipy, 1.4 GB w/ all five; 4 GB wasm ceiling | ~15 MB interpreter, full host RAM |
| packages | curated 351-wheel lockfile; **no numba, no tvb-gdist, no arbitrary C extensions** | anything pip-installable, incl. tvb-library + numba + gdist |
| perf | ~1.5× slower on BLAS microbench | native |
| determinism/replay | wasm is deterministic; op log replays exactly | subprocess + wall-clock is fine, but seccomp profiles are brittle |

Honest verdict: **the wasm route loses if the workload needs packages Pyodide
doesn't ship** — for tvbl that means numba-jitted kernels and `tvb-gdist`
(geodesics), which only exist in this repo's emscripten-forge pyjs build. For
pure-numpy/scipy simulation-analysis loops (SBI with sklearn, plotting with
matplotlib, h5py IO) wasm wins on isolation strength per unit of host
privilege: the host process can be run with no network and no fs grants, and
the guest physically cannot reach either. Recommendation: prototype on
Node+Pyodide (this POC), keep the op interface identical so a CPython-subprocess
backend can be swapped in later for numba-heavy workloads.

## Prototype definition of done — met

Single host process (`src/host.js`) spawns one worker (`src/worker.js`) that
boots Pyodide **fully offline from `vendor/pyodide/`** (no fetch at init —
verified by pointing `indexURL`/`lockFileURL` at the vendored dir and loading
numpy+scipy from local wheels). Guest Python calls two host ops through opaque
handles — `echo` and `run_mock_simulation` (mocked TVB-like backend living only
in the host) — across the postMessage boundary; every op call is appended to a
per-run JSONL log (`runs/run-*.jsonl`) that doubles as a replay log; the run
result is returned to the host. No MCP wiring, no approval gates, no real APIs.

Measured e2e: worker ready (pyodide + numpy + scipy, vendored) in
~2.5 s; guest run (echo + mock sim + numpy FFT) in **323 ms**; 6 JSONL entries.

## Out of scope for this POC

MCP protocol wiring, approval gates with suspend/resume, result cap
enforcement beyond 256 KB, real EBrains/UNICORE APIs, transferable-buffer
fast path for ndarrays across the worker boundary, multi-worker pools,
per-run CPU/time limits (wasm has none built-in — host must enforce via
`worker.terminate()` + wall clock).
