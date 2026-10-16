# tvbl: a jupyter-lite based tvb


## Build locally

```sh
micromamba create -f build-environment.yml -n build-tvbl
micromamba activate build-tvbl
jupyter lite build --contents content
```

## run locally

I use uv, the list of requirements are in uv-requirements.txt plus a few, the environment.yml is still best to look at

```sh

rm -rf .venv

uv venv 
source .venv/bin/activate

uv pip install -r ./uv-requirements.txt

uv pip install pytest

which pytest

export PYTHONPATH=content

python -c 'import tvbl'

pytest


```

## run tests (wasm path)

The suite can also be run against the wasm build of the code — i.e. inside an
`emscripten-wasm32` Python where `pyjs` is importable and `tvbl.core.cfun`
actually uses the JavaScript implementation (validating it against numpy
within 1e-5). This is what runs inside JupyterLite, so it is the path worth
running before a deploy.

You need **micromamba** (it is what `build-environment.yml` / CI uses, and the
only way to create an `emscripten-wasm32` env), `pyjs-code-runner` (drives a
headless chromium via playwright) and a wasm env:

```sh
# one-time: micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba
mv /tmp/bin/micromamba ~/.local/bin/

# one-time: the wasm env (same channels/order as environment.yml, plus
# pytest/pyjs; 4x must precede dev or the ABI is mixed -- numba only exists on 4x)
export MAMBA_ROOT_PREFIX=~/.micromamba
micromamba create -n tvbl-wasm --platform emscripten-wasm32 \
  -c https://repo.prefix.dev/emscripten-forge-4x \
  -c https://repo.prefix.dev/emscripten-forge-dev \
  -c https://repo.prefix.dev/conda-forge \
  python numpy scipy tqdm pytest pyjs autograd numba

# one-time: the host-side runner
uv venv ~/.venvs/pyjs-runner
uv pip install --python ~/.venvs/pyjs-runner/bin/python pyjs-code-runner
~/.venvs/pyjs-runner/bin/playwright install chromium
```

Then, to run the tests:

```sh
./wasm-tests/run.sh
```

It packs the wasm env with `empack`, patches a private copy of the pyjs runtime
(see the numba note below), mounts `content/tvbl` and
`wasm-tests/run_wasm_tests.py` into the emscripten filesystem, runs pytest in
headless chromium and exits non-zero on failure (so it is CI-able). Paths can
be overridden with `TVBL_WASM_ENV` and `PYJS_RUNNER`.

Gotchas learned the hard way:

- `micromamba install` into an existing wasm env fails (`pip inspect` tries to
  run the wasm `python`); create the env with every package in one `create`.
- `pyjs-code-runner`'s `node` backend does not work with the emscripten-forge
  pyjs build — it only ships `pyjs_runtime_browser.{js,wasm}` — so use
  `browser-main --headless`.
- `pyjs.buffer_to_js_typed_array` only accepts 1-D buffers, hence the
  `.reshape(-1)` views in `content/tvbl/core.py`.
- numba & llvmlite only exist on `emscripten-forge-4x`, which is the
  emscripten-abi 4.x channel, while `emscripten-forge-dev` is still 3.1.73.
  `environment.yml` therefore lists 4x *first*: dev-first solves "fine" but
  silently mixes both ABIs (python/core from 4x, matplotlib/contourpy from
  dev).
- numba array kernels need `_nrt_python...so` (which defines
  `NRT_adapt_ndarray_from_python`) to be dlopen'd **RTLD_GLOBAL**. The pyjs
  runtime's `libraryType()` marks every `*cpython-3*-wasm32-emscripten.so` as
  "local" and dlopen()s it `RTLD_LOCAL`, while llvmlite dlopen()s each
  JIT-compiled function `RTLD_GLOBAL`, so the generated side module cannot
  resolve numba's `NRT_*` imports and every `@njit` call taking an array dies
  with `Dynamic linking error: cannot resolve symbol
  NRT_adapt_ndarray_from_python` (scalar-only `@njit` is fine).
  `wasm-tests/run.sh` works around it by patching that one flag in a private
  copy of `pyjs_runtime_browser.js` and passing it via `--pyjs-dir`. This is
  probably worth reporting upstream to emscripten-forge/pyjs — if the deployed
  JupyterLite kernel boots its env the same way, it is affected too.
- `environment.yml` (the deployed site kernel) has no `pytest`; to run the
  suite *inside* the deployed JupyterLite, add `pytest` there (it exists on
  emscripten-forge).

## ideas

most of want we want to do is possible 

- simulation
- sbi
- plotting

maybe we need interaction with storage, via

- ebrains-drive
- https://jupyter-server.readthedocs.io/en/latest/developers/contents.html to use ebrains-drive for storage?

maybe need pyunicore access?
