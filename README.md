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

The upstream `tvb_library` suite runs against the wasm build — i.e. inside an
`emscripten-wasm32` Python where `pyjs` is importable and numba/llvmlite JIT
into the browser's WebAssembly memory. This is the same environment the
JupyterLite kernel uses, so it is the path worth checking before a deploy.

`wasm-tests/run.sh` creates the wasm env (on first run), installs
`tvb_library` from the sibling checkout at `../tvb/tvb_library` (or from
`the-virtual-brain/tvb-root@master` when that checkout is absent), installs
`tvb-data`, patches a private copy of the pyjs runtime, and runs the tests in
headless chromium.

Prerequisites (one-time):

```sh
# micromamba (creates the emscripten-wasm32 env)
curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj -C /tmp bin/micromamba
mv /tmp/bin/micromamba ~/.local/bin/      # linux-64: .../linux-64/latest

# host-side runner (drives playwright/chromium)
uv venv ~/.venvs/pyjs-runner
uv pip install --python ~/.venvs/pyjs-runner/bin/python pyjs-code-runner
~/.venvs/pyjs-runner/bin/playwright install chromium
```

Then:

```sh
./wasm-tests/run.sh
```

The env is created with every package in one `micromamba create` (installing
into an existing wasm env later fails because micromamba tries to run the wasm
`python`). Paths can be overridden with `TVBL_WASM_ENV`, `TVB_LIBRARY_SRC`
(e.g. point at `../tvb/tvb_library` explicitly), `TVB_LIBRARY_SPEC`,
`PYJS_RUNNER`, and `HOST_PYTHON`. `TVBL_SKIP_INSTALL=1` reuses the packages
already in the env.

Expected result in this checkout: **1448 passed, 94 skipped**. The skips are
environmental, not disabled functionality: GPU/CUDA tests, six `tvb-gdist`
tests (no emscripten build), multiprocessing/prange tests (emscripten has no
processes), one numba kernel whose analysis pass needs >2 GiB, and one
benchmark that allocates beyond the wasm memory ceiling. They are skipped by
`wasm-tests/run_wasm_tests.py` with a reason in the pytest report.

The suite is split into five chunks (basic, datatypes, simulator top-level,
backend, hybrid), each in a fresh browser process. A single process running
everything exhausts the 2 GiB wasm memory late in the suite
(`_ArrayMemoryError`) even though each chunk passes on its own.

### Notebook demos

`content/demos/` holds a few upstream TVB demos (region-level simulation,
reduced Wong–Wang, and the hybrid JansenRit → FHN example). They are ordinary
notebooks; to execute them headlessly in the wasm runtime use:

```sh
~/.venvs/pyjs-runner/bin/pyjs_code_runner run script browser-main \
  --conda-env ~/.micromamba/envs/tvbl-wasm \
  --script /work/run_notebooks.py \
  --mount "$PWD/wasm-tests:/work" \
  --mount "$PWD/content/demos:/work/notebooks" \
  --pyjs-dir "$PWD/wasm-tests/.pyjs-patched" \
  --file-filter "$PWD/wasm-tests/empack_config.yaml" \
  --headless -w /home/web_user
```

`run_notebooks.py` executes all `*.ipynb` in the mounted directory (or the
single name in `/work/notebook.txt`) and strips `%magic`/`!shell` lines.
All three demos pass.

The GitHub Actions deployment additionally builds `tvb-gdist` with the
`rattler-build` recipe in `wasm-tests/recipes/tvb-gdist`, publishes it to a
local channel, and installs it into the kernel env. The test runner above does
not build it by default (see the recipe README for the local workflow); the
deployed site therefore has working surface/geodesic functionality.

Gotchas learned the hard way:

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
  `wasm-tests/run.sh` patches that one flag in a private copy of
  `pyjs_runtime_browser.js` and passes it via `--pyjs-dir`. This is worth
  reporting upstream to emscripten-forge/pyjs.
- matplotlib 3.11 changed the default `text.hinting` from `force_autohint` to
  FreeType's native `LOAD_DEFAULT`. The emscripten-forge FreeType build has no
  native hinter, so *any* Agg text rendering (plot titles, `tight_layout`,
  `savefig`) fails with `RuntimeError: failed to load glyph`. Both
  `environment.yml` and the test env therefore pin `matplotlib <3.11`.
- The JupyterLite kernel env (`environment.yml`) installs `tvb-library` and
  `tvb-data` through the `pip:` section. `jupyterlite-xeus` runs host pip with
  `--target` and only supports pure-Python wheels, which is all tvb needs.
  pip accepts a VCS ref there, so the package is pinned to
  `git+https://github.com/the-virtual-brain/tvb-root.git@master#subdirectory=tvb_library`
  rather than the older PyPI release. Use `@<tag>` or `@<commit>` for a
  reproducible pin; pip clones with `--filter=blob:none` (a few seconds).

## ideas

most of want we want to do is possible 

- simulation
- sbi
- plotting

maybe we need interaction with storage, via

- ebrains-drive
- https://jupyter-server.readthedocs.io/en/latest/developers/contents.html to use ebrains-drive for storage?

maybe need pyunicore access?
