# tvb-gdist for emscripten-wasm32

This recipe builds `tvb-gdist` as a wasm side module with the emscripten-forge
toolchain. It was verified against the `tvbl-wasm` env (Python 3.13 /
`emscripten-abi 4.0.9`); the resulting `gdist` module passes
`compute_gdist`, `local_gdist_matrix` and `distance_matrix_of_selected_points`
in headless chromium.

## Build

Create a host env with `rattler-build` (native, only used to drive the build):

```sh
micromamba create -n rattler-build -c conda-forge \
  python=3.13 'rattler-build>=0.67,<0.68' git patch curl

# variant.yaml is from emscripten-forge/recipes; it pins the emscripten 4.0.9
# compiler. Downloading it keeps the recipe in sync with that repo.
curl -fsSL https://raw.githubusercontent.com/emscripten-forge/recipes/main/variant.yaml \
  -o variant.yaml
```

Then from this directory:

```sh
~/.micromamba/envs/rattler-build/bin/rattler-build build \
  --recipe . \
  --target-platform emscripten-wasm32 \
  -c https://repo.prefix.dev/emscripten-forge-4x \
  -c https://repo.prefix.dev/emscripten-forge-dev \
  -c https://repo.prefix.dev/conda-forge \
  -c microsoft \
  -m variant.yaml
```

The build itself succeeds (Cython -> C++ via `em++`, ~35 s on an M-series
Mac). The package is written to `output/emscripten-wasm32/`; if the `pytester`
step fails, `rattler-build` moves it to `output/broken/` (the build artifact is
still usable).

## macOS gotcha

The `cross-python_emscripten-wasm32` package's activation script references an
undefined `CTYPES_INIT_FILE` on macOS. That makes `set -e` abort before the
build script runs. Patch the cached package once:

```sh
python3 - <<'PY'
from pathlib import Path
pattern = (
    "Library/Caches/rattler/cache/pkgs/cross-python_emscripten-wasm32-*"
    "/etc/conda/activate.d/activate_z-cross-python_emscripten-wasm32.sh"
)
for path in Path.home().glob(pattern):
    source = path.read_text()
    old = (
        "  sed 's/_sys.platform == \"darwin\"/False/g' \"$CTYPES_INIT_FILE\" "
        "> \"$CTYPES_INIT_FILE.tmp\" && mv \"$CTYPES_INIT_FILE.tmp\" \"$CTYPES_INIT_FILE\""
    )
    new = (
        '  CTYPES_INIT_FILE="${CTYPES_INIT_FILE:-$PREFIX/lib/python$PY_VER/ctypes/__init__.py}"\n'
        + old
    )
    if old in source and "CTYPES_INIT_FILE:-" not in source:
        path.write_text(source.replace(old, new))
        print("patched", path)
PY
```

(Linux CI does not hit this branch.) The proper fix belongs upstream in
emscripten-forge/recipes.

## Install into the test env

`micromamba install` cannot be used to add packages to an existing wasm env
(it tries to run the wasm `python`). Extract the package and copy the files:

```sh
micromamba package extract \
  output/emscripten-wasm32/tvb-gdist-*.conda /tmp/tvb-gdist-extract
cp -R /tmp/tvb-gdist-extract/lib/python3.13/site-packages/* \
  ~/.micromamba/envs/tvbl-wasm/lib/python3.13/site-packages/
printf 'pip\n' > ~/.micromamba/envs/tvbl-wasm/lib/python3.13/site-packages/tvb_gdist-2.9.2.dist-info/INSTALLER
```

Setting `INSTALLER` to `pip` lets empack (`wasm-tests/run.sh`) pack the module.

## What it unlocks / what still doesn't fit

With `gdist` installed, five of the six previously skipped gdist tests pass
(`patterns`, both projection-monitor surface tests, and two of the
`test_simulator_surface` cases). The remaining gdist tests are memory-bound in
the browser: `TestCouplingShape::test_shape` and the two
`test_simulator_surface[..-False]` cases allocate ~1 GiB / tens of MiB on top
of the surface mesh and hit the 2 GiB wasm ceiling, especially when run in the
same process as other tests.

## GitHub Actions

`.github/workflows/deploy.yml` builds this recipe before the JupyterLite build:

```sh
rattler-build build \
  --recipe wasm-tests/recipes/tvb-gdist \
  --target-platform emscripten-wasm32 \
  --test skip \
  --output-dir "$GITHUB_WORKSPACE/local-channel" \
  -c https://repo.prefix.dev/emscripten-forge-4x \
  -c https://repo.prefix.dev/emscripten-forge-dev \
  -c https://repo.prefix.dev/conda-forge \
  -c microsoft \
  -m wasm-tests/recipes/variant.yaml
```

`rattler-build` writes a valid `emscripten-wasm32/repodata.json` next to the
package, so `local-channel/` can be used directly as a conda channel. The
workflow then generates `environment.ci.yml` from `environment.yml` with that
channel added and `tvb-gdist` in `dependencies`, and passes it to
`jupyter lite build --XeusAddon.environment_file=environment.ci.yml`.

`--test skip` is intentional: the package's `pytester` test needs a full
playwright install inside the test environment. The same runtime checks are
covered by the `wasm-tests/run.sh` suite when gdist is installed.

On Linux (GitHub Actions) the macOS `CTYPES_INIT_FILE` activation workaround is
not needed.
