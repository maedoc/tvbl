#!/usr/bin/env bash
#
# Run the upstream tvb_library test suite in the emscripten-wasm32 runtime
# used by JupyterLite, under headless chromium via pyjs-code-runner.
#
# Usage:
#   ./wasm-tests/run.sh
#
# Useful overrides:
#   TVBL_WASM_ENV       existing emscripten env prefix (default: $MAMBA_ROOT_PREFIX/envs/tvbl-wasm)
#   TVB_LIBRARY_SRC     local tvb_library checkout (default: ../tvb/tvb_library)
#   TVB_LIBRARY_SPEC    pip requirement to use when TVB_LIBRARY_SRC is absent
#   PYJS_RUNNER         pyjs_code_runner binary
#   HOST_PYTHON         host Python used only to build/copy pure-Python wheels
#   TVBL_SKIP_ENV=1     do not create/recreate the wasm env
#   TVBL_SKIP_INSTALL=1 do not reinstall tvb_library/tvb-data
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
export MAMBA_ROOT_PREFIX
ENV_PREFIX="${TVBL_WASM_ENV:-$MAMBA_ROOT_PREFIX/envs/tvbl-wasm}"
PYJS_RUNNER="${PYJS_RUNNER:-$HOME/.venvs/pyjs-runner/bin/pyjs_code_runner}"
HOST_PYTHON="${HOST_PYTHON:-python3}"
CACHE_DIR="${PYJS_CACHE_DIR:-$ROOT/wasm-tests/.cache}"
PATCHED_PYJS="${PYJS_PATCH_DIR:-$ROOT/wasm-tests/.pyjs-patched}"
TVB_LIBRARY_SPEC="${TVB_LIBRARY_SPEC:-tvb-library @ git+https://github.com/maedoc/tvb.git#subdirectory=tvb_library}"

if [ -d "${TVB_LIBRARY_SRC:-$ROOT/../tvb/tvb_library}" ]; then
    TVB_LIBRARY_SRC="${TVB_LIBRARY_SRC:-$ROOT/../tvb/tvb_library}"
else
    TVB_LIBRARY_SRC=""
fi

if [ ! -d "$ENV_PREFIX" ] && [ "${TVBL_SKIP_ENV:-0}" != 1 ]; then
    if ! command -v micromamba >/dev/null 2>&1; then
        echo "micromamba is required to create $ENV_PREFIX (see README.md)" >&2
        exit 1
    fi
    echo "creating emscripten env $ENV_PREFIX" >&2
    micromamba create -y -p "$ENV_PREFIX" --platform emscripten-wasm32 \
        -c https://repo.prefix.dev/emscripten-forge-4x \
        -c https://repo.prefix.dev/emscripten-forge-dev \
        -c https://repo.prefix.dev/conda-forge \
        python numpy scipy matplotlib autograd tqdm scikit-learn ipywidgets \
        joblib setuptools pandas numba pytest pytest-benchmark pyjs \
        networkx numexpr mako deprecated six autopep8 docutils packaging \
        pip h5py
fi

if [ ! -x "$PYJS_RUNNER" ]; then
    echo "pyjs-code-runner not found at $PYJS_RUNNER" >&2
    echo "install it with: uv venv ~/.venvs/pyjs-runner && uv pip install --python ~/.venvs/pyjs-runner/bin/python pyjs-code-runner && ~/.venvs/pyjs-runner/bin/playwright install chromium" >&2
    exit 1
fi

SITE_PACKAGES="$(find "$ENV_PREFIX/lib" -maxdepth 2 -type d -name site-packages | head -1)"
if [ -z "$SITE_PACKAGES" ]; then
    echo "could not find site-packages under $ENV_PREFIX/lib" >&2
    exit 1
fi

if [ "${TVBL_SKIP_INSTALL:-0}" != 1 ]; then
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    echo "installing tvb_library into $SITE_PACKAGES" >&2
    if command -v uv >/dev/null 2>&1; then
        PIP=(uv pip install --python "$HOST_PYTHON" --no-deps)
    else
        PIP=("$HOST_PYTHON" -m pip install --no-deps)
    fi
    if [ -n "$TVB_LIBRARY_SRC" ]; then
        "${PIP[@]}" --target "$TMP/tvb" "$TVB_LIBRARY_SRC"
    else
        "${PIP[@]}" --target "$TMP/tvb" "$TVB_LIBRARY_SPEC"
    fi
    "${PIP[@]}" --target "$TMP/tvb-data" tvb-data

    rm -rf "$SITE_PACKAGES/tvb" "$SITE_PACKAGES/tvb_data" \
        "$SITE_PACKAGES"/tvb_library-*.dist-info "$SITE_PACKAGES"/tvb_data-*.dist-info
    cp -R "$TMP/tvb/." "$SITE_PACKAGES/"
    cp -R "$TMP/tvb-data/." "$SITE_PACKAGES/"
    # empack only sees pip packages when INSTALLER says "pip".
    for dist in "$SITE_PACKAGES"/tvb_library-*.dist-info "$SITE_PACKAGES"/tvb_data-*.dist-info; do
        printf 'pip\n' > "$dist/INSTALLER"
    done
fi

# pyjs dlopen()s cpython extension modules RTLD_LOCAL, but numba/llvmlite
# JIT-compile side modules that import numba's NRT_* symbols and need
# RTLD_GLOBAL.  Patch a private copy of the runtime instead of the env.
rm -rf "$PATCHED_PYJS"
mkdir -p "$PATCHED_PYJS"
cp "$ENV_PREFIX/lib_js/pyjs/pyjs_runtime_browser.js" \
   "$ENV_PREFIX/lib_js/pyjs/pyjs_runtime_browser.wasm" "$PATCHED_PYJS/"
"$HOST_PYTHON" - "$PATCHED_PYJS/pyjs_runtime_browser.js" <<'PY'
import sys
path = sys.argv[1]
source = open(path).read()
old = 'return"local"'
new = 'return"global"'
if old not in source:
    raise SystemExit("pyjs runtime patch point changed; inspect libraryType()")
open(path, "w").write(source.replace(old, new))
PY

# The env cache is keyed by package metadata, not by files we copied in, so
# drop the tvb package tarballs before every run.
mkdir -p "$CACHE_DIR"
rm -f "$CACHE_DIR"/tvb_library-*.tar.gz "$CACHE_DIR"/tvb_data-*.tar.gz

# The wasm runtime is capped at 2 GiB.  A single pytest process running the
# whole suite slowly accumulates memory (numba JIT code, simulation caches)
# and starts failing late with MemoryError.  Running each directory in its own
# browser process keeps peak memory low at the cost of a short bootstrap per
# chunk.
CHUNK_FILE="$ROOT/wasm-tests/chunk.txt"
STATUS=0

run_chunk() {
    local name="$1"
    shift
    printf '%s\n' "$@" > "$CHUNK_FILE"
    echo "== tvb_library tests: $name" >&2
    if ! "$PYJS_RUNNER" run script browser-main \
        --conda-env "$ENV_PREFIX" \
        --script /work/run_wasm_tests.py \
        --mount "$ROOT/wasm-tests:/work" \
        --pyjs-dir "$PATCHED_PYJS" \
        --file-filter "$ROOT/wasm-tests/empack_config.yaml" \
        --env-cache-dir "$CACHE_DIR" \
        --headless \
        -w /home/web_user; then
        STATUS=1
    fi
}

run_chunk basic basic
run_chunk datatypes datatypes
run_chunk "simulator (top-level)" \
    simulator \
    --ignore=simulator/backend \
    --ignore=simulator/hybrid
run_chunk "simulator/backend" simulator/backend
run_chunk "simulator/hybrid" simulator/hybrid

exit "$STATUS"
