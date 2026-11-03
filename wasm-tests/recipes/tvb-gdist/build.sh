#!/bin/bash
set -euxo pipefail

export LDFLAGS="-s MODULARIZE=1 -s LINKABLE=1 -s EXPORT_ALL=1 -s WASM=1 -s SIDE_MODULE=1 -s WASM_BIGINT -s ALLOW_MEMORY_GROWTH=1"
export PIP_CERT="$(${PYTHON} -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="$PIP_CERT"
${PYTHON} -m pip install . -vv --no-deps --no-build-isolation --no-index
