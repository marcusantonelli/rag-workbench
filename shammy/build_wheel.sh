#!/usr/bin/env bash
# Build the Shammy Python wheel and install it into the current Python.
#   pip install maturin   (once; needs a Rust toolchain)
set -euo pipefail
cd "$(dirname "$0")/crates/shammy-py"
PY="${PYTHON:-python}"
rm -rf ../../target/wheels
maturin build --release -i "$PY" --out ../../target/wheels
"$PY" -m pip install --force-reinstall --no-deps ../../target/wheels/shammy-*.whl ${PIP_ARGS:-}
