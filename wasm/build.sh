#!/usr/bin/env bash
# Source an Emscripten 4.0.9 environment first; requires nightly + rust-src.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYO3_CONFIG_FILE="$PWD/wasm/pyo3-config.txt"
export CARGO_PROFILE_DEV_DEBUG=0
export RUSTFLAGS='-C target-feature=+simd128 -C relocation-model=pic -C link-arg=-sSIDE_MODULE=2 -C link-arg=-sWASM_BIGINT'
cargo +nightly build --release --locked -Zbuild-std --target wasm32-unknown-emscripten
python3 wasm/package.py
