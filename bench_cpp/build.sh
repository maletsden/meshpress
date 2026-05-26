#!/usr/bin/env bash
# Linux build script for the STRIDE / DGF / meshopt bench binaries.
# Equivalent to the build_*.bat scripts used on Windows.

set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"
NVCC="$CUDA_HOME/bin/nvcc"

if [ ! -x "$NVCC" ]; then
    echo "nvcc not found at $NVCC. Set CUDA_HOME or install the CUDA Toolkit." >&2
    exit 1
fi

DGF_SDK="../third_party/DGF-SDK"
MESHOPT="../third_party/meshoptimizer"

if [ ! -d "$DGF_SDK" ]; then
    echo "Submodule $DGF_SDK is empty. Run:" >&2
    echo "    git submodule update --init --recursive" >&2
    exit 1
fi

mkdir -p ./build_out

# DGF reference decoder bench
"$NVCC" -O3 -std=c++17 -arch=sm_80 \
    -I "$DGF_SDK/include" \
    dgf_decode_bench.cu \
    -L "$DGF_SDK/build/lib" -lDGF \
    -o build_out/dgf_decode_bench

# STRIDE (dup) decoder bench
"$NVCC" -O3 -std=c++17 -arch=sm_80 \
    stride_dup_decode_bench.cu \
    -o build_out/stride_dup_decode_bench

# STRIDE earlier-variant bench
"$NVCC" -O3 -std=c++17 -arch=sm_80 \
    stride_decode_bench.cu \
    -o build_out/stride_decode_bench

# meshopt CPU shim (shared library used by the Python driver)
"$CXX" -O3 -std=c++17 -shared -fPIC \
    meshopt_shim.cpp \
    -o build_out/meshopt_shim.so

echo "Built binaries under bench_cpp/build_out/"
