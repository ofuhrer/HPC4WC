#!/usr/bin/env bash

# Go to project root
cd "$(dirname "$0")/.."

# Binaries and parameters
NAIVE_BIN="./build/test_mnist"
STENCIL_BIN="./build/test_mnist_stencil"
GPU_BIN="./build/test_mnist_gpu"
TOTAL=10000
DEFAULT_CHUNK=10000

# Helper: measure and return execution time (banner to stderr)
measure() {
  local label=$1; shift
  echo "Benchmarking ${label}..." >&2
  local start=$(date +%s.%N)
  "$@" > /dev/null
  local end=$(date +%s.%N)
  echo "$(echo "$end - $start" | bc)"
}

# 1) CPU: Naive
D1=$(measure "Naive CNN on ${TOTAL} samples" "$NAIVE_BIN")

# 2) CPU: Stencil
D2=$(measure "Stencil CNN on ${TOTAL} samples" "$STENCIL_BIN")

# 3) GPU (single chunk)
D3=$(measure "GPU CNN on ${TOTAL} samples (chunk=${DEFAULT_CHUNK})" \
     "$GPU_BIN" $TOTAL $DEFAULT_CHUNK)

# Summary
echo
echo "---------------- Inference Time Summary ----------------"
printf "Naive     : %8.3f s\n" "$D1"
printf "Stencil   : %8.3f s\n" "$D2"
printf "GPU : %8.3f s\n" "$D3"
echo "--------------------------------------------------------"
