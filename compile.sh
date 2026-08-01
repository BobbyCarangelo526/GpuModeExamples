#!/usr/bin/env bash
# Compile + run a standalone CUDA program. Usage: ./compile.sh pmpp/hello.cu
set -e
src="$1"
out="${src%.cu}"
nvcc -O2 "$src" -o "$out"
echo "--- running $out ---"
"./$out"
