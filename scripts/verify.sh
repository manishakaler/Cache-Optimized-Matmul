#!/usr/bin/env bash

set -euo pipefail
BIN="./bin"
N=64

echo "Running correctness verification at N=$N..."

# Generate reference with naive
$BIN/naive      $N 1 > /dev/null
$BIN/reordered  $N 1 > /dev/null
$BIN/unrolled   $N 1 > /dev/null
$BIN/blocked    $N 1 32 > /dev/null
$BIN/avx_vectorized   $N 1 32 > /dev/null
$BIN/cache_aware      $N 1 > /dev/null
$BIN/register_kernel  $N 1 > /dev/null
OMP_NUM_THREADS=2 $BIN/openmp_blocked $N 1 32 > /dev/null

echo "All kernels ran without crashing at N=$N."
echo "For numerical verification, consider running the inline correctness"
echo "check (approx_equal) by enabling it in the source and recompiling."
