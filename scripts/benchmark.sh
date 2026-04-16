#!/usr/bin/env bash

set -euo pipefail

BIN="./bin"
RES="./results"
N_SIZES="512 1024 2048"
N_LARGE="4096"
ITERS=3
BLOCK=32
THREADS="1 2 4 8"

mkdir -p "$RES"

# --------------------------------------------------------------------------
# Helper: run perf stat on a single command, append one CSV row
# --------------------------------------------------------------------------
perf_row() {
    local cmd="$1"
    local outfile="$2"
    local extra_cols="${3:-}"   # optional prefix columns (e.g. thread count)

    local tmpf
    tmpf=$(mktemp /tmp/perf_XXXXXX.txt)

    # Run with perf stat; suppress stdout of the program itself
    perf stat -e task-clock,L1-dcache-load-misses,L1-dcache-loads,\
LLC-load-misses,LLC-loads,dTLB-load-misses,dTLB-loads \
        -o "$tmpf" -- bash -c "$cmd" > /dev/null 2>&1 || true

    TIME=$(grep  "seconds time elapsed"    "$tmpf" | awk '{print $1}')
    CPUS=$(awk '/CPUs utilized/ {for(i=1;i<=NF;i++) if($i=="CPUs") print $(i-1)}' "$tmpf")
    L1P=$(grep   "L1-dcache-load-misses"   "$tmpf" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
    LLCP=$(grep  "LLC-load-misses"         "$tmpf" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
    TLBP=$(grep  "dTLB-load-misses"        "$tmpf" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')

    CPUS="${CPUS:-0.000}"; L1P="${L1P:-0.00}"; LLCP="${LLCP:-0.00}"; TLBP="${TLBP:-0.00}"
    rm -f "$tmpf"

    if [[ -n "$extra_cols" ]]; then
        echo "${extra_cols},${TIME},${CPUS},${L1P},${LLCP},${TLBP}" >> "$outfile"
    else
        echo "${TIME},${CPUS},${L1P},${LLCP},${TLBP}" >> "$outfile"
    fi
}

# --------------------------------------------------------------------------
# 1. Single-thread kernels
# --------------------------------------------------------------------------
echo "=== Benchmarking single-thread kernels ==="

for name in naive reordered unrolled; do
    outfile="$RES/results_${name}.csv"
    echo "N,Time(s),CPUs,L1_Miss_%,LLC_Miss_%,dTLB_Miss_%" > "$outfile"
    for N in $N_SIZES; do
        echo "  $name N=$N"
        perf_row "$BIN/$name $N $ITERS" "$outfile" "$N"
    done
done

# blocked needs block size argument
for name in blocked avx_vectorized; do
    outfile="$RES/results_${name}.csv"
    echo "N,Time(s),CPUs,L1_Miss_%,LLC_Miss_%,dTLB_Miss_%" > "$outfile"
    for N in $N_SIZES $N_LARGE; do
        echo "  $name N=$N B=$BLOCK"
        if [[ "$name" == "avx_vectorized" ]]; then
            perf_row "$BIN/$name $N $ITERS $BLOCK" "$outfile" "$N"
        else
            perf_row "$BIN/$name $N $ITERS $BLOCK" "$outfile" "$N"
        fi
    done
done

for name in cache_aware register_kernel; do
    outfile="$RES/results_${name}.csv"
    echo "N,Time(s),CPUs,L1_Miss_%,LLC_Miss_%,dTLB_Miss_%" > "$outfile"
    for N in $N_SIZES $N_LARGE; do
        echo "  $name N=$N"
        perf_row "$BIN/$name $N $ITERS" "$outfile" "$N"
    done
done

# --------------------------------------------------------------------------
# 2. Block size sweep (blocked kernel)
# --------------------------------------------------------------------------
echo "=== Block size sweep ==="
bsweep="$RES/results_block_sweep.csv"
echo "N,BlockSize,Time(s),CPUs,L1_Miss_%,LLC_Miss_%,dTLB_Miss_%" > "$bsweep"
for BS in 16 32 64 128; do
    for N in 1024 2048; do
        echo "  blocked N=$N B=$BS"
        local_tmpf=$(mktemp /tmp/perf_XXXXXX.txt)
        perf stat -e task-clock,L1-dcache-load-misses,L1-dcache-loads,\
LLC-load-misses,LLC-loads,dTLB-load-misses,dTLB-loads \
            -o "$local_tmpf" -- $BIN/blocked $N $ITERS $BS > /dev/null 2>&1 || true
        TIME=$(grep "seconds time elapsed" "$local_tmpf" | awk '{print $1}')
        CPUS=$(awk '/CPUs utilized/ {for(i=1;i<=NF;i++) if($i=="CPUs") print $(i-1)}' "$local_tmpf")
        L1P=$(grep "L1-dcache-load-misses" "$local_tmpf" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
        LLCP=$(grep "LLC-load-misses" "$local_tmpf" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
        TLBP=$(grep "dTLB-load-misses" "$local_tmpf" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
        CPUS="${CPUS:-0.000}"; L1P="${L1P:-0.00}"; LLCP="${LLCP:-0.00}"; TLBP="${TLBP:-0.00}"
        rm -f "$local_tmpf"
        echo "$N,$BS,$TIME,$CPUS,$L1P,$LLCP,$TLBP" >> "$bsweep"
    done
done

# --------------------------------------------------------------------------
# 3. OpenMP thread scaling
# --------------------------------------------------------------------------
echo "=== OpenMP thread scaling ==="
mt_out="$RES/results_openmp.csv"
echo "Threads,N,Time(s),CPUs,L1_Miss_%,LLC_Miss_%,dTLB_Miss_%" > "$mt_out"

for T in $THREADS; do
    export OMP_NUM_THREADS=$T
    for N in $N_SIZES $N_LARGE; do
        echo "  openmp T=$T N=$N"
        tmpf2=$(mktemp /tmp/perf_XXXXXX.txt)
        perf stat -e task-clock,L1-dcache-load-misses,L1-dcache-loads,\
LLC-load-misses,LLC-loads,dTLB-load-misses,dTLB-loads \
            -o "$tmpf2" -- $BIN/openmp_blocked $N $ITERS $BLOCK > /dev/null 2>&1 || true
        TIME=$(grep "seconds time elapsed" "$tmpf2" | awk '{print $1}')
        CPUS=$(awk '/CPUs utilized/ {for(i=1;i<=NF;i++) if($i=="CPUs") print $(i-1)}' "$tmpf2")
        L1P=$(grep "L1-dcache-load-misses" "$tmpf2" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
        LLCP=$(grep "LLC-load-misses" "$tmpf2" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
        TLBP=$(grep "dTLB-load-misses" "$tmpf2" | awk -F'#' '{print $2}' | awk '{gsub(/%/,""); print $1}')
        CPUS="${CPUS:-0.000}"; L1P="${L1P:-0.00}"; LLCP="${LLCP:-0.00}"; TLBP="${TLBP:-0.00}"
        rm -f "$tmpf2"
        echo "$T,$N,$TIME,$CPUS,$L1P,$LLCP,$TLBP" >> "$mt_out"
    done
done

echo ""
echo "Done! Results in $RES/"
echo "Run: python3 scripts/plot_results.py   to generate plots"
