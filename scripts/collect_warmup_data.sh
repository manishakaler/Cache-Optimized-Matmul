#!/usr/bin/env bash

set -euo pipefail

BIN="./bin/warmup"
ITERS=5

# Array sizes in number of elements (long = 8 bytes each)
# Covers L1 → L2 → L3 → DRAM transition points
SIZES="1024 4096 16384 65536 262144 1048576 4194304 16777216 33554432 67108864"
# Bytes:  8KB  32KB  128KB  512KB    2MB      8MB      32MB     128MB    256MB   512MB

STRIDES="1 2 4 8 16 32 64"

echo "=== Building warmup ==="
make warmup

# --------------------------------------------------------------------------
# access_time_data.csv
# --------------------------------------------------------------------------
AOUT="results/access_time_data.csv"
mkdir -p results
echo "access_mode,stride,array_size_in_elements,iterations,runtime_in_seconds" > "$AOUT"

echo ""
echo "=== Sequential access ==="
for SZ in $SIZES; do
    T=$( { time $BIN $SZ sequential $ITERS ; } 2>&1 | grep real | awk '{print $2}' | \
         sed 's/m/:/g' | awk -F: '{print $1*60 + $2}' )
    # Use /usr/bin/time for cleaner output
    T=$( /usr/bin/time -f "%e" $BIN $SZ sequential $ITERS 2>&1 | tail -1 )
    echo "sequential,1,$SZ,$ITERS,$T" >> "$AOUT"
    echo "  sz=$SZ -> ${T}s"
done

echo ""
echo "=== Random access ==="
for SZ in $SIZES; do
    T=$( /usr/bin/time -f "%e" $BIN $SZ random $ITERS 2>&1 | tail -1 )
    echo "random,1,$SZ,$ITERS,$T" >> "$AOUT"
    echo "  sz=$SZ -> ${T}s"
done

echo ""
echo "=== Strided access ==="
for STRIDE in $STRIDES; do
    for SZ in $SIZES; do
        T=$( /usr/bin/time -f "%e" $BIN $SZ strided $ITERS $STRIDE 2>&1 | tail -1 )
        echo "strided,$STRIDE,$SZ,$ITERS,$T" >> "$AOUT"
        echo "  stride=$STRIDE sz=$SZ -> ${T}s"
    done
done

# --------------------------------------------------------------------------
# cache_miss_data.csv  (perf stat on strided only)
# --------------------------------------------------------------------------
COUT="results/cache_miss_data.csv"
echo "stride,array_size_in_elements,iterations,l1_misses,l2_misses,l3_misses,runtime_in_seconds" > "$COUT"

echo ""
echo "=== Perf cache miss data (strided only) ==="
for STRIDE in $STRIDES; do
    for SZ in $SIZES; do
        TMPF=$(mktemp /tmp/perf_XXXXXX.txt)
        perf stat \
            -e L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses \
            -o "$TMPF" -- $BIN $SZ strided $ITERS $STRIDE > /dev/null 2>&1 || true

        TIME=$(grep "seconds time elapsed" "$TMPF" | awk '{print $1}')
        L1=$(  grep "L1-dcache-load-misses" "$TMPF" | awk '{print $1}' | tr -d ',')
        L2=$(  grep "l2_rqsts.miss"          "$TMPF" | awk '{print $1}' | tr -d ',')
        L3=$(  grep "LLC-load-misses"         "$TMPF" | awk '{print $1}' | tr -d ',')

        L1="${L1:-0}"; L2="${L2:-0}"; L3="${L3:-0}"; TIME="${TIME:-0}"
        echo "$STRIDE,$SZ,$ITERS,$L1,$L2,$L3,$TIME" >> "$COUT"
        echo "  stride=$STRIDE sz=$SZ  L1=$L1 L2=$L2 L3=$L3 t=${TIME}s"
        rm -f "$TMPF"
    done
done

echo ""
echo "Done!"
echo "  results/access_time_data.csv"
echo "  results/cache_miss_data.csv"
echo ""
echo "Now run: python3 scripts/plot_results.py"
