#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS="${1:-${ROOT}/results}"
mkdir -p "$RESULTS"

NVCC="nvcc -arch=sm_90a -std=c++17 -O3 -lcuda -I${ROOT}/include"
BIN="/tmp/tma_bench_$$"
mkdir -p "$BIN"
trap 'rm -rf "$BIN"' EXIT

echo "=== Building ==="
$NVCC "${ROOT}/benchmarks/bench_1d_copy.cu"    -o "${BIN}/b1d"    && echo "  [OK] 1d"
$NVCC "${ROOT}/benchmarks/bench_2d_stride.cu"  -o "${BIN}/b2d"    && echo "  [OK] 2d"
$NVCC "${ROOT}/benchmarks/bench_gemm.cu"       -o "${BIN}/bgemm"  && echo "  [OK] gemm"
$NVCC "${ROOT}/benchmarks/bench_overlap.cu"    -o "${BIN}/bovlp"  && echo "  [OK] overlap"
$NVCC "${ROOT}/benchmarks/bench_small_xfer.cu" -o "${BIN}/bsmall" && echo "  [OK] small"

echo ""
echo "=== Running ==="
for b in b1d b2d bgemm bovlp bsmall; do
    name="${b#b}"
    echo ""; echo "--- ${name} ---"
    "${BIN}/${b}" "${RESULTS}/bench_${name}.csv"
done

# Aggregate
S="${RESULTS}/summary.csv"; first=1
for f in "${RESULTS}"/bench_*.csv; do
    [ -f "$f" ] || continue
    if [ "$first" -eq 1 ]; then cat "$f" > "$S"; first=0; else tail -n +2 "$f" >> "$S"; fi
done
echo ""; echo "=== Summary: ${S} ($(wc -l < "$S") lines) ==="
