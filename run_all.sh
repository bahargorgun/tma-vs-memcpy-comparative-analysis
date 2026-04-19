#!/usr/bin/env bash
# run_all.sh  —  Build all benchmarks, run them, and aggregate results into
#                a single summary CSV: results/summary.csv
#
# Usage:
#   ./run_all.sh [results_dir]
#
# Requirements:
#   CUDA >= 12.0; H100 (sm_90a) recommended for TMA paths.
#
# Output files in results/:
#   bench_1d_copy.csv
#   bench_2d_stride.csv
#   bench_overlap.csv
#   bench_gemm.csv
#   bench_small_xfer.csv
#   summary.csv          <- aggregated, all rows from the above

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="${ROOT}/benchmarks"
INCLUDE_DIR="${ROOT}/include"
RESULTS_DIR="${1:-${ROOT}/results}"
mkdir -p "$RESULTS_DIR"

# ── Detect architecture support ──────────────────────────────────────────────
ARCH_FLAG="-arch=sm_90a"
if ! nvcc $ARCH_FLAG --dryrun /dev/null -o /dev/null 2>/dev/null; then
    echo "[INFO] sm_90a not supported by this compiler; using sm_80."
    echo "       TMA benchmarks will be skipped at runtime on non-H100 GPUs."
    ARCH_FLAG="-arch=sm_80"
fi

NVCC="nvcc ${ARCH_FLAG} -std=c++17 -O3 -lcuda -I${INCLUDE_DIR}"

BINDIR="/tmp/tma_benchmarks_$$"
mkdir -p "$BINDIR"
trap 'rm -rf "$BINDIR"' EXIT

# ── Build ────────────────────────────────────────────────────────────────────
echo "===  Building benchmarks  ==="
$NVCC "${BENCH_DIR}/bench_1d_copy.cu"    -o "${BINDIR}/bench_1d"
echo "  [OK] bench_1d"
$NVCC "${BENCH_DIR}/bench_2d_stride.cu"  -o "${BINDIR}/bench_2d"
echo "  [OK] bench_2d"
$NVCC "${BENCH_DIR}/bench_overlap.cu"    -o "${BINDIR}/bench_overlap"
echo "  [OK] bench_overlap"
$NVCC "${BENCH_DIR}/bench_gemm.cu"       -o "${BINDIR}/bench_gemm"
echo "  [OK] bench_gemm"
$NVCC "${BENCH_DIR}/bench_small_xfer.cu" -o "${BINDIR}/bench_small"
echo "  [OK] bench_small"

# ── Run ──────────────────────────────────────────────────────────────────────
echo ""
echo "===  Running benchmarks  ==="

run_bench() {
    local bin="$1" csv="$2" label="$3"
    echo ""
    echo "--- ${label} ---"
    "${bin}" "${csv}"
}

run_bench "${BINDIR}/bench_1d"      "${RESULTS_DIR}/bench_1d_copy.csv"    "1-D copy (pageable/pinned/TMA)"
run_bench "${BINDIR}/bench_2d"      "${RESULTS_DIR}/bench_2d_stride.csv"  "2-D strided (cudaMemcpy2D/TMA)"
run_bench "${BINDIR}/bench_overlap" "${RESULTS_DIR}/bench_overlap.csv"    "Overlap pipeline (seq/async/TMA)"
run_bench "${BINDIR}/bench_gemm"    "${RESULTS_DIR}/bench_gemm.csv"       "GEMM tile-load (global/TMA)"
run_bench "${BINDIR}/bench_small"   "${RESULTS_DIR}/bench_small_xfer.csv" "Small transfers (latency)"

# ── Aggregate into summary.csv ───────────────────────────────────────────────
SUMMARY="${RESULTS_DIR}/summary.csv"
echo ""
echo "===  Aggregating results → ${SUMMARY}  ==="

# Write header once, then append data rows (strip headers from subsequent files)
first=1
for f in \
    "${RESULTS_DIR}/bench_1d_copy.csv" \
    "${RESULTS_DIR}/bench_2d_stride.csv" \
    "${RESULTS_DIR}/bench_overlap.csv" \
    "${RESULTS_DIR}/bench_gemm.csv" \
    "${RESULTS_DIR}/bench_small_xfer.csv"
do
    if [ ! -f "$f" ]; then continue; fi
    if [ "$first" -eq 1 ]; then
        cat "$f" > "$SUMMARY"
        first=0
    else
        tail -n +2 "$f" >> "$SUMMARY"
    fi
done

echo "  Written: $(wc -l < "$SUMMARY") lines"
echo ""
echo "===  Done  ===  Results in: ${RESULTS_DIR}/"
