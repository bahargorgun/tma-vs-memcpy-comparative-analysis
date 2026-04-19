#!/usr/bin/env bash
# run.sh  —  Build and run one benchmark at a time.
#
# Usage:
#   ./run.sh <benchmark> [csv_output]
#
# <benchmark> is one of:
#   1d      bench_1d_copy.cu        (1-D H2D bandwidth: pageable/pinned/TMA)
#   2d      bench_2d_stride.cu      (2-D strided: cudaMemcpy2D vs TMA)
#   overlap bench_overlap.cu        (compute-transfer overlap: seq/async/TMA)
#   gemm    bench_gemm.cu           (GEMM tile-load: global vs TMA)
#   small   bench_small_xfer.cu     (small-transfer latency: pageable/pinned/async/TMA)
#
# Requirements:
#   CUDA >= 12.0, GPU with sm_90a (H100) for TMA paths (non-H100 GPUs skip TMA)
#
# Example:
#   ./run.sh 1d results/my_1d.csv

set -euo pipefail

BENCH_DIR="$(dirname "$0")/benchmarks"
INCLUDE_DIR="$(dirname "$0")/include"
RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"

# Compiler flags common to all benchmarks
NVCC_FLAGS="-arch=sm_90a -std=c++17 -O3 -lcuda -I${INCLUDE_DIR}"
# Fallback for pre-Hopper GPUs (still compiles, TMA paths skip at runtime)
NVCC_FLAGS_COMPAT="-arch=sm_80 -std=c++17 -O3 -lcuda -I${INCLUDE_DIR}"

# Detect whether sm_90a is supported by trying a quick query
ARCH_FLAG="-arch=sm_90a"
if ! nvcc $ARCH_FLAG --dryrun /dev/null -o /dev/null 2>/dev/null; then
    echo "[INFO] Compiler does not support sm_90a; falling back to sm_80."
    echo "       TMA paths will be skipped at runtime on non-H100 devices."
    ARCH_FLAG="-arch=sm_80"
fi

NVCC="nvcc ${ARCH_FLAG} -std=c++17 -O3 -lcuda -I${INCLUDE_DIR}"

TARGET="${1:-all}"
CSV="${2:-}"

compile_and_run() {
    local src="$1"
    local bin="$2"
    local csv_arg="${3:-}"

    echo ""
    echo "=== Compiling ${src} ==="
    $NVCC "${BENCH_DIR}/${src}" -o "${bin}"
    echo "=== Running ${bin} ==="
    if [ -n "$csv_arg" ]; then
        "./${bin}" "$csv_arg"
    else
        "./${bin}"
    fi
}

case "$TARGET" in
    1d)
        compile_and_run "bench_1d_copy.cu"    "bench_1d"    "${CSV:-${RESULTS_DIR}/bench_1d_copy.csv}"
        ;;
    2d)
        compile_and_run "bench_2d_stride.cu"  "bench_2d"    "${CSV:-${RESULTS_DIR}/bench_2d_stride.csv}"
        ;;
    overlap)
        compile_and_run "bench_overlap.cu"    "bench_overlap" "${CSV:-${RESULTS_DIR}/bench_overlap.csv}"
        ;;
    gemm)
        compile_and_run "bench_gemm.cu"       "bench_gemm"  "${CSV:-${RESULTS_DIR}/bench_gemm.csv}"
        ;;
    small)
        compile_and_run "bench_small_xfer.cu" "bench_small" "${CSV:-${RESULTS_DIR}/bench_small_xfer.csv}"
        ;;
    all)
        compile_and_run "bench_1d_copy.cu"    "bench_1d"    "${RESULTS_DIR}/bench_1d_copy.csv"
        compile_and_run "bench_2d_stride.cu"  "bench_2d"    "${RESULTS_DIR}/bench_2d_stride.csv"
        compile_and_run "bench_overlap.cu"    "bench_overlap" "${RESULTS_DIR}/bench_overlap.csv"
        compile_and_run "bench_gemm.cu"       "bench_gemm"  "${RESULTS_DIR}/bench_gemm.csv"
        compile_and_run "bench_small_xfer.cu" "bench_small" "${RESULTS_DIR}/bench_small_xfer.csv"
        echo ""
        echo "=== All benchmarks complete. CSVs in ${RESULTS_DIR}/ ==="
        ;;
    *)
        echo "Unknown benchmark: $TARGET"
        echo "Use: 1d | 2d | overlap | gemm | small | all"
        exit 1
        ;;
esac
