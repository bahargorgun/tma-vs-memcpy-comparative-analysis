// include/tma_utils.cuh
// Shared utilities for TMA vs cudaMemcpy benchmarks.
// Requires CUDA >= 12.0 and sm_90a (NVIDIA H100 Hopper) for TMA paths.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// ─── Error-checking macros ────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CU_CHECK(expr)                                                       \
    do {                                                                     \
        CUresult _e = (expr);                                                \
        if (_e != CUDA_SUCCESS) {                                            \
            const char* _s = nullptr;                                        \
            cuGetErrorString(_e, &_s);                                       \
            fprintf(stderr, "Driver error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, _s ? _s : "unknown");               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ─── GPU timer ────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&start));
                  CUDA_CHECK(cudaEventCreate(&stop)); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void begin()       { CUDA_CHECK(cudaEventRecord(start)); }
    float end_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ─── TMA descriptor builders ──────────────────────────────────────────────────
// These are pure CPU functions; no CUDA context required.

// Build a 1-D TMA descriptor for a contiguous float32 array of N elements
// with tile width tile_elems.
// Constraint: tile_elems must be a multiple of 4 (≥ 4) and ≤ 256.
inline CUtensorMap make_tma_1d_f32(void* ptr, uint64_t N, uint32_t tile_elems) {
    CUtensorMap map{};
    uint64_t g[1] = { N };
    uint32_t b[1] = { tile_elems };
    CU_CHECK(cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        1,          // rank
        ptr,        // global base pointer
        g,          // global dimension array [N]
        nullptr,    // global strides (null → element-size stride for rank-1)
        b,          // box (tile) dimensions
        nullptr,    // element strides (null → all 1)
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

// Build a 2-D TMA descriptor for a float32 matrix.
//   cols        : number of columns (fast/inner dimension, x)
//   rows        : number of rows    (slow/outer dimension, y)
//   pitch_bytes : byte stride between rows (≥ cols * sizeof(float))
//   tile_cols   : tile width  in elements (≤ 64 for float32)
//   tile_rows   : tile height in elements (≤ 256)
inline CUtensorMap make_tma_2d_f32(
    void*    ptr,
    uint64_t cols,  uint64_t rows,
    uint64_t pitch_bytes,
    uint32_t tile_cols, uint32_t tile_rows)
{
    CUtensorMap map{};
    // Dimension order: [0] = fast (cols), [1] = slow (rows)
    uint64_t g[2]      = { cols, rows };
    uint64_t stride[1] = { pitch_bytes };  // byte stride between rows (rank-1 strides)
    uint32_t b[2]      = { tile_cols, tile_rows };
    CU_CHECK(cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,          // rank
        ptr,
        g,
        stride,
        b,
        nullptr,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

// ─── Runtime capability check ─────────────────────────────────────────────────

// Returns true if the current device supports sm_90 (TMA / Hopper).
inline bool tma_supported() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major,
               cudaDevAttrComputeCapabilityMajor, device));
    return major >= 9;
}
