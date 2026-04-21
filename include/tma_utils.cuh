// include/tma_utils.cuh
// Shared utilities for TMA vs cp.async benchmarks.
// Requires CUDA >= 12.0 and sm_90a (NVIDIA H100 Hopper) for TMA paths.
// cp.async paths require sm_80+ (A100/H100).

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

    void begin()   { CUDA_CHECK(cudaEventRecord(start)); }
    float end_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ─── Runtime capability check ─────────────────────────────────────────────────

inline bool tma_supported() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major,
               cudaDevAttrComputeCapabilityMajor, device));
    return major >= 9;
}

// ─── TMA descriptor builders ──────────────────────────────────────────────────

// 1-D: N float32 eleman, tile_elems genisliginde tile
inline CUtensorMap make_tma_1d_f32(void* ptr, uint64_t N, uint32_t tile_elems) {
    CUtensorMap map{};
    uint64_t globalDim[1]     = { N };
    uint64_t globalStrides[1] = { N * sizeof(float) };
    uint32_t boxDim[1]        = { tile_elems };
    uint32_t elemStrides[1]   = { 1 };
    CU_CHECK(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1,
        ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

// 2-D: cols x rows float32 matris
inline CUtensorMap make_tma_2d_f32(
    void* ptr, uint64_t cols, uint64_t rows,
    uint64_t pitch_bytes, uint32_t tile_cols, uint32_t tile_rows)
{
    CUtensorMap map{};
    uint64_t globalDim[2]     = { cols, rows };
    uint64_t globalStrides[1] = { pitch_bytes };
    uint32_t boxDim[2]        = { tile_cols, tile_rows };
    uint32_t elemStrides[2]   = { 1, 1 };
    CU_CHECK(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}
