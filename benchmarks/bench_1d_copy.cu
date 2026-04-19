// benchmarks/bench_1d_copy.cu
// Benchmark: 1-D contiguous transfer — cudaMemcpy (pageable/pinned) vs TMA global→shared
//
// Three paths measured:
//   [P] cudaMemcpy — host (pageable) → device
//   [N] cudaMemcpy — host (pinned)   → device
//   [T] TMA        — device global   → shared memory (H100+ only)
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_1d_copy.cu -o bench_1d -Iinclude
// Run:
//   ./bench_1d [csv_out_path]   (default: results/bench_1d_copy.csv)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstring>

// ─── TMA tile geometry ────────────────────────────────────────────────────────
// For float32: inner-dimension limit is 256 bytes → 64 elements.
static constexpr uint32_t TILE_1D = 64;

// ─── Benchmark parameters ─────────────────────────────────────────────────────
static constexpr int WARMUP = 3;
static constexpr int RUNS   = 10;

// ─── cudaMemcpy bandwidth ─────────────────────────────────────────────────────

static float memcpy_bw(void* h, float* d, size_t bytes) {
    GpuTimer t;
    t.begin();
    CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    return t.end_ms();
}

// ─── TMA kernel: global → shared (one tile per loop iteration) ───────────────

__global__ void tma_1d_kernel(
    const __grid_constant__ CUtensorMap tma_map,
    size_t num_tiles,
    float* __restrict__ sink)   // anti-DCE
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TILE_1D];
    __shared__ barrier_t bar;

    if (threadIdx.x == 0) init(&bar, 1);
    __syncthreads();

    int parity   = 0;
    float accum  = 0.f;

    for (size_t tile = (size_t)blockIdx.x; tile < num_tiles;
         tile += (size_t)gridDim.x)
    {
        const int coord = (int)(tile * TILE_1D);

        if (threadIdx.x == 0) {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            uint32_t sptr = __cvta_generic_to_shared(smem);
            uint32_t exp  = TILE_1D * (uint32_t)sizeof(float);

            // Announce expected transaction bytes (also counts as the single arrival)
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bptr), "r"(exp) : "memory");

            // Issue asynchronous bulk copy: global → shared
            asm volatile(
                "cp.async.bulk.tensor.1d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sptr), "l"(&tma_map), "r"(coord), "r"(bptr)
                : "memory");
        }

        // All threads spin-wait for TMA completion
        {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            asm volatile(
                "{\n\t"
                ".reg .pred P;\n\t"
                "TMA1D_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra TMA1D_WAIT_%=;\n\t"
                "}"
                :: "r"(bptr), "r"(parity) : "memory");
        }
        parity ^= 1;
        __syncthreads();

        // Use data to prevent dead-code elimination
        accum += smem[threadIdx.x & (TILE_1D - 1)];
    }

    // Write only if condition is never true; prevents DCE
    if (accum == 3.14159265f) *sink = accum;
#endif
}

// ─── TMA host-side driver ─────────────────────────────────────────────────────

// Returns average time (ms) to TMA-copy the entire d_data array
// from global memory into shared memory (tiled).
static float run_tma_1d(float* d_data, size_t N) {
    size_t num_tiles = N / TILE_1D;
    if (num_tiles == 0) return 0.f;

    // Allocate anti-DCE sink
    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    // Build TMA descriptor
    CUtensorMap tma_map = make_tma_1d_f32(d_data, (uint64_t)N, TILE_1D);

    // Grid: enough blocks to saturate H100 (132 SMs)
    uint32_t blocks  = (uint32_t)std::min((size_t)8192u, num_tiles);
    uint32_t threads = 32;   // 1 warp per block

    // Warmup
    for (int r = 0; r < WARMUP; ++r)
        tma_1d_kernel<<<blocks, threads>>>(tma_map, num_tiles, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        tma_1d_kernel<<<blocks, threads>>>(tma_map, num_tiles, d_sink);
        total += t.end_ms();
    }

    CUDA_CHECK(cudaFree(d_sink));
    return total / RUNS;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_1d_copy.csv";

    // Print device info
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name
              << "  sm_" << prop.major << prop.minor << "\n\n";

    bool tma_ok = tma_supported();
    if (!tma_ok)
        std::cout << "[WARNING] TMA requires sm_90+ (H100). "
                     "TMA column will be skipped.\n\n";

    // Transfer sizes
    const size_t sizes[] = {
        1ULL  << 20,   //   1 MB
        4ULL  << 20,   //   4 MB
        16ULL << 20,   //  16 MB
        64ULL << 20,   //  64 MB
        256ULL<< 20,   // 256 MB
        1ULL  << 30,   //   1 GB
        2ULL  << 30,   //   2 GB
    };
    const int NUM_SIZES = (int)(sizeof(sizes) / sizeof(sizes[0]));

    // Open CSV
    std::ofstream csv(csv_path);
    csv << "benchmark,size_mb,method,avg_time_ms,bw_gbs\n";

    // Print table header
    std::cout << std::string(122, '-') << "\n";
    std::cout << std::left
              << std::setw(10) << "Size(MB)"
              << std::setw(18) << "  P_time(ms)"
              << std::setw(18) << "  Pin_time(ms)"
              << std::setw(18) << "  TMA_time(ms)"
              << std::setw(16) << "  P_BW(GB/s)"
              << std::setw(16) << "  Pin_BW(GB/s)"
              << std::setw(16) << "  TMA_BW(GB/s)"
              << std::setw(10) << "  Pin/P"
              << "\n";
    std::cout << std::string(122, '-') << "\n";

    for (int i = 0; i < NUM_SIZES; ++i) {
        size_t bytes = sizes[i];
        size_t N     = bytes / sizeof(float);
        size_t mb    = bytes >> 20;

        // Allocate device buffer
        float* d_buf = nullptr;
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
            std::cout << std::setw(10) << mb
                      << "  SKIPPED (insufficient VRAM)\n";
            continue;
        }

        // ── Pageable host memory ──────────────────────────────────────────────
        float* h_page = (float*)malloc(bytes);
        if (!h_page) { cudaFree(d_buf); continue; }
        for (size_t j = 0; j < N; ++j) h_page[j] = 1.0f;

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice));

        float sum_p = 0.f;
        for (int r = 0; r < RUNS; ++r) sum_p += memcpy_bw(h_page, d_buf, bytes);
        float avg_p = sum_p / RUNS;
        float bw_p  = (bytes / 1e9f) / (avg_p / 1e3f);

        // ── Pinned host memory ────────────────────────────────────────────────
        float* h_pin = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_pin, bytes, cudaHostAllocDefault));
        for (size_t j = 0; j < N; ++j) h_pin[j] = 1.0f;

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));

        float sum_pin = 0.f;
        for (int r = 0; r < RUNS; ++r) sum_pin += memcpy_bw(h_pin, d_buf, bytes);
        float avg_pin = sum_pin / RUNS;
        float bw_pin  = (bytes / 1e9f) / (avg_pin / 1e3f);

        // Keep data on device for TMA benchmark
        CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));

        // ── TMA: global → shared ──────────────────────────────────────────────
        float avg_tma = 0.f, bw_tma = 0.f;
        if (tma_ok && N >= TILE_1D) {
            avg_tma = run_tma_1d(d_buf, N);
            bw_tma  = (bytes / 1e9f) / (avg_tma / 1e3f);
        }

        // Print table row
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << mb
                  << std::setw(18) << avg_p
                  << std::setw(18) << avg_pin
                  << std::setw(18) << (tma_ok ? avg_tma : -1.f)
                  << std::setw(16) << bw_p
                  << std::setw(16) << bw_pin
                  << std::setw(16) << (tma_ok ? bw_tma : -1.f)
                  << std::setw(10) << (bw_pin / bw_p)
                  << "\n";

        // Write CSV
        csv << "1d_copy," << mb << ",cudaMemcpy_pageable,"
            << avg_p << "," << bw_p << "\n";
        csv << "1d_copy," << mb << ",cudaMemcpy_pinned,"
            << avg_pin << "," << bw_pin << "\n";
        if (tma_ok && N >= TILE_1D)
            csv << "1d_copy," << mb << ",TMA_global_shared,"
                << avg_tma << "," << bw_tma << "\n";

        free(h_page);
        CUDA_CHECK(cudaFreeHost(h_pin));
        CUDA_CHECK(cudaFree(d_buf));
    }

    std::cout << std::string(122, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  P   = pageable host memory (cudaMemcpy H2D)\n"
              << "  Pin = pinned host memory   (cudaMemcpy H2D)\n"
              << "  TMA = Tensor Memory Accelerator, global→shared"
                 " (device-side, H100+ only)\n"
              << "        Bandwidth reflects HBM3→L2→SMEM path,"
                 " not PCIe.\n"
              << "\nCSV written to: " << csv_path << "\n";

    return 0;
}
