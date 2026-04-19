// benchmarks/bench_small_xfer.cu
// Benchmark: small-transfer latency regime
//
// Small copies expose per-launch overhead and latency, not peak bandwidth.
// Sizes from 1 KB to 4 MB, comparing:
//   [P]   cudaMemcpy pageable
//   [N]   cudaMemcpy pinned
//   [AS]  cudaMemcpyAsync (pinned, with stream sync)
//   [TMA] TMA 1-D global→shared  (H100+ only)
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_small_xfer.cu -o bench_small -Iinclude
// Run:
//   ./bench_small [csv_out_path]   (default: results/bench_small_xfer.csv)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

// ─── TMA tile geometry ────────────────────────────────────────────────────────
static constexpr uint32_t TILE_1D = 64;   // float32: 256 bytes per TMA call

// ─── Benchmark parameters ─────────────────────────────────────────────────────
static constexpr int WARMUP = 5;
static constexpr int RUNS   = 50;   // many runs for reliable small-latency estimate

// ─── TMA 1D kernel (same as bench_1d_copy, factored for reuse) ────────────────

__global__ void tma_small_kernel(
    const __grid_constant__ CUtensorMap tma_map,
    size_t num_tiles,
    float* __restrict__ sink)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TILE_1D];
    __shared__ barrier_t bar;

    if (threadIdx.x == 0) init(&bar, 1);
    __syncthreads();

    int parity  = 0;
    float accum = 0.f;

    for (size_t tile = (size_t)blockIdx.x; tile < num_tiles;
         tile += (size_t)gridDim.x)
    {
        const int coord = (int)(tile * TILE_1D);

        if (threadIdx.x == 0) {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            uint32_t sptr = __cvta_generic_to_shared(smem);
            uint32_t exp  = TILE_1D * (uint32_t)sizeof(float);

            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bptr), "r"(exp) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.1d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sptr), "l"(&tma_map), "r"(coord), "r"(bptr)
                : "memory");
        }

        {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "SM_TMA_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra SM_TMA_WAIT_%=;\n\t"
                "}"
                :: "r"(bptr), "r"(parity) : "memory");
        }
        parity ^= 1;
        __syncthreads();
        accum += smem[threadIdx.x & (TILE_1D - 1)];
    }

    if (accum == 3.14159265f) *sink = accum;
#endif
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_small_xfer.csv";

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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    // Sizes: 1 KB → 4 MB (powers of 4 for clean log scale)
    const size_t sizes[] = {
        1ULL   << 10,   //   1 KB
        4ULL   << 10,   //   4 KB
        16ULL  << 10,   //  16 KB
        64ULL  << 10,   //  64 KB
        256ULL << 10,   // 256 KB
        1ULL   << 20,   //   1 MB
        4ULL   << 20,   //   4 MB
    };
    const int NUM_SIZES = (int)(sizeof(sizes) / sizeof(sizes[0]));

    std::ofstream csv(csv_path);
    csv << "benchmark,size_bytes,method,avg_time_ms,latency_us,bw_gbs\n";

    std::cout << std::string(120, '-') << "\n";
    std::cout << std::left
              << std::setw(12) << "Size"
              << std::setw(16) << "P_lat(us)"
              << std::setw(16) << "Pin_lat(us)"
              << std::setw(16) << "Async_lat(us)"
              << std::setw(16) << "TMA_lat(us)"
              << std::setw(14) << "P_BW(GB/s)"
              << std::setw(14) << "Pin_BW(GB/s)"
              << std::setw(14) << "TMA_BW(GB/s)"
              << "\n";
    std::cout << std::string(120, '-') << "\n";

    for (int i = 0; i < NUM_SIZES; ++i) {
        size_t bytes = sizes[i];
        size_t N     = bytes / sizeof(float);
        if (N == 0) N = 1;

        // Label
        char label[32];
        if (bytes < (1 << 20))
            snprintf(label, sizeof(label), "%zu KB", bytes >> 10);
        else
            snprintf(label, sizeof(label), "%zu MB", bytes >> 20);

        float* d_buf = nullptr;
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
            std::cout << std::setw(12) << label << "  SKIPPED\n";
            continue;
        }

        // ── Pageable ──────────────────────────────────────────────────────────
        float* h_page = (float*)malloc(std::max(bytes, sizeof(float)));
        for (size_t j = 0; j < N; ++j) h_page[j] = 1.0f;

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice));

        float sp = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice));
            sp += t.end_ms();
        }
        float avg_p = sp / RUNS;
        float bw_p  = (bytes / 1e9f) / (avg_p / 1e3f);

        // ── Pinned ────────────────────────────────────────────────────────────
        float* h_pin = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_pin, bytes, cudaHostAllocDefault));
        for (size_t j = 0; j < N; ++j) h_pin[j] = 1.0f;

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));

        float sn = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));
            sn += t.end_ms();
        }
        float avg_pin = sn / RUNS;
        float bw_pin  = (bytes / 1e9f) / (avg_pin / 1e3f);

        // ── Async (pinned) ────────────────────────────────────────────────────
        for (int r = 0; r < WARMUP; ++r) {
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pin, bytes,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        float sa = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pin, bytes,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sa += t.end_ms();
        }
        float avg_as = sa / RUNS;
        float bw_as  = (bytes / 1e9f) / (avg_as / 1e3f);

        // Copy data to device for TMA benchmark
        CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));

        // ── TMA ───────────────────────────────────────────────────────────────
        float avg_tma = 0.f, bw_tma = 0.f;
        if (tma_ok && N >= TILE_1D) {
            size_t num_tiles = N / TILE_1D;
            CUtensorMap tma_map = make_tma_1d_f32(d_buf, (uint64_t)N, TILE_1D);
            uint32_t blocks  = (uint32_t)std::min((size_t)8192u, num_tiles);
            uint32_t threads = 32;

            for (int r = 0; r < WARMUP; ++r)
                tma_small_kernel<<<blocks, threads>>>(tma_map, num_tiles, d_sink);
            CUDA_CHECK(cudaDeviceSynchronize());

            float st = 0.f;
            for (int r = 0; r < RUNS; ++r) {
                GpuTimer t; t.begin();
                tma_small_kernel<<<blocks, threads>>>(tma_map, num_tiles, d_sink);
                st += t.end_ms();
            }
            avg_tma = st / RUNS;
            bw_tma  = (bytes / 1e9f) / (avg_tma / 1e3f);
        } else if (tma_ok) {
            // Size smaller than one TMA tile — single call with min granularity
            // Round up N to TILE_1D
            float* d_pad = nullptr;
            CUDA_CHECK(cudaMalloc(&d_pad, TILE_1D * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_pad, h_pin, bytes, cudaMemcpyHostToDevice));
            CUtensorMap tma_map = make_tma_1d_f32(d_pad, TILE_1D, TILE_1D);
            uint32_t blocks = 1, threads = 32;

            for (int r = 0; r < WARMUP; ++r)
                tma_small_kernel<<<blocks, threads>>>(tma_map, 1u, d_sink);
            CUDA_CHECK(cudaDeviceSynchronize());

            float st = 0.f;
            for (int r = 0; r < RUNS; ++r) {
                GpuTimer t; t.begin();
                tma_small_kernel<<<blocks, threads>>>(tma_map, 1u, d_sink);
                st += t.end_ms();
            }
            avg_tma = st / RUNS;
            bw_tma  = (TILE_1D * sizeof(float) / 1e9f) / (avg_tma / 1e3f);
            CUDA_CHECK(cudaFree(d_pad));
        }

        // Print table row (latency in µs)
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(12) << label
                  << std::setw(16) << (avg_p   * 1e3f)
                  << std::setw(16) << (avg_pin  * 1e3f)
                  << std::setw(16) << (avg_as   * 1e3f)
                  << std::setw(16) << (tma_ok ? avg_tma * 1e3f : -1.f)
                  << std::setw(14) << bw_p
                  << std::setw(14) << bw_pin
                  << std::setw(14) << (tma_ok ? bw_tma : -1.f)
                  << "\n";

        auto wcsv = [&](const char* m, float t, float bw) {
            csv << "small_xfer," << bytes << "," << m << ","
                << t << "," << (t * 1e3f) << "," << bw << "\n";
        };
        wcsv("cudaMemcpy_pageable", avg_p,   bw_p);
        wcsv("cudaMemcpy_pinned",   avg_pin, bw_pin);
        wcsv("cudaMemcpyAsync",     avg_as,  bw_as);
        if (tma_ok)
            wcsv("TMA_global_shared", avg_tma, bw_tma);

        free(h_page);
        CUDA_CHECK(cudaFreeHost(h_pin));
        CUDA_CHECK(cudaFree(d_buf));
    }

    std::cout << std::string(120, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  P     = pageable cudaMemcpy H2D\n"
              << "  Pin   = pinned cudaMemcpy H2D\n"
              << "  Async = cudaMemcpyAsync + stream sync (pinned)\n"
              << "  TMA   = global→shared TMA (H100+ only, "
                 "min granularity = 256 B)\n"
              << "  Latency shown in µs; bandwidth in GB/s.\n"
              << "\nCSV written to: " << csv_path << "\n";

    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
