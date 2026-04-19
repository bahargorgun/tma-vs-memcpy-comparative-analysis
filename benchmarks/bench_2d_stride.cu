// benchmarks/bench_2d_stride.cu
// Benchmark: 2-D strided transfer — cudaMemcpy2D vs TMA 2-D global→shared
//
// Two paths:
//   [M2D] cudaMemcpy2D  — host (pinned) → device, with row padding
//   [TMA] TMA 2D        — device global → shared memory (H100+ only)
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_2d_stride.cu -o bench_2d -Iinclude
// Run:
//   ./bench_2d [csv_out_path]   (default: results/bench_2d_stride.csv)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

// ─── TMA tile geometry ────────────────────────────────────────────────────────
// For float32: inner dim ≤ 256 bytes → ≤ 64 columns; outer dim ≤ 256 rows.
static constexpr uint32_t TILE_COLS = 32;   // elements (128 bytes)
static constexpr uint32_t TILE_ROWS = 32;   // elements

// ─── Benchmark parameters ─────────────────────────────────────────────────────
static constexpr int WARMUP = 3;
static constexpr int RUNS   = 10;

// ─── TMA 2D kernel: global → shared (one tile per loop iteration) ─────────────

__global__ void tma_2d_kernel(
    const __grid_constant__ CUtensorMap tma_map,
    uint32_t num_tile_cols,   // number of tiles in x
    uint32_t num_tile_rows,   // number of tiles in y
    float* __restrict__ sink)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TILE_ROWS * TILE_COLS];
    __shared__ barrier_t bar;

    if (threadIdx.x == 0) init(&bar, 1);
    __syncthreads();

    int parity  = 0;
    float accum = 0.f;

    // Each block processes a subset of tiles, strided by gridDim.x
    uint32_t total_tiles = num_tile_cols * num_tile_rows;
    for (uint32_t tid = blockIdx.x; tid < total_tiles; tid += gridDim.x) {
        uint32_t tx = tid % num_tile_cols;
        uint32_t ty = tid / num_tile_cols;
        int cx = (int)(tx * TILE_COLS);
        int cy = (int)(ty * TILE_ROWS);

        if (threadIdx.x == 0) {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            uint32_t sptr = __cvta_generic_to_shared(smem);
            uint32_t exp  = TILE_COLS * TILE_ROWS * (uint32_t)sizeof(float);

            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bptr), "r"(exp) : "memory");

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(sptr), "l"(&tma_map),
                   "r"(cx), "r"(cy),
                   "r"(bptr)
                : "memory");
        }

        // All threads wait
        {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            asm volatile(
                "{\n\t"
                ".reg .pred P;\n\t"
                "TMA2D_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra TMA2D_WAIT_%=;\n\t"
                "}"
                :: "r"(bptr), "r"(parity) : "memory");
        }
        parity ^= 1;
        __syncthreads();

        accum += smem[threadIdx.x & (TILE_COLS * TILE_ROWS - 1)];
    }

    if (accum == 3.14159265f) *sink = accum;
#endif
}

// ─── Host driver for TMA 2D ───────────────────────────────────────────────────

// d_mat   : device pointer to matrix
// cols    : number of columns (elements, = width)
// rows    : number of rows (= height)
// pitch_b : byte stride between rows (>= cols * sizeof(float))
// Returns average time (ms) to copy the entire matrix tile-by-tile via TMA.
static float run_tma_2d(
    float* d_mat, uint64_t cols, uint64_t rows, uint64_t pitch_b)
{
    uint32_t num_tile_cols = (uint32_t)(cols / TILE_COLS);
    uint32_t num_tile_rows = (uint32_t)(rows / TILE_ROWS);
    if (num_tile_cols == 0 || num_tile_rows == 0) return 0.f;

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    CUtensorMap tma_map = make_tma_2d_f32(
        d_mat, cols, rows, pitch_b, TILE_COLS, TILE_ROWS);

    uint32_t total_tiles = num_tile_cols * num_tile_rows;
    uint32_t blocks  = std::min(8192u, total_tiles);
    uint32_t threads = 32;

    for (int r = 0; r < WARMUP; ++r)
        tma_2d_kernel<<<blocks, threads>>>(
            tma_map, num_tile_cols, num_tile_rows, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        tma_2d_kernel<<<blocks, threads>>>(
            tma_map, num_tile_cols, num_tile_rows, d_sink);
        total += t.end_ms();
    }

    CUDA_CHECK(cudaFree(d_sink));
    return total / RUNS;
}

// ─── cudaMemcpy2D driver ──────────────────────────────────────────────────────

static float run_memcpy2d(
    float* h_pin, float* d_mat,
    size_t width_elems, size_t height,
    size_t pitch_bytes)
{
    // Warmup
    for (int r = 0; r < WARMUP; ++r)
        CUDA_CHECK(cudaMemcpy2D(
            d_mat, pitch_bytes,
            h_pin, pitch_bytes,
            width_elems * sizeof(float), height,
            cudaMemcpyHostToDevice));

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        CUDA_CHECK(cudaMemcpy2D(
            d_mat, pitch_bytes,
            h_pin, pitch_bytes,
            width_elems * sizeof(float), height,
            cudaMemcpyHostToDevice));
        total += t.end_ms();
    }
    return total / RUNS;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_2d_stride.csv";

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

    // Matrix configurations: {width, height}
    const int configs[][2] = {
        { 512,  512},
        {1024,  512},
        {1024, 1024},
        {2048, 1024},
        {2048, 2048},
        {4096, 2048},
        {4096, 4096},
    };
    const int NUM_CFG = (int)(sizeof(configs) / sizeof(configs[0]));

    std::ofstream csv(csv_path);
    csv << "benchmark,width,height,pitch_factor,method,avg_time_ms,"
           "payload_mb,bw_gbs,efficiency_pct\n";

    std::cout << std::string(128, '-') << "\n";
    std::cout << std::left
              << std::setw(7)  << "W"
              << std::setw(7)  << "H"
              << std::setw(10) << "Pitch×"
              << std::setw(18) << "M2D_time(ms)"
              << std::setw(18) << "TMA_time(ms)"
              << std::setw(16) << "M2D_BW(GB/s)"
              << std::setw(16) << "TMA_BW(GB/s)"
              << std::setw(14) << "Payload(MB)"
              << "\n";
    std::cout << std::string(128, '-') << "\n";

    for (int i = 0; i < NUM_CFG; ++i) {
        size_t W = (size_t)configs[i][0];
        size_t H = (size_t)configs[i][1];

        // 2× row padding (50 % wasted bandwidth, typical strided case)
        size_t pitch_elems = W * 2;
        size_t pitch_bytes = pitch_elems * sizeof(float);
        size_t alloc_bytes = pitch_bytes * H;
        size_t payload_bytes = W * H * sizeof(float);

        float* h_pin = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_pin, alloc_bytes, cudaHostAllocDefault));
        for (size_t j = 0; j < pitch_elems * H; ++j) h_pin[j] = 1.0f;

        float* d_mat = nullptr;
        if (cudaMalloc(&d_mat, alloc_bytes) != cudaSuccess) {
            std::cout << std::setw(7) << W << std::setw(7) << H
                      << "  SKIPPED (insufficient VRAM)\n";
            CUDA_CHECK(cudaFreeHost(h_pin));
            continue;
        }

        // Initial H2D for TMA baseline
        CUDA_CHECK(cudaMemcpy2D(
            d_mat, pitch_bytes,
            h_pin, pitch_bytes,
            W * sizeof(float), H,
            cudaMemcpyHostToDevice));

        // ── cudaMemcpy2D ──────────────────────────────────────────────────────
        float avg_m2d = run_memcpy2d(h_pin, d_mat, W, H, pitch_bytes);
        float bw_m2d  = (alloc_bytes / 1e9f) / (avg_m2d / 1e3f);

        // ── TMA 2D ────────────────────────────────────────────────────────────
        float avg_tma = 0.f, bw_tma = 0.f;
        if (tma_ok && W >= TILE_COLS && H >= TILE_ROWS) {
            avg_tma = run_tma_2d(d_mat, (uint64_t)W, (uint64_t)H, pitch_bytes);
            bw_tma  = (payload_bytes / 1e9f) / (avg_tma / 1e3f);
        }

        float payload_mb = (float)(payload_bytes >> 20);
        // Efficiency = actual payload / bytes physically transferred (stride waste)
        float eff_m2d = 100.f * (float)payload_bytes / (float)alloc_bytes;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(7)  << W
                  << std::setw(7)  << H
                  << std::setw(10) << "2x"
                  << std::setw(18) << avg_m2d
                  << std::setw(18) << (tma_ok ? avg_tma : -1.f)
                  << std::setw(16) << bw_m2d
                  << std::setw(16) << (tma_ok ? bw_tma : -1.f)
                  << std::setw(14) << payload_mb
                  << "\n";

        csv << "2d_stride," << W << "," << H << ",2x,cudaMemcpy2D,"
            << avg_m2d << "," << payload_mb << "," << bw_m2d << ","
            << eff_m2d << "\n";
        if (tma_ok && W >= TILE_COLS && H >= TILE_ROWS)
            csv << "2d_stride," << W << "," << H << ",2x,TMA_2d_global_shared,"
                << avg_tma << "," << payload_mb << "," << bw_tma << ","
                << 100.f << "\n";

        CUDA_CHECK(cudaFreeHost(h_pin));
        CUDA_CHECK(cudaFree(d_mat));
    }

    std::cout << std::string(128, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  M2D  = cudaMemcpy2D H2D, pitch = 2× width (50 %% row padding)\n"
              << "  TMA  = TMA 2-D global→shared, tile "
              << TILE_COLS << "×" << TILE_ROWS << " elements (H100+ only)\n"
              << "  M2D BW uses allocated bytes (incl. padding); "
                 "TMA BW uses payload bytes only.\n"
              << "\nCSV written to: " << csv_path << "\n";

    return 0;
}
