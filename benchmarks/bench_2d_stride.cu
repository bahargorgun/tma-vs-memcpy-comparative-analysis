// benchmarks/bench_2d_stride.cu
// Benchmark: 2-D strided HBM3→shared memory transfer
//
// Ana karsilastirma (ayni donanim yolu, HBM3→SMEM):
//   [CPA] cp.async : thread bazli tile yukleme
//   [TMA] TMA 2D   : donanim bazli tile yukleme
//
// Referans (farkli donanim yolu, PCIe):
//   [M2D] cudaMemcpy2D pinned H2D
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_2d_stride.cu -o bench_2d -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr uint32_t TILE_COLS = 32;
static constexpr uint32_t TILE_ROWS = 32;
static constexpr int WARMUP = 5;
static constexpr int RUNS   = 20;

// ─── cp.async kernel: 2D strided global → shared ─────────────────────────────
// Her block bir TILE_ROWS x TILE_COLS tile yukler.
// pitch_elems: bir satirin eleman sayisi (stride dahil)

__global__ void cpasync_2d_kernel(
    const float* __restrict__ src,
    uint32_t num_tile_cols,
    uint32_t num_tile_rows,
    uint32_t pitch_elems,
    float* __restrict__ sink)
{
    __shared__ alignas(16) float smem[TILE_ROWS * TILE_COLS];
    float accum = 0.f;

    uint32_t total_tiles = num_tile_cols * num_tile_rows;
    for (uint32_t tid = blockIdx.x; tid < total_tiles; tid += gridDim.x) {
        uint32_t tx = tid % num_tile_cols;
        uint32_t ty = tid / num_tile_cols;

        // Her thread bir satirdan bir eleman yukler
        uint32_t local_row = threadIdx.x / TILE_COLS;
        uint32_t local_col = threadIdx.x % TILE_COLS;

        if (local_row < TILE_ROWS) {
            uint32_t global_row = ty * TILE_ROWS + local_row;
            uint32_t global_col = tx * TILE_COLS + local_col;
            size_t   global_idx = (size_t)global_row * pitch_elems + global_col;

            uint32_t sptr = __cvta_generic_to_shared(
                &smem[local_row * TILE_COLS + local_col]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"(sptr), "l"(&src[global_idx]) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();

        accum += smem[threadIdx.x % (TILE_ROWS * TILE_COLS)];
        __syncthreads();
    }

    if (accum == 3.14159265f) *sink = accum;
}

// ─── TMA kernel: 2D global → shared ──────────────────────────────────────────

__global__ void tma_2d_kernel(
    const __grid_constant__ CUtensorMap tma_map,
    uint32_t num_tile_cols,
    uint32_t num_tile_rows,
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
                :: "r"(sptr), "l"(&tma_map), "r"(cx), "r"(cy), "r"(bptr)
                : "memory");
        }
        {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "TMA2D_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra TMA2D_WAIT_%=;\n\t"
                "}" :: "r"(bptr), "r"(parity) : "memory");
        }
        parity ^= 1;
        __syncthreads();
        accum += smem[threadIdx.x % (TILE_COLS * TILE_ROWS)];
    }

    if (accum == 3.14159265f) *sink = accum;
#endif
}

// ─── Driver: cp.async 2D ─────────────────────────────────────────────────────

static float run_cpasync_2d(
    float* d_mat, uint64_t cols, uint64_t rows, uint64_t pitch_elems)
{
    uint32_t ntc = (uint32_t)(cols / TILE_COLS);
    uint32_t ntr = (uint32_t)(rows / TILE_ROWS);
    if (ntc == 0 || ntr == 0) return 0.f;

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    uint32_t total = ntc * ntr;
    uint32_t blocks  = std::min(8192u, total);
    uint32_t threads = TILE_ROWS * TILE_COLS;

    for (int r = 0; r < WARMUP; ++r)
        cpasync_2d_kernel<<<blocks, threads>>>(
            d_mat, ntc, ntr, (uint32_t)pitch_elems, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        cpasync_2d_kernel<<<blocks, threads>>>(
            d_mat, ntc, ntr, (uint32_t)pitch_elems, d_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        tot += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    CUDA_CHECK(cudaFree(d_sink));
    return tot / RUNS;
}

// ─── Driver: TMA 2D ──────────────────────────────────────────────────────────

static float run_tma_2d(
    float* d_mat, uint64_t cols, uint64_t rows, uint64_t pitch_b)
{
    uint32_t ntc = (uint32_t)(cols / TILE_COLS);
    uint32_t ntr = (uint32_t)(rows / TILE_ROWS);
    if (ntc == 0 || ntr == 0) return 0.f;

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    CUtensorMap tma_map = make_tma_2d_f32(
        d_mat, cols, rows, pitch_b, TILE_COLS, TILE_ROWS);

    uint32_t total = ntc * ntr;
    uint32_t blocks  = std::min(8192u, total);
    uint32_t threads = 32;

    for (int r = 0; r < WARMUP; ++r)
        tma_2d_kernel<<<blocks, threads>>>(tma_map, ntc, ntr, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        tma_2d_kernel<<<blocks, threads>>>(tma_map, ntc, ntr, d_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        tot += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    CUDA_CHECK(cudaFree(d_sink));
    return tot / RUNS;
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

    const int configs[][2] = {
        { 512,  512}, {1024,  512}, {1024, 1024},
        {2048, 1024}, {2048, 2048}, {4096, 2048}, {4096, 4096},
    };
    const int NUM_CFG = (int)(sizeof(configs) / sizeof(configs[0]));

    std::ofstream csv(csv_path);
    csv << "benchmark,width,height,method,avg_time_ms,payload_mb,bw_gbs\n";

    std::cout << "=== Ana Karsilastirma: cp.async vs TMA (HBM3 → Shared Memory) ===\n";
    std::cout << std::string(110, '-') << "\n";
    std::cout << std::left
              << std::setw(7)  << "W"
              << std::setw(7)  << "H"
              << std::setw(20) << "CPA_time(ms)"
              << std::setw(20) << "TMA_time(ms)"
              << std::setw(18) << "CPA_BW(GB/s)"
              << std::setw(18) << "TMA_BW(GB/s)"
              << std::setw(12) << "TMA/CPA"
              << std::setw(12) << "Payload(MB)"
              << "\n";
    std::cout << std::string(110, '-') << "\n";

    for (int i = 0; i < NUM_CFG; ++i) {
        size_t W = (size_t)configs[i][0];
        size_t H = (size_t)configs[i][1];

        // Stride: 2x padding (gercek hayatta tipik)
        size_t pitch_elems = W * 2;
        size_t pitch_bytes = pitch_elems * sizeof(float);
        size_t alloc_bytes = pitch_bytes * H;
        size_t payload_bytes = W * H * sizeof(float);

        float* d_mat = nullptr;
        if (cudaMalloc(&d_mat, alloc_bytes) != cudaSuccess) {
            std::cout << std::setw(7) << W << std::setw(7) << H << "  SKIPPED\n";
            continue;
        }
        CUDA_CHECK(cudaMemset(d_mat, 0x3f, alloc_bytes));

        float avg_cpa = run_cpasync_2d(d_mat, W, H, pitch_elems);
        float bw_cpa  = (payload_bytes / 1e9f) / (avg_cpa / 1e3f);

        float avg_tma = 0.f, bw_tma = 0.f;
        if (tma_ok && W >= TILE_COLS && H >= TILE_ROWS) {
            avg_tma = run_tma_2d(d_mat, W, H, pitch_bytes);
            bw_tma  = (payload_bytes / 1e9f) / (avg_tma / 1e3f);
        }

        float payload_mb = (float)(payload_bytes >> 20);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(7)  << W
                  << std::setw(7)  << H
                  << std::setw(20) << avg_cpa
                  << std::setw(20) << (tma_ok ? avg_tma : -1.f)
                  << std::setw(18) << bw_cpa
                  << std::setw(18) << (tma_ok ? bw_tma : -1.f)
                  << std::setw(12) << (tma_ok ? bw_tma / bw_cpa : -1.f)
                  << std::setw(12) << payload_mb
                  << "\n";

        csv << "2d_stride," << W << "," << H << ",cp.async,"
            << avg_cpa << "," << payload_mb << "," << bw_cpa << "\n";
        if (tma_ok && W >= TILE_COLS && H >= TILE_ROWS)
            csv << "2d_stride," << W << "," << H << ",TMA,"
                << avg_tma << "," << payload_mb << "," << bw_tma << "\n";

        CUDA_CHECK(cudaFree(d_mat));
    }
    std::cout << std::string(110, '-') << "\n";

    // ── Referans: cudaMemcpy2D ────────────────────────────────────────────────
    std::cout << "\n=== Referans: cudaMemcpy2D (PCIe yolu, farkli donanim) ===\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left
              << std::setw(7) << "W" << std::setw(7) << "H"
              << std::setw(20) << "M2D_time(ms)"
              << std::setw(18) << "M2D_BW(GB/s)"
              << std::setw(12) << "Payload(MB)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int i = 0; i < NUM_CFG; ++i) {
        size_t W = (size_t)configs[i][0];
        size_t H = (size_t)configs[i][1];
        size_t pitch_bytes = W * 2 * sizeof(float);
        size_t alloc_bytes = pitch_bytes * H;
        size_t payload_bytes = W * H * sizeof(float);

        float* h_pin = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_pin, alloc_bytes, cudaHostAllocDefault));
        float* d_mat = nullptr;
        if (cudaMalloc(&d_mat, alloc_bytes) != cudaSuccess) {
            CUDA_CHECK(cudaFreeHost(h_pin)); continue;
        }

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy2D(d_mat, pitch_bytes, h_pin, pitch_bytes,
                W * sizeof(float), H, cudaMemcpyHostToDevice));

        float tot = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpy2D(d_mat, pitch_bytes, h_pin, pitch_bytes,
                W * sizeof(float), H, cudaMemcpyHostToDevice));
            tot += t.end_ms();
        }
        float avg_m2d = tot / RUNS;
        float bw_m2d  = (alloc_bytes / 1e9f) / (avg_m2d / 1e3f);
        float payload_mb = (float)(payload_bytes >> 20);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(7) << W << std::setw(7) << H
                  << std::setw(20) << avg_m2d
                  << std::setw(18) << bw_m2d
                  << std::setw(12) << payload_mb << "\n";

        csv << "2d_stride_ref," << W << "," << H << ",cudaMemcpy2D,"
            << avg_m2d << "," << payload_mb << "," << bw_m2d << "\n";

        CUDA_CHECK(cudaFreeHost(h_pin));
        CUDA_CHECK(cudaFree(d_mat));
    }
    std::cout << std::string(80, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  Ana karsilastirma: cp.async ve TMA, ikisi de HBM3→SMEM.\n"
              << "  TMA/CPA > 1 ise TMA daha hizli.\n"
              << "  cudaMemcpy2D referans: PCIe yolu, karsilastirilamaz.\n"
              << "\nCSV: " << csv_path << "\n";
    return 0;
}
