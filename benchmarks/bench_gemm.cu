// benchmarks/bench_gemm.cu
// Benchmark: GEMM-style tiled matrix multiply — cudaMemcpy tile-load vs TMA tile-load
//
// Measures how much time is spent staging input tiles for a matrix-multiply
// workload. Both paths compute C = A × B (single-precision, naive tiled GEMM)
// but differ in how they bring A/B tiles from global memory into shared memory:
//
//   [MCP] cudaMemcpy path : explicit cudaMemcpy2D to copy tiles into a staging
//                           buffer, then shared-mem GEMM kernel reads from global
//   [TMA] TMA path        : kernel loads tiles directly from global→shared via
//                           TMA descriptors (H100+ only)
//
// Matrix sizes: N×N square matrices (N = 512, 1024, 2048)
// Tile size: TILE_M × TILE_K and TILE_K × TILE_N
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_gemm.cu -o bench_gemm -Iinclude
// Run:
//   ./bench_gemm [csv_out_path]   (default: results/bench_gemm.csv)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

// ─── Tile geometry ────────────────────────────────────────────────────────────
static constexpr int BM = 32;    // tile rows in M
static constexpr int BN = 32;    // tile cols in N
static constexpr int BK = 32;    // tile depth in K
// TMA inner-dim constraint for float32: ≤ 64 elements (256 bytes)
static constexpr uint32_t TMA_BM = 32;   // must be ≤ 64
static constexpr uint32_t TMA_BN = 32;
static constexpr uint32_t TMA_BK = 32;

static constexpr int WARMUP = 2;
static constexpr int RUNS   = 5;

// ─── Baseline GEMM kernel (loads tiles via pointer, no shared-mem staging) ────

__global__ void gemm_global_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ─── TMA-based GEMM kernel ────────────────────────────────────────────────────
// Loads tiles of A and B from global memory into shared memory using TMA,
// then computes the tile multiply-accumulate.

__global__ void gemm_tma_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    float* __restrict__ C,
    int M, int N, int K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    __shared__ alignas(16) float sA[TMA_BM * TMA_BK];
    __shared__ alignas(16) float sB[TMA_BK * TMA_BN];
    __shared__ barrier_t barA, barB;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&barA, 1);
        init(&barB, 1);
    }
    __syncthreads();

    int tile_row = blockIdx.y;   // tile index in M
    int tile_col = blockIdx.x;   // tile index in N

    float acc = 0.f;
    int parA  = 0, parB = 0;

    int num_k_tiles = K / (int)TMA_BK;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Coordinates: (col, row) in element units
        int ax = kt * (int)TMA_BK;    // column in A = k-offset
        int ay = tile_row * (int)TMA_BM; // row in A
        int bx = tile_col * (int)TMA_BN; // column in B
        int by = kt * (int)TMA_BK;    // row in B = k-offset

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            uint32_t bpA  = __cvta_generic_to_shared(&barA);
            uint32_t bpB  = __cvta_generic_to_shared(&barB);
            uint32_t spA  = __cvta_generic_to_shared(sA);
            uint32_t spB  = __cvta_generic_to_shared(sB);
            uint32_t expA = TMA_BM * TMA_BK * (uint32_t)sizeof(float);
            uint32_t expB = TMA_BK * TMA_BN * (uint32_t)sizeof(float);

            // Load tile of A
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bpA), "r"(expA) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(spA), "l"(&tma_A), "r"(ax), "r"(ay), "r"(bpA)
                : "memory");

            // Load tile of B
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bpB), "r"(expB) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(spB), "l"(&tma_B), "r"(bx), "r"(by), "r"(bpB)
                : "memory");
        }

        // Wait for both tiles
        {
            uint32_t bpA = __cvta_generic_to_shared(&barA);
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "GEMM_WAITA_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra GEMM_WAITA_%=;\n\t"
                "}"
                :: "r"(bpA), "r"(parA) : "memory");
        }
        parA ^= 1;

        {
            uint32_t bpB = __cvta_generic_to_shared(&barB);
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "GEMM_WAITB_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra GEMM_WAITB_%=;\n\t"
                "}"
                :: "r"(bpB), "r"(parB) : "memory");
        }
        parB ^= 1;
        __syncthreads();

        // Tile multiply-accumulate (each thread computes one C element)
        int r = threadIdx.y;   // row within tile
        int c = threadIdx.x;   // col within tile
        for (int k = 0; k < (int)TMA_BK; ++k)
            acc += sA[r * TMA_BK + k] * sB[k * TMA_BN + c];

        __syncthreads();
    }

    int crow = tile_row * (int)TMA_BM + threadIdx.y;
    int ccol = tile_col * (int)TMA_BN + threadIdx.x;
    if (crow < M && ccol < N)
        C[crow * N + ccol] = acc;
#endif
}

// ─── Host drivers ─────────────────────────────────────────────────────────────

static float run_gemm_global(float* dA, float* dB, float* dC, int N) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

    for (int r = 0; r < WARMUP; ++r)
        gemm_global_kernel<<<grid, block>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        gemm_global_kernel<<<grid, block>>>(dA, dB, dC, N, N, N);
        total += t.end_ms();
    }
    return total / RUNS;
}

static float run_gemm_tma(float* dA, float* dB, float* dC, int N) {
    // TMA descriptors: A is M×K (N×N), B is K×N (N×N)
    size_t pitch_bytes = (size_t)N * sizeof(float);

    CUtensorMap tmaA = make_tma_2d_f32(dA, (uint64_t)N, (uint64_t)N,
                                        pitch_bytes, TMA_BK, TMA_BM);
    CUtensorMap tmaB = make_tma_2d_f32(dB, (uint64_t)N, (uint64_t)N,
                                        pitch_bytes, TMA_BN, TMA_BK);

    dim3 block(TMA_BN, TMA_BM);
    dim3 grid((N + TMA_BN - 1) / TMA_BN, (N + TMA_BM - 1) / TMA_BM);

    for (int r = 0; r < WARMUP; ++r)
        gemm_tma_kernel<<<grid, block>>>(tmaA, tmaB, dC, N, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        gemm_tma_kernel<<<grid, block>>>(tmaA, tmaB, dC, N, N, N);
        total += t.end_ms();
    }
    return total / RUNS;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_gemm.csv";

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

    const int sizes[] = { 512, 1024, 2048 };
    const int NUM_SIZES = (int)(sizeof(sizes) / sizeof(sizes[0]));

    std::ofstream csv(csv_path);
    csv << "benchmark,matrix_n,method,avg_time_ms,gflops,bw_gbs\n";

    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(20) << "Global_time(ms)"
              << std::setw(20) << "TMA_time(ms)"
              << std::setw(16) << "Global_GFLOPS"
              << std::setw(16) << "TMA_GFLOPS"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (int i = 0; i < NUM_SIZES; ++i) {
        int N = sizes[i];
        size_t bytes = (size_t)N * N * sizeof(float);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        if (cudaMalloc(&dA, bytes) != cudaSuccess ||
            cudaMalloc(&dB, bytes) != cudaSuccess ||
            cudaMalloc(&dC, bytes) != cudaSuccess)
        {
            std::cout << std::setw(8) << N << "  SKIPPED (insufficient VRAM)\n";
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            continue;
        }

        // Init to 1.0
        CUDA_CHECK(cudaMemset(dA, 0x3f, bytes));
        CUDA_CHECK(cudaMemset(dB, 0x3f, bytes));
        CUDA_CHECK(cudaMemset(dC, 0,    bytes));

        float t_global = run_gemm_global(dA, dB, dC, N);
        float t_tma    = tma_ok ? run_gemm_tma(dA, dB, dC, N) : -1.f;

        // GFLOPS = 2 * N^3 / time_s
        double ops = 2.0 * (double)N * N * N;
        float gf_global = (float)(ops / ((double)t_global / 1e3) / 1e9);
        float gf_tma    = tma_ok
                        ? (float)(ops / ((double)t_tma / 1e3) / 1e9) : -1.f;

        // Bandwidth: read A+B, write C = 3 × N² × 4 bytes
        float read_bytes = 3.f * (float)bytes;
        float bw_global  = read_bytes / 1e9f / (t_global / 1e3f);
        float bw_tma     = tma_ok
                         ? read_bytes / 1e9f / (t_tma / 1e3f) : -1.f;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8)  << N
                  << std::setw(20) << t_global
                  << std::setw(20) << (tma_ok ? t_tma : -1.f)
                  << std::setw(16) << gf_global
                  << std::setw(16) << (tma_ok ? gf_tma : -1.f)
                  << std::setw(10) << (tma_ok ? t_global / t_tma : -1.f)
                  << "\n";

        csv << "gemm," << N << ",global_loads,"
            << t_global << "," << gf_global << "," << bw_global << "\n";
        if (tma_ok)
            csv << "gemm," << N << ",TMA_tile_loads,"
                << t_tma << "," << gf_tma << "," << bw_tma << "\n";

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    std::cout << std::string(100, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  Global path: each thread reads from global memory directly.\n"
              << "  TMA path:    tiles loaded from global→shared via TMA, "
                 "then thread-local accumulate.\n"
              << "  Both kernels perform the same floating-point work.\n"
              << "\nCSV written to: " << csv_path << "\n";

    return 0;
}
