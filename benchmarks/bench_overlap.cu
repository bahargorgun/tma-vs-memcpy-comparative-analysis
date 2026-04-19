// benchmarks/bench_overlap.cu
// Benchmark: compute–transfer overlap — cudaMemcpyAsync vs TMA async pipeline
//
// Three paths measured over NUM_TILES tiles of data:
//   [SEQ]  Sequential: cudaMemcpy H2D then compute kernel (baseline)
//   [OVLP] Async overlap: cudaMemcpyAsync + compute on ping-pong streams
//   [TMA]  TMA pipeline: data pre-staged on device; each sub-tile loaded
//          via TMA global→shared then computed (H100+ only)
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_overlap.cu -o bench_overlap -Iinclude
// Run:
//   ./bench_overlap [csv_out_path]   (default: results/bench_overlap.csv)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

// ─── Configuration ────────────────────────────────────────────────────────────
static constexpr size_t TOTAL_MB  = 256;
static constexpr int    NUM_TILES = 8;
static constexpr size_t TOTAL_BYTES = TOTAL_MB * 1024ULL * 1024ULL;
static constexpr size_t TILE_BYTES  = TOTAL_BYTES / NUM_TILES;
static constexpr size_t TILE_ELEMS  = TILE_BYTES / sizeof(float);

// TMA inner dimension: 64 float32 elements = 256 bytes (hardware limit)
static constexpr uint32_t TMA_TILE = 64;

// Simulated compute workload
static constexpr int COMPUTE_ITERS = 100;

static constexpr int WARMUP = 3;
static constexpr int RUNS   = 10;

// ─── Compute kernel (used by sequential and async-overlap paths) ───────────────

__global__ void compute_kernel(float* __restrict__ data, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = data[idx];
    for (int i = 0; i < COMPUTE_ITERS; ++i)
        x = x * 1.0001f + 0.0001f;
    data[idx] = x;
}

// ─── TMA pipeline kernel ──────────────────────────────────────────────────────
// Each block iterates over its assigned TMA sub-tiles.  For each sub-tile it:
//   1. Issues an asynchronous TMA load from global memory into shared memory.
//   2. Waits for the load to complete (mbarrier).
//   3. Performs COMPUTE_ITERS of work on the loaded data.
//
// This sequential load-then-compute pattern within the block still achieves
// overlap across blocks, reflecting real tiled kernel behaviour.  The mbarrier
// is reinitialised each iteration (parity-tracked) so the barrier can be reused
// without shared-memory reallocation.

__global__ void tma_pipeline_kernel(
    const __grid_constant__ CUtensorMap tma_map,
    uint32_t num_subtiles,
    float* __restrict__ result)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TMA_TILE];
    __shared__ barrier_t bar;

    if (threadIdx.x == 0) init(&bar, 1);
    __syncthreads();

    int parity  = 0;
    float accum = 0.f;

    for (uint32_t st = blockIdx.x; st < num_subtiles; st += gridDim.x) {
        const int coord = (int)(st * TMA_TILE);

        // ── Issue TMA load ────────────────────────────────────────────────────
        if (threadIdx.x == 0) {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            uint32_t sptr = __cvta_generic_to_shared(smem);
            uint32_t exp  = TMA_TILE * (uint32_t)sizeof(float);

            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bptr), "r"(exp) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.1d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sptr), "l"(&tma_map), "r"(coord), "r"(bptr)
                : "memory");
        }

        // ── Wait for TMA completion ───────────────────────────────────────────
        {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "OVLP_TMA_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra OVLP_TMA_WAIT_%=;\n\t"
                "}"
                :: "r"(bptr), "r"(parity) : "memory");
        }
        parity ^= 1;
        __syncthreads();

        // ── Compute on loaded tile ────────────────────────────────────────────
        for (uint32_t e = threadIdx.x; e < TMA_TILE; e += blockDim.x) {
            float x = smem[e];
            for (int i = 0; i < COMPUTE_ITERS; ++i)
                x = x * 1.0001f + 0.0001f;
            accum += x;
        }
        __syncthreads();
    }

    if (accum == 3.14159265f) *result = accum;
#endif
}

// ─── Sequential cudaMemcpy + compute ─────────────────────────────────────────

static float run_sequential(float* h_pin, float* d_buf) {
    const size_t N_tile = TILE_ELEMS;
    const dim3 blk(256), grd((int)((N_tile + 255) / 256));

    // Warmup
    for (int r = 0; r < WARMUP; ++r) {
        for (int t = 0; t < NUM_TILES; ++t) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin + t * N_tile,
                                  TILE_BYTES, cudaMemcpyHostToDevice));
            compute_kernel<<<grd, blk>>>(d_buf, N_tile);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        for (int tile = 0; tile < NUM_TILES; ++tile) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin + tile * N_tile,
                                  TILE_BYTES, cudaMemcpyHostToDevice));
            compute_kernel<<<grd, blk>>>(d_buf, N_tile);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        total += t.end_ms();
    }
    return total / RUNS;
}

// ─── Async overlap: cudaMemcpyAsync + compute (ping-pong streams) ─────────────

static float run_async_overlap(float* h_pin, float* d_ping[2]) {
    const size_t N_tile = TILE_ELEMS;
    const dim3 blk(256), grd((int)((N_tile + 255) / 256));

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    // Warmup
    for (int r = 0; r < WARMUP; ++r) {
        for (int t = 0; t < NUM_TILES; ++t) {
            int s = t & 1;
            CUDA_CHECK(cudaMemcpyAsync(d_ping[s], h_pin + t * N_tile,
                                       TILE_BYTES, cudaMemcpyHostToDevice,
                                       streams[s]));
            compute_kernel<<<grd, blk, 0, streams[s]>>>(d_ping[s], N_tile);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t;
        t.begin();
        for (int tile = 0; tile < NUM_TILES; ++tile) {
            int s = tile & 1;
            CUDA_CHECK(cudaMemcpyAsync(d_ping[s], h_pin + tile * N_tile,
                                       TILE_BYTES, cudaMemcpyHostToDevice,
                                       streams[s]));
            compute_kernel<<<grd, blk, 0, streams[s]>>>(d_ping[s], N_tile);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        total += t.end_ms();
    }

    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    return total / RUNS;
}

// ─── TMA pipeline driver ──────────────────────────────────────────────────────

static float run_tma_pipeline(float* d_full) {
    const size_t N_tile = TILE_ELEMS;
    const uint32_t sub  = (uint32_t)(N_tile / TMA_TILE);  // sub-tiles per tile

    float* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    const uint32_t blocks  = std::min(8192u, sub);
    const uint32_t threads = 64;

    // Warmup
    for (int r = 0; r < WARMUP; ++r) {
        for (int t = 0; t < NUM_TILES; ++t) {
            float* tile_ptr = d_full + (size_t)t * N_tile;
            CUtensorMap tma = make_tma_1d_f32(tile_ptr, (uint64_t)N_tile, TMA_TILE);
            tma_pipeline_kernel<<<blocks, threads>>>(tma, sub, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer timer;
        timer.begin();
        for (int t = 0; t < NUM_TILES; ++t) {
            float* tile_ptr = d_full + (size_t)t * N_tile;
            CUtensorMap tma = make_tma_1d_f32(tile_ptr, (uint64_t)N_tile, TMA_TILE);
            tma_pipeline_kernel<<<blocks, threads>>>(tma, sub, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        total += timer.end_ms();
    }

    CUDA_CHECK(cudaFree(d_result));
    return total / RUNS;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_overlap.csv";

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name
              << "  sm_" << prop.major << prop.minor << "\n";
    std::cout << "Config: " << TOTAL_MB << " MB total, "
              << NUM_TILES << " tiles × "
              << (TILE_BYTES >> 20) << " MB/tile, "
              << "compute_iters=" << COMPUTE_ITERS << "\n\n";

    bool tma_ok = tma_supported();
    if (!tma_ok)
        std::cout << "[WARNING] TMA requires sm_90+ (H100). "
                     "TMA path will be skipped.\n\n";

    // Host buffer (pinned)
    float* h_pin = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pin, TOTAL_BYTES, cudaHostAllocDefault));
    for (size_t j = 0; j < TOTAL_BYTES / sizeof(float); ++j) h_pin[j] = 1.0f;

    // Sequential: 1 device tile
    float* d_seq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seq, TILE_BYTES));

    // Async overlap: 2 ping-pong tiles
    float* d_ping[2];
    CUDA_CHECK(cudaMalloc(&d_ping[0], TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_ping[1], TILE_BYTES));

    // TMA: full array pre-staged on device
    float* d_full = nullptr;
    CUDA_CHECK(cudaMalloc(&d_full, TOTAL_BYTES));
    CUDA_CHECK(cudaMemcpy(d_full, h_pin, TOTAL_BYTES, cudaMemcpyHostToDevice));

    float t_seq  = run_sequential(h_pin, d_seq);
    float t_ovlp = run_async_overlap(h_pin, d_ping);
    float t_tma  = tma_ok ? run_tma_pipeline(d_full) : -1.f;

    float bw_seq  = (TOTAL_BYTES / 1e9f) / (t_seq  / 1e3f);
    float bw_ovlp = (TOTAL_BYTES / 1e9f) / (t_ovlp / 1e3f);
    float bw_tma  = tma_ok ? (TOTAL_BYTES / 1e9f) / (t_tma / 1e3f) : -1.f;

    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left
              << std::setw(32) << "Method"
              << std::setw(14) << "Time(ms)"
              << std::setw(14) << "BW(GB/s)"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    auto row = [&](const char* name, float t, float bw) {
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(32) << name
                  << std::setw(14) << t
                  << std::setw(14) << bw
                  << std::setw(10) << (t_seq / t)
                  << "\n";
    };

    row("Sequential (cudaMemcpy)",          t_seq,  bw_seq);
    row("Async overlap (cudaMemcpyAsync)",  t_ovlp, bw_ovlp);
    if (tma_ok)
        row("TMA pipeline (global→shared)", t_tma,  bw_tma);

    std::cout << std::string(80, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  Sequential and Async paths include PCIe H2D transfer cost.\n"
              << "  TMA path is fully on-device (data pre-staged via cudaMemcpy);\n"
              << "  it measures HBM3→SMEM staging + compute, with no PCIe.\n";

    std::ofstream csv(csv_path);
    csv << "benchmark,total_mb,num_tiles,compute_iters,method,"
           "avg_time_ms,bw_gbs,speedup_vs_seq\n";
    auto wcsv = [&](const char* m, float t, float bw) {
        csv << "overlap," << TOTAL_MB << "," << NUM_TILES << ","
            << COMPUTE_ITERS << "," << m << ","
            << t << "," << bw << "," << (t_seq / t) << "\n";
    };
    wcsv("Sequential_cudaMemcpy",          t_seq,  bw_seq);
    wcsv("Async_overlap_cudaMemcpyAsync",  t_ovlp, bw_ovlp);
    if (tma_ok)
        wcsv("TMA_pipeline_global_shared", t_tma,  bw_tma);

    std::cout << "\nCSV written to: " << csv_path << "\n";

    CUDA_CHECK(cudaFreeHost(h_pin));
    CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_ping[0]));
    CUDA_CHECK(cudaFree(d_ping[1]));
    CUDA_CHECK(cudaFree(d_full));
    return 0;
}
