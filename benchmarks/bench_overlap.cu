// benchmarks/bench_overlap.cu
// Benchmark: compute-transfer overlap pipeline
//
// Ana karsilastirma (ayni donanim yolu):
//   [CPA] cp.async pipeline : tile yukleme + compute, cp.async ile
//   [TMA] TMA pipeline      : tile yukleme + compute, TMA ile
//
// Referans (farkli donanim yolu, PCIe):
//   [SEQ]  Sequential cudaMemcpy + compute
//   [OVLP] cudaMemcpyAsync ping-pong + compute
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_overlap.cu -o bench_overlap -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr size_t TOTAL_MB    = 256;
static constexpr int    NUM_TILES   = 8;
static constexpr size_t TOTAL_BYTES = TOTAL_MB * 1024ULL * 1024ULL;
static constexpr size_t TILE_BYTES  = TOTAL_BYTES / NUM_TILES;
static constexpr size_t TILE_ELEMS  = TILE_BYTES / sizeof(float);
static constexpr uint32_t TMA_TILE  = 64;
static constexpr int COMPUTE_ITERS  = 100;
static constexpr int WARMUP = 5;
static constexpr int RUNS   = 20;

// ─── cp.async pipeline kernel ─────────────────────────────────────────────────

__global__ void cpasync_pipeline_kernel(
    const float* __restrict__ src,
    uint32_t num_subtiles,
    float* __restrict__ result)
{
    __shared__ alignas(16) float smem[TMA_TILE];
    float accum = 0.f;

    for (uint32_t st = blockIdx.x; st < num_subtiles; st += gridDim.x) {
        size_t base = (size_t)st * TMA_TILE;

        // cp.async ile tile yukle
        if (threadIdx.x < TMA_TILE) {
            uint32_t sptr = __cvta_generic_to_shared(&smem[threadIdx.x]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"(sptr), "l"(&src[base + threadIdx.x]) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();

        // Compute
        for (uint32_t e = threadIdx.x; e < TMA_TILE; e += blockDim.x) {
            float x = smem[e];
            for (int i = 0; i < COMPUTE_ITERS; ++i)
                x = x * 1.0001f + 0.0001f;
            accum += x;
        }
        __syncthreads();
    }
    if (accum == 3.14159265f) *result = accum;
}

// ─── TMA pipeline kernel ──────────────────────────────────────────────────────

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
        if (threadIdx.x == 0) {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            uint32_t sptr = __cvta_generic_to_shared(smem);
            uint32_t exp  = TMA_TILE * (uint32_t)sizeof(float);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bptr), "r"(exp) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.1d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sptr), "l"(&tma_map), "r"(coord), "r"(bptr) : "memory");
        }
        { uint32_t bptr = __cvta_generic_to_shared(&bar);
          asm volatile("{\n\t.reg .pred P;\n\tOVLP_WAIT_%=:\n\t"
              "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
              "@!P bra OVLP_WAIT_%=;\n\t}" :: "r"(bptr), "r"(parity) : "memory"); }
        parity ^= 1;
        __syncthreads();

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

// ─── Referans: Sequential cudaMemcpy + compute ────────────────────────────────

__global__ void compute_kernel(float* __restrict__ data, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = data[idx];
    for (int i = 0; i < COMPUTE_ITERS; ++i) x = x * 1.0001f + 0.0001f;
    data[idx] = x;
}

static float run_sequential(float* h_pin, float* d_buf) {
    const size_t N = TILE_ELEMS;
    const dim3 blk(256), grd((int)((N+255)/256));
    for (int r = 0; r < WARMUP; ++r) {
        for (int t = 0; t < NUM_TILES; ++t) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin+t*N, TILE_BYTES, cudaMemcpyHostToDevice));
            compute_kernel<<<grd,blk>>>(d_buf, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t; t.begin();
        for (int tile = 0; tile < NUM_TILES; ++tile) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin+tile*N, TILE_BYTES, cudaMemcpyHostToDevice));
            compute_kernel<<<grd,blk>>>(d_buf, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        tot += t.end_ms();
    }
    return tot / RUNS;
}

static float run_async_overlap(float* h_pin, float* d_ping[2]) {
    const size_t N = TILE_ELEMS;
    const dim3 blk(256), grd((int)((N+255)/256));
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
    for (int r = 0; r < WARMUP; ++r) {
        for (int t = 0; t < NUM_TILES; ++t) {
            int s = t & 1;
            CUDA_CHECK(cudaMemcpyAsync(d_ping[s], h_pin+t*N, TILE_BYTES,
                cudaMemcpyHostToDevice, streams[s]));
            compute_kernel<<<grd,blk,0,streams[s]>>>(d_ping[s], N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        GpuTimer t; t.begin();
        for (int tile = 0; tile < NUM_TILES; ++tile) {
            int s = tile & 1;
            CUDA_CHECK(cudaMemcpyAsync(d_ping[s], h_pin+tile*N, TILE_BYTES,
                cudaMemcpyHostToDevice, streams[s]));
            compute_kernel<<<grd,blk,0,streams[s]>>>(d_ping[s], N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        tot += t.end_ms();
    }
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    return tot / RUNS;
}

// ─── Driver: cp.async pipeline ────────────────────────────────────────────────

static float run_cpasync_pipeline(float* d_full) {
    const uint32_t sub = (uint32_t)(TILE_ELEMS / TMA_TILE);
    float* d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    uint32_t blocks = std::min(8192u, sub), threads = 64;
    for (int r = 0; r < WARMUP; ++r)
        for (int t = 0; t < NUM_TILES; ++t)
            cpasync_pipeline_kernel<<<blocks,threads>>>(
                d_full + (size_t)t*TILE_ELEMS, sub, d_res);
    CUDA_CHECK(cudaDeviceSynchronize());
    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < NUM_TILES; ++t)
            cpasync_pipeline_kernel<<<blocks,threads>>>(
                d_full + (size_t)t*TILE_ELEMS, sub, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        tot += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    CUDA_CHECK(cudaFree(d_res));
    return tot / RUNS;
}

// ─── Driver: TMA pipeline ─────────────────────────────────────────────────────

static float run_tma_pipeline(float* d_full) {
    const uint32_t sub = (uint32_t)(TILE_ELEMS / TMA_TILE);
    float* d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    uint32_t blocks = std::min(8192u, sub), threads = 64;
    for (int r = 0; r < WARMUP; ++r)
        for (int t = 0; t < NUM_TILES; ++t) {
            CUtensorMap tma = make_tma_1d_f32(
                d_full+(size_t)t*TILE_ELEMS, (uint64_t)TILE_ELEMS, TMA_TILE);
            tma_pipeline_kernel<<<blocks,threads>>>(tma, sub, d_res);
        }
    CUDA_CHECK(cudaDeviceSynchronize());
    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < NUM_TILES; ++t) {
            CUtensorMap tma = make_tma_1d_f32(
                d_full+(size_t)t*TILE_ELEMS, (uint64_t)TILE_ELEMS, TMA_TILE);
            tma_pipeline_kernel<<<blocks,threads>>>(tma, sub, d_res);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        tot += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    CUDA_CHECK(cudaFree(d_res));
    return tot / RUNS;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_overlap.csv";
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n";
    std::cout << "Config: " << TOTAL_MB << " MB, " << NUM_TILES << " tiles, "
              << "compute_iters=" << COMPUTE_ITERS << "\n\n";

    bool tma_ok = tma_supported();

    float* h_pin = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pin, TOTAL_BYTES, cudaHostAllocDefault));
    for (size_t j = 0; j < TOTAL_BYTES/sizeof(float); ++j) h_pin[j] = 1.0f;

    float* d_seq = nullptr; CUDA_CHECK(cudaMalloc(&d_seq, TILE_BYTES));
    float* d_ping[2];
    CUDA_CHECK(cudaMalloc(&d_ping[0], TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_ping[1], TILE_BYTES));
    float* d_full = nullptr; CUDA_CHECK(cudaMalloc(&d_full, TOTAL_BYTES));
    CUDA_CHECK(cudaMemcpy(d_full, h_pin, TOTAL_BYTES, cudaMemcpyHostToDevice));

    float t_cpa = run_cpasync_pipeline(d_full);
    float t_tma = tma_ok ? run_tma_pipeline(d_full) : -1.f;
    float t_seq = run_sequential(h_pin, d_seq);
    float t_ovlp = run_async_overlap(h_pin, d_ping);

    float bw_cpa  = (TOTAL_BYTES/1e9f)/(t_cpa/1e3f);
    float bw_tma  = tma_ok ? (TOTAL_BYTES/1e9f)/(t_tma/1e3f) : -1.f;
    float bw_seq  = (TOTAL_BYTES/1e9f)/(t_seq/1e3f);
    float bw_ovlp = (TOTAL_BYTES/1e9f)/(t_ovlp/1e3f);

    std::cout << "=== Ana Karsilastirma: cp.async vs TMA Pipeline (HBM3 yolu) ===\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(30) << "Method"
              << std::setw(14) << "Time(ms)"
              << std::setw(14) << "BW(GB/s)"
              << std::setw(10) << "TMA/CPA" << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << std::setw(30) << "cp.async pipeline"
              << std::setw(14) << t_cpa << std::setw(14) << bw_cpa
              << std::setw(10) << (tma_ok ? t_cpa/t_tma : -1.f) << "\n";
    if (tma_ok)
        std::cout << std::setw(30) << "TMA pipeline"
                  << std::setw(14) << t_tma << std::setw(14) << bw_tma
                  << std::setw(10) << 1.0f << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::cout << "\n=== Referans: PCIe Yollari ===\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(30) << "Method"
              << std::setw(14) << "Time(ms)"
              << std::setw(14) << "BW(GB/s)"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::setw(30) << "Sequential (cudaMemcpy)"
              << std::setw(14) << t_seq << std::setw(14) << bw_seq
              << std::setw(10) << 1.0f << "\n";
    std::cout << std::setw(30) << "Async overlap"
              << std::setw(14) << t_ovlp << std::setw(14) << bw_ovlp
              << std::setw(10) << t_seq/t_ovlp << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::ofstream csv(csv_path);
    csv << "benchmark,method,avg_time_ms,bw_gbs\n";
    csv << "overlap,cp.async_pipeline," << t_cpa << "," << bw_cpa << "\n";
    if (tma_ok) csv << "overlap,TMA_pipeline," << t_tma << "," << bw_tma << "\n";
    csv << "overlap_ref,Sequential_cudaMemcpy," << t_seq << "," << bw_seq << "\n";
    csv << "overlap_ref,Async_cudaMemcpyAsync," << t_ovlp << "," << bw_ovlp << "\n";

    std::cout << "\nNotes:\n"
              << "  cp.async ve TMA: ayni HBM3 yolu, adil karsilastirma.\n"
              << "  PCIe referans: farkli donanim yolu, karsilastirilamaz.\n"
              << "\nCSV: " << csv_path << "\n";

    CUDA_CHECK(cudaFreeHost(h_pin));
    CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_ping[0]));
    CUDA_CHECK(cudaFree(d_ping[1]));
    CUDA_CHECK(cudaFree(d_full));
    return 0;
}
