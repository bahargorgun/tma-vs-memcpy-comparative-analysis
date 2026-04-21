// benchmarks/bench_small_xfer.cu
// Benchmark: kucuk transfer latency — cp.async vs TMA
//
// Ana karsilastirma (HBM3→SMEM, ayni yol):
//   [CPA] cp.async : thread bazli tile yukleme latency
//   [TMA] TMA      : donanim bazli tile yukleme latency
//
// Referans (PCIe, farkli yol):
//   [P]   cudaMemcpy pageable
//   [Pin] cudaMemcpy pinned
//   [AS]  cudaMemcpyAsync
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_small_xfer.cu -o bench_small -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr uint32_t TILE_1D = 64;
static constexpr int WARMUP = 5;
static constexpr int RUNS   = 50;

// ─── cp.async kernel ──────────────────────────────────────────────────────────

__global__ void cpasync_small_kernel(
    const float* __restrict__ src,
    size_t num_tiles,
    float* __restrict__ sink)
{
    __shared__ alignas(16) float smem[TILE_1D];
    float accum = 0.f;
    for (size_t tile = (size_t)blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        size_t base = tile * TILE_1D;
        if (threadIdx.x < TILE_1D) {
            uint32_t sptr = __cvta_generic_to_shared(&smem[threadIdx.x]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"(sptr), "l"(&src[base + threadIdx.x]) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        accum += smem[threadIdx.x & (TILE_1D - 1)];
    }
    if (accum == 3.14159265f) *sink = accum;
}

// ─── TMA kernel ───────────────────────────────────────────────────────────────

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
    int parity = 0;
    float accum = 0.f;
    for (size_t tile = (size_t)blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        const int coord = (int)(tile * TILE_1D);
        if (threadIdx.x == 0) {
            uint32_t bptr = __cvta_generic_to_shared(&bar);
            uint32_t sptr = __cvta_generic_to_shared(smem);
            uint32_t exp  = TILE_1D * (uint32_t)sizeof(float);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(bptr), "r"(exp) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.1d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sptr), "l"(&tma_map), "r"(coord), "r"(bptr) : "memory");
        }
        { uint32_t bptr = __cvta_generic_to_shared(&bar);
          asm volatile("{\n\t.reg .pred P;\n\tSM_WAIT_%=:\n\t"
              "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
              "@!P bra SM_WAIT_%=;\n\t}" :: "r"(bptr), "r"(parity) : "memory"); }
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
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n\n";

    bool tma_ok = tma_supported();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    const size_t sizes[] = {
        1ULL<<10, 4ULL<<10, 16ULL<<10, 64ULL<<10,
        256ULL<<10, 1ULL<<20, 4ULL<<20,
    };
    const int NS = (int)(sizeof(sizes)/sizeof(sizes[0]));

    std::ofstream csv(csv_path);
    csv << "benchmark,size_bytes,method,avg_time_ms,latency_us,bw_gbs\n";

    // ── Ana karsilastirma: cp.async vs TMA ────────────────────────────────────
    std::cout << "=== Ana Karsilastirma: cp.async vs TMA (HBM3 → Shared Memory) ===\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left
              << std::setw(10) << "Size"
              << std::setw(18) << "CPA_lat(us)"
              << std::setw(18) << "TMA_lat(us)"
              << std::setw(16) << "CPA_BW(GB/s)"
              << std::setw(16) << "TMA_BW(GB/s)"
              << std::setw(10) << "TMA/CPA" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (int i = 0; i < NS; ++i) {
        size_t bytes = sizes[i];
        size_t N     = bytes / sizeof(float);
        if (N < TILE_1D) N = TILE_1D;

        char label[32];
        if (bytes < (1<<20)) snprintf(label, sizeof(label), "%zu KB", bytes>>10);
        else                  snprintf(label, sizeof(label), "%zu MB", bytes>>20);

        float* d_buf = nullptr;
        if (cudaMalloc(&d_buf, N*sizeof(float)) != cudaSuccess) continue;
        CUDA_CHECK(cudaMemset(d_buf, 0x3f, N*sizeof(float)));

        // cp.async
        size_t num_tiles = N / TILE_1D;
        uint32_t blocks_cpa = (uint32_t)std::min((size_t)132u, num_tiles);
        for (int r = 0; r < WARMUP; ++r)
            cpasync_small_kernel<<<blocks_cpa, TILE_1D>>>(d_buf, num_tiles, d_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        float sc = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            cpasync_small_kernel<<<blocks_cpa, TILE_1D>>>(d_buf, num_tiles, d_sink);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            sc += std::chrono::duration<float, std::milli>(t1 - t0).count();
        }
        float avg_cpa = sc / RUNS;
        float bw_cpa  = (N*sizeof(float)/1e9f)/(avg_cpa/1e3f);

        // TMA
        float avg_tma = 0.f, bw_tma = 0.f;
        if (tma_ok) {
            CUtensorMap tma_map = make_tma_1d_f32(d_buf, (uint64_t)N, TILE_1D);
            uint32_t blocks_tma = (uint32_t)std::min((size_t)132u, num_tiles);
            for (int r = 0; r < WARMUP; ++r)
                tma_small_kernel<<<blocks_tma, 32>>>(tma_map, num_tiles, d_sink);
            CUDA_CHECK(cudaDeviceSynchronize());
            float st = 0.f;
            for (int r = 0; r < RUNS; ++r) {
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t0 = std::chrono::high_resolution_clock::now();
                tma_small_kernel<<<blocks_tma, 32>>>(tma_map, num_tiles, d_sink);
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t1 = std::chrono::high_resolution_clock::now();
                st += std::chrono::duration<float, std::milli>(t1 - t0).count();
            }
            avg_tma = st / RUNS;
            bw_tma  = (N*sizeof(float)/1e9f)/(avg_tma/1e3f);
        }

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << label
                  << std::setw(18) << (avg_cpa*1e3f)
                  << std::setw(18) << (tma_ok ? avg_tma*1e3f : -1.f)
                  << std::setw(16) << bw_cpa
                  << std::setw(16) << (tma_ok ? bw_tma : -1.f)
                  << std::setw(10) << (tma_ok ? bw_tma/bw_cpa : -1.f)
                  << "\n";

        csv << "small_xfer," << bytes << ",cp.async,"
            << avg_cpa << "," << (avg_cpa*1e3f) << "," << bw_cpa << "\n";
        if (tma_ok)
            csv << "small_xfer," << bytes << ",TMA,"
                << avg_tma << "," << (avg_tma*1e3f) << "," << bw_tma << "\n";

        CUDA_CHECK(cudaFree(d_buf));
    }
    std::cout << std::string(90, '-') << "\n";

    // ── Referans: PCIe yollari ────────────────────────────────────────────────
    std::cout << "\n=== Referans: PCIe Yollari (Host → GPU) ===\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left
              << std::setw(10) << "Size"
              << std::setw(18) << "P_lat(us)"
              << std::setw(18) << "Pin_lat(us)"
              << std::setw(18) << "Async_lat(us)"
              << std::setw(16) << "Pin_BW(GB/s)" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (int i = 0; i < NS; ++i) {
        size_t bytes = sizes[i];
        size_t N     = bytes / sizeof(float);
        if (N == 0) N = 1;

        char label[32];
        if (bytes < (1<<20)) snprintf(label, sizeof(label), "%zu KB", bytes>>10);
        else                  snprintf(label, sizeof(label), "%zu MB", bytes>>20);

        float* d_buf = nullptr;
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) continue;

        float* h_page = (float*)malloc(std::max(bytes, sizeof(float)));
        float* h_pin  = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_pin, bytes, cudaHostAllocDefault));
        for (size_t j = 0; j < N; ++j) { h_page[j] = 1.f; h_pin[j] = 1.f; }

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice));
        float sp = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice));
            sp += t.end_ms();
        }
        float avg_p = sp / RUNS;

        for (int r = 0; r < WARMUP; ++r)
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));
        float sn = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));
            sn += t.end_ms();
        }
        float avg_pin = sn / RUNS;
        float bw_pin  = (bytes/1e9f)/(avg_pin/1e3f);

        for (int r = 0; r < WARMUP; ++r) {
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pin, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        float sa = 0.f;
        for (int r = 0; r < RUNS; ++r) {
            GpuTimer t; t.begin();
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pin, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sa += t.end_ms();
        }
        float avg_as = sa / RUNS;

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << label
                  << std::setw(18) << (avg_p  *1e3f)
                  << std::setw(18) << (avg_pin*1e3f)
                  << std::setw(18) << (avg_as *1e3f)
                  << std::setw(16) << bw_pin << "\n";

        csv << "small_xfer_ref," << bytes << ",pageable,"
            << avg_p << "," << (avg_p*1e3f) << "," << (bytes/1e9f)/(avg_p/1e3f) << "\n";
        csv << "small_xfer_ref," << bytes << ",pinned,"
            << avg_pin << "," << (avg_pin*1e3f) << "," << bw_pin << "\n";
        csv << "small_xfer_ref," << bytes << ",async,"
            << avg_as << "," << (avg_as*1e3f) << "," << (bytes/1e9f)/(avg_as/1e3f) << "\n";

        free(h_page);
        CUDA_CHECK(cudaFreeHost(h_pin));
        CUDA_CHECK(cudaFree(d_buf));
    }
    std::cout << std::string(90, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  Ana karsilastirma: cp.async ve TMA, ikisi de HBM3→SMEM.\n"
              << "  PCIe referans: karsilastirilamaz, sadece context icin.\n"
              << "\nCSV: " << csv_path << "\n";

    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
