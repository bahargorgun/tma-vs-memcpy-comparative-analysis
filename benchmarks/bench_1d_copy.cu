// benchmarks/bench_1d_copy.cu
// Benchmark: 1-D contiguous HBM3→shared memory transfer
//
// Karsilastirma: her iki yol da ayni donanim yolunu (HBM3→SMEM) kullanir.
//   [CPA] cp.async : thread'ler koordine eder, DMA tasinir (sm_80+)
//   [TMA] TMA      : donanim halleder, thread sadece koordinat verir (sm_90+)
//
// Referans olarak PCIe yollari da gosterilir (ayri bolumde):
//   [P]   cudaMemcpy pageable H2D
//   [Pin] cudaMemcpy pinned   H2D
//
// Build:
//   nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
//        benchmarks/bench_1d_copy.cu -o bench_1d -Iinclude
// Run:
//   ./bench_1d [csv_out_path]

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr uint32_t TILE_1D = 64;   // float32: 256 byte
static constexpr int WARMUP = 5;
static constexpr int RUNS   = 20;

// ─── cp.async kernel: global → shared ────────────────────────────────────────
// Her block bir tile yukler, threadler koordine eder.

__global__ void cpasync_1d_kernel(
    const float* __restrict__ src,
    size_t num_tiles,
    float* __restrict__ sink)
{
    __shared__ alignas(16) float smem[TILE_1D];
    float accum = 0.f;

    for (size_t tile = (size_t)blockIdx.x; tile < num_tiles;
         tile += (size_t)gridDim.x)
    {
        size_t base = tile * TILE_1D;

        // cp.async ile global → shared (her thread bir eleman)
        if (threadIdx.x < TILE_1D) {
            uint32_t sptr = __cvta_generic_to_shared(&smem[threadIdx.x]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"(sptr), "l"(&src[base + threadIdx.x])
                : "memory");
        }
        // Tum cp.async komutlarinin bitmesini bekle
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();

        accum += smem[threadIdx.x & (TILE_1D - 1)];
    }

    if (accum == 3.14159265f) *sink = accum;
}

// ─── TMA kernel: global → shared ─────────────────────────────────────────────

__global__ void tma_1d_kernel(
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
                "TMA1D_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra TMA1D_WAIT_%=;\n\t"
                "}" :: "r"(bptr), "r"(parity) : "memory");
        }
        parity ^= 1;
        __syncthreads();
        accum += smem[threadIdx.x & (TILE_1D - 1)];
    }

    if (accum == 3.14159265f) *sink = accum;
#endif
}

// ─── Driver: cp.async ────────────────────────────────────────────────────────

static float run_cpasync_1d(float* d_data, size_t N) {
    size_t num_tiles = N / TILE_1D;
    if (num_tiles == 0) return 0.f;
    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    uint32_t blocks  = (uint32_t)std::min((size_t)132u, num_tiles);
    uint32_t threads = TILE_1D;

    for (int r = 0; r < WARMUP; ++r)
        cpasync_1d_kernel<<<blocks, threads>>>(d_data, num_tiles, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        cpasync_1d_kernel<<<blocks, threads>>>(d_data, num_tiles, d_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    CUDA_CHECK(cudaFree(d_sink));
    return total / RUNS;
}

// ─── Driver: TMA ─────────────────────────────────────────────────────────────

static float run_tma_1d(float* d_data, size_t N) {
    size_t num_tiles = N / TILE_1D;
    if (num_tiles == 0) return 0.f;
    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    CUtensorMap tma_map = make_tma_1d_f32(d_data, (uint64_t)N, TILE_1D);
    uint32_t blocks  = (uint32_t)std::min((size_t)132u, num_tiles);
    uint32_t threads = 32;

    for (int r = 0; r < WARMUP; ++r)
        tma_1d_kernel<<<blocks, threads>>>(tma_map, num_tiles, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        tma_1d_kernel<<<blocks, threads>>>(tma_map, num_tiles, d_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    CUDA_CHECK(cudaFree(d_sink));
    return total / RUNS;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "results/bench_1d_copy.csv";

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name
              << "  sm_" << prop.major << prop.minor << "\n\n";

    bool tma_ok = tma_supported();
    if (!tma_ok)
        std::cout << "[WARNING] TMA requires sm_90+ (H100).\n\n";

    const size_t sizes[] = {
        1ULL  << 20,
        4ULL  << 20,
        16ULL << 20,
        64ULL << 20,
        256ULL<< 20,
        1ULL  << 30,
        2ULL  << 30,
    };
    const int NUM_SIZES = (int)(sizeof(sizes) / sizeof(sizes[0]));

    std::ofstream csv(csv_path);
    csv << "benchmark,size_mb,method,avg_time_ms,bw_gbs\n";

    // ── Ana karsilastirma: cp.async vs TMA (HBM3 → SMEM) ────────────────────
    std::cout << "=== Ana Karsilastirma: cp.async vs TMA (HBM3 → Shared Memory) ===\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left
              << std::setw(10) << "Size(MB)"
              << std::setw(20) << "CPA_time(ms)"
              << std::setw(20) << "TMA_time(ms)"
              << std::setw(18) << "CPA_BW(GB/s)"
              << std::setw(18) << "TMA_BW(GB/s)"
              << std::setw(12) << "TMA/CPA"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (int i = 0; i < NUM_SIZES; ++i) {
        size_t bytes = sizes[i];
        size_t N     = bytes / sizeof(float);
        size_t mb    = bytes >> 20;

        float* d_buf = nullptr;
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
            std::cout << std::setw(10) << mb << "  SKIPPED\n";
            continue;
        }
        // Veriyi device'a yukle
        float* h_tmp = (float*)malloc(bytes);
        for (size_t j = 0; j < N; ++j) h_tmp[j] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_buf, h_tmp, bytes, cudaMemcpyHostToDevice));
        free(h_tmp);

        float avg_cpa = run_cpasync_1d(d_buf, N);
        float bw_cpa  = (bytes / 1e9f) / (avg_cpa / 1e3f);

        float avg_tma = 0.f, bw_tma = 0.f;
        if (tma_ok && N >= TILE_1D) {
            avg_tma = run_tma_1d(d_buf, N);
            bw_tma  = (bytes / 1e9f) / (avg_tma / 1e3f);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << mb
                  << std::setw(20) << avg_cpa
                  << std::setw(20) << (tma_ok ? avg_tma : -1.f)
                  << std::setw(18) << bw_cpa
                  << std::setw(18) << (tma_ok ? bw_tma : -1.f)
                  << std::setw(12) << (tma_ok ? bw_tma / bw_cpa : -1.f)
                  << "\n";

        csv << "1d_copy," << mb << ",cp.async," << avg_cpa << "," << bw_cpa << "\n";
        if (tma_ok && N >= TILE_1D)
            csv << "1d_copy," << mb << ",TMA," << avg_tma << "," << bw_tma << "\n";

        CUDA_CHECK(cudaFree(d_buf));
    }
    std::cout << std::string(100, '-') << "\n";

    // ── Referans: PCIe yollari ────────────────────────────────────────────────
    std::cout << "\n=== Referans: PCIe Yollari (Host → GPU, farkli donanim yolu) ===\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left
              << std::setw(10) << "Size(MB)"
              << std::setw(20) << "Pageable(ms)"
              << std::setw(20) << "Pinned(ms)"
              << std::setw(18) << "P_BW(GB/s)"
              << std::setw(18) << "Pin_BW(GB/s)"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int i = 0; i < NUM_SIZES; ++i) {
        size_t bytes = sizes[i];
        size_t N     = bytes / sizeof(float);
        size_t mb    = bytes >> 20;

        float* d_buf = nullptr;
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
            std::cout << std::setw(10) << mb << "  SKIPPED\n";
            continue;
        }

        float* h_page = (float*)malloc(bytes);
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

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << mb
                  << std::setw(20) << avg_p
                  << std::setw(20) << avg_pin
                  << std::setw(18) << bw_p
                  << std::setw(18) << bw_pin
                  << "\n";

        csv << "1d_copy_ref," << mb << ",pageable," << avg_p << "," << bw_p << "\n";
        csv << "1d_copy_ref," << mb << ",pinned,"   << avg_pin << "," << bw_pin << "\n";

        free(h_page);
        CUDA_CHECK(cudaFreeHost(h_pin));
        CUDA_CHECK(cudaFree(d_buf));
    }
    std::cout << std::string(80, '-') << "\n";
    std::cout << "\nNotes:\n"
              << "  Ana karsilastirma: cp.async ve TMA ikisi de HBM3→SMEM yolunu kullanir.\n"
              << "  TMA/CPA > 1 ise TMA daha hizli demektir.\n"
              << "  PCIe referans: karsilastirilamaz, sadece context icin gosterilir.\n"
              << "\nCSV: " << csv_path << "\n";
    return 0;
}
