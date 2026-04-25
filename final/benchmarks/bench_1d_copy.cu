// benchmarks/bench_1d_copy.cu
// 1-D Contiguous HBM3→SMEM: cp.async vs TMA
// Referans: PCIe yollari (pageable/pinned)
//
// nvcc -arch=sm_90a -std=c++17 -O3 -lcuda bench_1d_copy.cu -o bench_1d -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr uint32_t TILE = 64;
static constexpr int WARMUP = 5, RUNS = 20;

// ── cp.async kernel ──────────────────────────────────────────────────────────
__global__ void cpasync_1d(const float* __restrict__ src, size_t ntiles, float* __restrict__ sink) {
    __shared__ alignas(16) float smem[TILE];
    float acc = 0.f;
    for (size_t t = blockIdx.x; t < ntiles; t += gridDim.x) {
        size_t base = t * TILE;
        if (threadIdx.x < TILE) {
            uint32_t sp = __cvta_generic_to_shared(&smem[threadIdx.x]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"(sp), "l"(&src[base + threadIdx.x]) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        acc += smem[threadIdx.x & (TILE-1)];
    }
    if (acc == 3.14159265f) *sink = acc;
}

// ── TMA kernel ───────────────────────────────────────────────────────────────
__global__ void tma_1d(const __grid_constant__ CUtensorMap map, size_t ntiles, float* __restrict__ sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using bar_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TILE];
    __shared__ bar_t bar;
    if (threadIdx.x == 0) init(&bar, 1);
    __syncthreads();
    int par = 0; float acc = 0.f;
    for (size_t t = blockIdx.x; t < ntiles; t += gridDim.x) {
        int coord = (int)(t * TILE);
        if (threadIdx.x == 0) {
            uint32_t bp = __cvta_generic_to_shared(&bar);
            uint32_t sp = __cvta_generic_to_shared(smem);
            uint32_t ex = TILE * (uint32_t)sizeof(float);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bp), "r"(ex) : "memory");
            asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sp), "l"(&map), "r"(coord), "r"(bp) : "memory");
        }
        { uint32_t bp = __cvta_generic_to_shared(&bar);
          asm volatile("{\n\t.reg .pred P;\n\tW_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n\t@!P bra W_%=;\n\t}" :: "r"(bp), "r"(par) : "memory"); }
        par ^= 1; __syncthreads();
        acc += smem[threadIdx.x & (TILE-1)];
    }
    if (acc == 3.14159265f) *sink = acc;
#endif
}

// ── Zamanlama yardimcilari ───────────────────────────────────────────────────
static float time_kernel(std::function<void()> fn) {
    for (int r = 0; r < WARMUP; ++r) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        tot += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    return tot / RUNS;
}

static float time_memcpy(void* h, float* d, size_t bytes) {
    for (int r = 0; r < WARMUP; ++r)
        CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    float tot = 0.f;
    for (int r = 0; r < RUNS; ++r) { GpuTimer t; t.begin(); CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); tot += t.end_ms(); }
    return tot / RUNS;
}

int main(int argc, char** argv) {
    const char* csv = (argc>1) ? argv[1] : "results/bench_1d_copy.csv";
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n\n";
    bool tma_ok = tma_supported();

    const size_t sizes[] = { 1ULL<<20, 4ULL<<20, 16ULL<<20, 64ULL<<20, 256ULL<<20, 1ULL<<30, 2ULL<<30 };
    const int NS = sizeof(sizes)/sizeof(sizes[0]);

    std::ofstream fcsv(csv); fcsv << "benchmark,size_mb,method,avg_time_ms,bw_gbs\n";

    // ── Ana: cp.async vs TMA ─────────────────────────────────────────────────
    std::cout << "=== cp.async vs TMA (HBM3 → Shared Memory) ===\n";
    std::cout << std::string(100,'-') << "\n";
    std::cout << std::left << std::setw(10)<<"Size(MB)" << std::setw(18)<<"CPA_time(ms)" << std::setw(18)<<"TMA_time(ms)"
              << std::setw(16)<<"CPA_BW(GB/s)" << std::setw(16)<<"TMA_BW(GB/s)" << std::setw(10)<<"TMA/CPA" << "\n";
    std::cout << std::string(100,'-') << "\n";

    for (int i = 0; i < NS; ++i) {
        size_t bytes = sizes[i], N = bytes/4, mb = bytes>>20;
        float* d; if (cudaMalloc(&d, bytes)!=cudaSuccess) continue;
        { float* h=(float*)malloc(bytes); for(size_t j=0;j<N;j++) h[j]=1.f; CUDA_CHECK(cudaMemcpy(d,h,bytes,cudaMemcpyHostToDevice)); free(h); }
        size_t nt = N/TILE; uint32_t blk = (uint32_t)std::min((size_t)132u, nt);

        float tc = time_kernel([&]{ cpasync_1d<<<blk, TILE>>>(d, nt, d); });
        float bc = (bytes/1e9f)/(tc/1e3f);
        float tt = 0, bt = 0;
        if (tma_ok && N>=TILE) {
            CUtensorMap m = make_tma_1d_f32(d,(uint64_t)N,TILE);
            tt = time_kernel([&]{ tma_1d<<<blk, 32>>>(m, nt, d); });
            bt = (bytes/1e9f)/(tt/1e3f);
        }
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10)<<mb << std::setw(18)<<tc << std::setw(18)<<(tma_ok?tt:-1.f)
                  << std::setw(16)<<bc << std::setw(16)<<(tma_ok?bt:-1.f) << std::setw(10)<<(tma_ok?bt/bc:-1.f) << "\n";
        fcsv << "1d,"<<mb<<",cp.async,"<<tc<<","<<bc<<"\n";
        if (tma_ok) fcsv << "1d,"<<mb<<",TMA,"<<tt<<","<<bt<<"\n";
        CUDA_CHECK(cudaFree(d));
    }

    // ── Referans: PCIe ───────────────────────────────────────────────────────
    std::cout << "\n=== Referans: PCIe (Host → GPU) ===\n";
    std::cout << std::string(80,'-') << "\n";
    std::cout << std::left << std::setw(10)<<"Size(MB)" << std::setw(18)<<"Pageable(ms)" << std::setw(18)<<"Pinned(ms)"
              << std::setw(16)<<"P_BW(GB/s)" << std::setw(16)<<"Pin_BW(GB/s)" << "\n";
    std::cout << std::string(80,'-') << "\n";

    for (int i = 0; i < NS; ++i) {
        size_t bytes = sizes[i], N = bytes/4, mb = bytes>>20;
        float* d; if (cudaMalloc(&d,bytes)!=cudaSuccess) continue;
        float* hp=(float*)malloc(bytes); for(size_t j=0;j<N;j++) hp[j]=1.f;
        float tp = time_memcpy(hp,d,bytes), bp = (bytes/1e9f)/(tp/1e3f);
        float* hn; CUDA_CHECK(cudaHostAlloc(&hn,bytes,cudaHostAllocDefault)); for(size_t j=0;j<N;j++) hn[j]=1.f;
        float tn = time_memcpy(hn,d,bytes), bn = (bytes/1e9f)/(tn/1e3f);
        std::cout << std::fixed << std::setprecision(3) << std::setw(10)<<mb << std::setw(18)<<tp << std::setw(18)<<tn << std::setw(16)<<bp << std::setw(16)<<bn << "\n";
        fcsv << "1d_ref,"<<mb<<",pageable,"<<tp<<","<<bp<<"\n";
        fcsv << "1d_ref,"<<mb<<",pinned,"<<tn<<","<<bn<<"\n";
        free(hp); CUDA_CHECK(cudaFreeHost(hn)); CUDA_CHECK(cudaFree(d));
    }
    std::cout << "\nCSV: " << csv << "\n";
    return 0;
}
