// benchmarks/bench_2d_stride.cu
// 2-D Strided HBM3→SMEM: cp.async vs TMA
// TMA'nin en buyuk avantaji: stride bilgisi descriptor'da, donanim halleder
//
// nvcc -arch=sm_90a -std=c++17 -O3 -lcuda bench_2d_stride.cu -o bench_2d -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr uint32_t TC = 32, TR = 32;
static constexpr int WARMUP = 5, RUNS = 20;

// ── cp.async 2D kernel ───────────────────────────────────────────────────────
__global__ void cpasync_2d(const float* __restrict__ src, uint32_t ntc, uint32_t ntr,
                           uint32_t pitch_elems, float* __restrict__ sink) {
    __shared__ alignas(16) float smem[TR*TC];
    float acc = 0.f;
    uint32_t total = ntc * ntr;
    for (uint32_t tid = blockIdx.x; tid < total; tid += gridDim.x) {
        uint32_t tx = tid % ntc, ty = tid / ntc;
        uint32_t lr = threadIdx.x / TC, lc = threadIdx.x % TC;
        if (lr < TR) {
            uint32_t gr = ty*TR+lr, gc = tx*TC+lc;
            uint32_t sp = __cvta_generic_to_shared(&smem[lr*TC+lc]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"(sp), "l"(&src[(size_t)gr*pitch_elems+gc]) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        acc += smem[threadIdx.x % (TR*TC)];
        __syncthreads();
    }
    if (acc == 3.14159265f) *sink = acc;
}

// ── TMA 2D kernel ────────────────────────────────────────────────────────────
__global__ void tma_2d(const __grid_constant__ CUtensorMap map, uint32_t ntc, uint32_t ntr,
                       float* __restrict__ sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using bar_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TR*TC];
    __shared__ bar_t bar;
    if (threadIdx.x == 0) init(&bar, 1);
    __syncthreads();
    int par = 0; float acc = 0.f;
    uint32_t total = ntc * ntr;
    for (uint32_t tid = blockIdx.x; tid < total; tid += gridDim.x) {
        int cx = (int)((tid%ntc)*TC), cy = (int)((tid/ntc)*TR);
        if (threadIdx.x == 0) {
            uint32_t bp = __cvta_generic_to_shared(&bar);
            uint32_t sp = __cvta_generic_to_shared(smem);
            uint32_t ex = TC*TR*(uint32_t)sizeof(float);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bp), "r"(ex) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(sp), "l"(&map), "r"(cx), "r"(cy), "r"(bp) : "memory");
        }
        { uint32_t bp = __cvta_generic_to_shared(&bar);
          asm volatile("{\n\t.reg .pred P;\n\tW_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n\t@!P bra W_%=;\n\t}" :: "r"(bp), "r"(par) : "memory"); }
        par ^= 1; __syncthreads();
        acc += smem[threadIdx.x % (TC*TR)];
    }
    if (acc == 3.14159265f) *sink = acc;
#endif
}

static float time_kernel(std::function<void()> fn) {
    for (int r=0;r<WARMUP;++r) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    float tot=0;
    for (int r=0;r<RUNS;++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0=std::chrono::high_resolution_clock::now(); fn(); CUDA_CHECK(cudaDeviceSynchronize());
        auto t1=std::chrono::high_resolution_clock::now();
        tot+=std::chrono::duration<float,std::milli>(t1-t0).count();
    }
    return tot/RUNS;
}

int main(int argc, char** argv) {
    const char* csv = (argc>1)?argv[1]:"results/bench_2d_stride.csv";
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n\n";
    bool tma_ok = tma_supported();

    const int cfgs[][2] = {{512,512},{1024,512},{1024,1024},{2048,1024},{2048,2048},{4096,2048},{4096,4096}};
    const int NC = sizeof(cfgs)/sizeof(cfgs[0]);
    std::ofstream fcsv(csv); fcsv << "benchmark,width,height,method,avg_time_ms,payload_mb,bw_gbs\n";

    std::cout << "=== cp.async vs TMA 2-D Strided (HBM3 → SMEM, 2x padding) ===\n";
    std::cout << std::string(110,'-') << "\n";
    std::cout << std::left << std::setw(7)<<"W" << std::setw(7)<<"H" << std::setw(18)<<"CPA(ms)" << std::setw(18)<<"TMA(ms)"
              << std::setw(16)<<"CPA_BW(GB/s)" << std::setw(16)<<"TMA_BW(GB/s)" << std::setw(10)<<"TMA/CPA" << std::setw(10)<<"Payload(MB)" << "\n";
    std::cout << std::string(110,'-') << "\n";

    for (int i=0; i<NC; ++i) {
        size_t W=cfgs[i][0], H=cfgs[i][1];
        size_t pitch_e = W*2, pitch_b = pitch_e*4;
        size_t alloc = pitch_b*H, payload = W*H*4;
        float* d; if(cudaMalloc(&d,alloc)!=cudaSuccess) continue;
        CUDA_CHECK(cudaMemset(d,0x3f,alloc));
        float* sink; CUDA_CHECK(cudaMalloc(&sink,4));

        uint32_t ntc=W/TC, ntr=H/TR, total=ntc*ntr;
        uint32_t blk_cpa=std::min(8192u,total), blk_tma=std::min(8192u,total);

        float tc = time_kernel([&]{ cpasync_2d<<<blk_cpa,TR*TC>>>(d,ntc,ntr,(uint32_t)pitch_e,sink); });
        float bc = (payload/1e9f)/(tc/1e3f);
        float tt=0, bt=0;
        if (tma_ok && W>=TC && H>=TR) {
            CUtensorMap m = make_tma_2d_f32(d,W,H,pitch_b,TC,TR);
            tt = time_kernel([&]{ tma_2d<<<blk_tma,32>>>(m,ntc,ntr,sink); });
            bt = (payload/1e9f)/(tt/1e3f);
        }
        float pmb = (float)(payload>>20);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(7)<<W << std::setw(7)<<H << std::setw(18)<<tc << std::setw(18)<<(tma_ok?tt:-1.f)
                  << std::setw(16)<<bc << std::setw(16)<<(tma_ok?bt:-1.f) << std::setw(10)<<(tma_ok?bt/bc:-1.f) << std::setw(10)<<pmb << "\n";
        fcsv << "2d,"<<W<<","<<H<<",cp.async,"<<tc<<","<<pmb<<","<<bc<<"\n";
        if (tma_ok) fcsv << "2d,"<<W<<","<<H<<",TMA,"<<tt<<","<<pmb<<","<<bt<<"\n";
        CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(sink));
    }
    std::cout << std::string(110,'-') << "\nCSV: " << csv << "\n";
    return 0;
}
