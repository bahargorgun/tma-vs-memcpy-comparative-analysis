// benchmarks/bench_small_xfer.cu
// Small Transfer Latency: cp.async vs TMA
// Referans: PCIe (pageable/pinned/async)
//
// nvcc -arch=sm_90a -std=c++17 -O3 -lcuda bench_small_xfer.cu -o bench_small -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr uint32_t TILE=64;
static constexpr int WARMUP=5, RUNS=50;

__global__ void cpasync_sm(const float* __restrict__ src, size_t nt, float* __restrict__ sink) {
    __shared__ alignas(16) float smem[TILE]; float acc=0.f;
    for(size_t t=blockIdx.x; t<nt; t+=gridDim.x) {
        size_t b=t*TILE;
        if(threadIdx.x<TILE) { uint32_t sp=__cvta_generic_to_shared(&smem[threadIdx.x]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sp), "l"(&src[b+threadIdx.x]) : "memory"); }
        asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
        acc+=smem[threadIdx.x&(TILE-1)];
    }
    if(acc==3.14159265f) *sink=acc;
}

__global__ void tma_sm(const __grid_constant__ CUtensorMap map, size_t nt, float* __restrict__ sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using bar_t=cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TILE]; __shared__ bar_t bar;
    if(threadIdx.x==0) init(&bar,1); __syncthreads();
    int par=0; float acc=0.f;
    for(size_t t=blockIdx.x; t<nt; t+=gridDim.x) {
        int coord=(int)(t*TILE);
        if(threadIdx.x==0) { uint32_t bp=__cvta_generic_to_shared(&bar), sp=__cvta_generic_to_shared(smem), ex=TILE*4;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bp), "r"(ex) : "memory");
            asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sp), "l"(&map), "r"(coord), "r"(bp) : "memory"); }
        { uint32_t bp=__cvta_generic_to_shared(&bar);
          asm volatile("{\n\t.reg .pred P;\n\tW_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n\t@!P bra W_%=;\n\t}" :: "r"(bp), "r"(par) : "memory"); }
        par^=1; __syncthreads(); acc+=smem[threadIdx.x&(TILE-1)];
    }
    if(acc==3.14159265f) *sink=acc;
#endif
}

static float time_kernel(std::function<void()> fn) {
    for(int r=0;r<WARMUP;++r) fn(); CUDA_CHECK(cudaDeviceSynchronize());
    float tot=0;
    for(int r=0;r<RUNS;++r) { CUDA_CHECK(cudaDeviceSynchronize());
        auto t0=std::chrono::high_resolution_clock::now(); fn(); CUDA_CHECK(cudaDeviceSynchronize());
        tot+=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count(); }
    return tot/RUNS;
}

int main(int argc, char** argv) {
    const char* csv=(argc>1)?argv[1]:"results/bench_small_xfer.csv";
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n\n";
    bool tma_ok=tma_supported();
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    float* d_sink; CUDA_CHECK(cudaMalloc(&d_sink,4));

    const size_t sizes[]={1ULL<<10,4ULL<<10,16ULL<<10,64ULL<<10,256ULL<<10,1ULL<<20,4ULL<<20};
    const int NS=sizeof(sizes)/sizeof(sizes[0]);
    std::ofstream fcsv(csv); fcsv << "benchmark,size_bytes,method,latency_us,bw_gbs\n";

    // ── Ana: cp.async vs TMA ─────────────────────────────────────────────────
    std::cout << "=== cp.async vs TMA Latency (HBM3 → SMEM) ===\n";
    std::cout << std::string(90,'-') << "\n";
    std::cout << std::left << std::setw(10)<<"Size" << std::setw(16)<<"CPA(us)" << std::setw(16)<<"TMA(us)"
              << std::setw(14)<<"CPA_BW(GB/s)" << std::setw(14)<<"TMA_BW(GB/s)" << std::setw(10)<<"TMA/CPA" << "\n";
    std::cout << std::string(90,'-') << "\n";

    for(int i=0; i<NS; ++i) {
        size_t bytes=sizes[i], N=bytes/4; if(N<TILE) N=TILE;
        char lbl[32]; if(bytes<(1<<20)) snprintf(lbl,32,"%zu KB",bytes>>10); else snprintf(lbl,32,"%zu MB",bytes>>20);
        float* d; if(cudaMalloc(&d,N*4)!=cudaSuccess) continue;
        CUDA_CHECK(cudaMemset(d,0x3f,N*4));
        size_t nt=N/TILE; uint32_t blk=(uint32_t)std::min((size_t)132u,nt);
        float tc = time_kernel([&]{ cpasync_sm<<<blk,TILE>>>(d,nt,d_sink); });
        float bc = (N*4/1e9f)/(tc/1e3f);
        float tt=0, bt=0;
        if(tma_ok) { CUtensorMap m=make_tma_1d_f32(d,(uint64_t)N,TILE);
            tt = time_kernel([&]{ tma_sm<<<blk,32>>>(m,nt,d_sink); }); bt=(N*4/1e9f)/(tt/1e3f); }
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10)<<lbl << std::setw(16)<<(tc*1e3f) << std::setw(16)<<(tma_ok?tt*1e3f:-1.f)
                  << std::setw(14)<<bc << std::setw(14)<<(tma_ok?bt:-1.f) << std::setw(10)<<(tma_ok?bt/bc:-1.f) << "\n";
        fcsv << "small,"<<bytes<<",cp.async,"<<(tc*1e3f)<<","<<bc<<"\n";
        if(tma_ok) fcsv << "small,"<<bytes<<",TMA,"<<(tt*1e3f)<<","<<bt<<"\n";
        CUDA_CHECK(cudaFree(d));
    }

    // ── Referans: PCIe ───────────────────────────────────────────────────────
    std::cout << "\n=== Referans: PCIe (Host → GPU) ===\n";
    std::cout << std::string(90,'-') << "\n";
    std::cout << std::left << std::setw(10)<<"Size" << std::setw(16)<<"P(us)" << std::setw(16)<<"Pin(us)"
              << std::setw(16)<<"Async(us)" << std::setw(14)<<"Pin_BW(GB/s)" << "\n";
    std::cout << std::string(90,'-') << "\n";

    for(int i=0; i<NS; ++i) {
        size_t bytes=sizes[i], N=bytes/4; if(!N) N=1;
        char lbl[32]; if(bytes<(1<<20)) snprintf(lbl,32,"%zu KB",bytes>>10); else snprintf(lbl,32,"%zu MB",bytes>>20);
        float* d; if(cudaMalloc(&d,bytes)!=cudaSuccess) continue;
        float* hp=(float*)malloc(std::max(bytes,(size_t)4)); float* hn;
        CUDA_CHECK(cudaHostAlloc(&hn,bytes,cudaHostAllocDefault));
        for(size_t j=0;j<N;j++){hp[j]=1.f;hn[j]=1.f;}

        // Pageable
        for(int r=0;r<WARMUP;++r) CUDA_CHECK(cudaMemcpy(d,hp,bytes,cudaMemcpyHostToDevice));
        float sp=0; for(int r=0;r<RUNS;++r){GpuTimer t;t.begin();CUDA_CHECK(cudaMemcpy(d,hp,bytes,cudaMemcpyHostToDevice));sp+=t.end_ms();}
        float ap=sp/RUNS;
        // Pinned
        for(int r=0;r<WARMUP;++r) CUDA_CHECK(cudaMemcpy(d,hn,bytes,cudaMemcpyHostToDevice));
        float sn=0; for(int r=0;r<RUNS;++r){GpuTimer t;t.begin();CUDA_CHECK(cudaMemcpy(d,hn,bytes,cudaMemcpyHostToDevice));sn+=t.end_ms();}
        float an=sn/RUNS, bn=(bytes/1e9f)/(an/1e3f);
        // Async
        for(int r=0;r<WARMUP;++r){CUDA_CHECK(cudaMemcpyAsync(d,hn,bytes,cudaMemcpyHostToDevice,stream));CUDA_CHECK(cudaStreamSynchronize(stream));}
        float sa=0; for(int r=0;r<RUNS;++r){GpuTimer t;t.begin();CUDA_CHECK(cudaMemcpyAsync(d,hn,bytes,cudaMemcpyHostToDevice,stream));CUDA_CHECK(cudaStreamSynchronize(stream));sa+=t.end_ms();}
        float aa=sa/RUNS;

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10)<<lbl << std::setw(16)<<(ap*1e3f) << std::setw(16)<<(an*1e3f) << std::setw(16)<<(aa*1e3f) << std::setw(14)<<bn << "\n";
        fcsv << "small_ref,"<<bytes<<",pageable,"<<(ap*1e3f)<<","<<(bytes/1e9f)/(ap/1e3f)<<"\n";
        fcsv << "small_ref,"<<bytes<<",pinned,"<<(an*1e3f)<<","<<bn<<"\n";
        fcsv << "small_ref,"<<bytes<<",async,"<<(aa*1e3f)<<","<<(bytes/1e9f)/(aa/1e3f)<<"\n";
        free(hp); CUDA_CHECK(cudaFreeHost(hn)); CUDA_CHECK(cudaFree(d));
    }
    std::cout << "\nCSV: " << csv << "\n";
    CUDA_CHECK(cudaFree(d_sink)); CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
