// benchmarks/bench_overlap.cu
// Compute-Transfer Pipeline: cp.async vs TMA
// Referans: Sequential/Async cudaMemcpy (PCIe)
//
// nvcc -arch=sm_90a -std=c++17 -O3 -lcuda bench_overlap.cu -o bench_overlap -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr size_t TOTAL_MB=256, NUM_TILES=8;
static constexpr size_t TOTAL_BYTES=TOTAL_MB*1024ULL*1024ULL;
static constexpr size_t TILE_BYTES=TOTAL_BYTES/NUM_TILES, TILE_ELEMS=TILE_BYTES/4;
static constexpr uint32_t TMA_TILE=64;
static constexpr int COMPUTE_ITERS=100, WARMUP=5, RUNS=20;

__global__ void compute_kern(float* __restrict__ d, size_t N) {
    size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    float x=d[i]; for(int j=0;j<COMPUTE_ITERS;++j) x=x*1.0001f+0.0001f; d[i]=x;
}

// ── cp.async pipeline kernel ─────────────────────────────────────────────────
__global__ void cpasync_pipe(const float* __restrict__ src, uint32_t nsub, float* __restrict__ res) {
    __shared__ alignas(16) float smem[TMA_TILE]; float acc=0.f;
    for (uint32_t st=blockIdx.x; st<nsub; st+=gridDim.x) {
        size_t base=(size_t)st*TMA_TILE;
        if (threadIdx.x<TMA_TILE) {
            uint32_t sp=__cvta_generic_to_shared(&smem[threadIdx.x]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sp), "l"(&src[base+threadIdx.x]) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
        for (uint32_t e=threadIdx.x; e<TMA_TILE; e+=blockDim.x) {
            float x=smem[e]; for(int i=0;i<COMPUTE_ITERS;++i) x=x*1.0001f+0.0001f; acc+=x;
        } __syncthreads();
    }
    if (acc==3.14159265f) *res=acc;
}

// ── TMA pipeline kernel ──────────────────────────────────────────────────────
__global__ void tma_pipe(const __grid_constant__ CUtensorMap map, uint32_t nsub, float* __restrict__ res) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using bar_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float smem[TMA_TILE]; __shared__ bar_t bar;
    if (threadIdx.x==0) init(&bar,1); __syncthreads();
    int par=0; float acc=0.f;
    for (uint32_t st=blockIdx.x; st<nsub; st+=gridDim.x) {
        int coord=(int)(st*TMA_TILE);
        if (threadIdx.x==0) {
            uint32_t bp=__cvta_generic_to_shared(&bar), sp=__cvta_generic_to_shared(smem), ex=TMA_TILE*4;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bp), "r"(ex) : "memory");
            asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                :: "r"(sp), "l"(&map), "r"(coord), "r"(bp) : "memory");
        }
        { uint32_t bp=__cvta_generic_to_shared(&bar);
          asm volatile("{\n\t.reg .pred P;\n\tW_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n\t@!P bra W_%=;\n\t}" :: "r"(bp), "r"(par) : "memory"); }
        par^=1; __syncthreads();
        for (uint32_t e=threadIdx.x; e<TMA_TILE; e+=blockDim.x) {
            float x=smem[e]; for(int i=0;i<COMPUTE_ITERS;++i) x=x*1.0001f+0.0001f; acc+=x;
        } __syncthreads();
    }
    if (acc==3.14159265f) *res=acc;
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
    const char* csv = (argc>1)?argv[1]:"results/bench_overlap.csv";
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n";
    std::cout << "Config: " << TOTAL_MB << " MB, " << NUM_TILES << " tiles, iters=" << COMPUTE_ITERS << "\n\n";
    bool tma_ok = tma_supported();

    float* h_pin; CUDA_CHECK(cudaHostAlloc(&h_pin,TOTAL_BYTES,cudaHostAllocDefault));
    for(size_t j=0;j<TOTAL_BYTES/4;++j) h_pin[j]=1.f;
    float* d_seq; CUDA_CHECK(cudaMalloc(&d_seq,TILE_BYTES));
    float *d_ping[2]; CUDA_CHECK(cudaMalloc(&d_ping[0],TILE_BYTES)); CUDA_CHECK(cudaMalloc(&d_ping[1],TILE_BYTES));
    float* d_full; CUDA_CHECK(cudaMalloc(&d_full,TOTAL_BYTES));
    CUDA_CHECK(cudaMemcpy(d_full,h_pin,TOTAL_BYTES,cudaMemcpyHostToDevice));

    uint32_t sub=(uint32_t)(TILE_ELEMS/TMA_TILE), blk=std::min(8192u,sub);
    float* d_res; CUDA_CHECK(cudaMalloc(&d_res,4));

    // cp.async pipeline
    float t_cpa = time_kernel([&]{
        for(int t=0;t<NUM_TILES;++t) cpasync_pipe<<<blk,64>>>(d_full+(size_t)t*TILE_ELEMS,sub,d_res);
    });

    // TMA pipeline
    float t_tma = -1.f;
    if (tma_ok) {
        t_tma = time_kernel([&]{
            for(int t=0;t<NUM_TILES;++t) {
                CUtensorMap m=make_tma_1d_f32(d_full+(size_t)t*TILE_ELEMS,(uint64_t)TILE_ELEMS,TMA_TILE);
                tma_pipe<<<blk,64>>>(m,sub,d_res);
            }
        });
    }

    // PCIe referans: sequential
    const dim3 cblk(256), cgrd((int)((TILE_ELEMS+255)/256));
    float t_seq = time_kernel([&]{
        for(int t=0;t<NUM_TILES;++t) {
            CUDA_CHECK(cudaMemcpy(d_seq,h_pin+t*TILE_ELEMS,TILE_BYTES,cudaMemcpyHostToDevice));
            compute_kern<<<cgrd,cblk>>>(d_seq,TILE_ELEMS); CUDA_CHECK(cudaDeviceSynchronize());
        }
    });

    // PCIe referans: async overlap
    cudaStream_t streams[2]; CUDA_CHECK(cudaStreamCreate(&streams[0])); CUDA_CHECK(cudaStreamCreate(&streams[1]));
    float t_ovlp = time_kernel([&]{
        for(int t=0;t<NUM_TILES;++t) {
            int s=t&1;
            CUDA_CHECK(cudaMemcpyAsync(d_ping[s],h_pin+t*TILE_ELEMS,TILE_BYTES,cudaMemcpyHostToDevice,streams[s]));
            compute_kern<<<cgrd,cblk,0,streams[s]>>>(d_ping[s],TILE_ELEMS);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    });
    CUDA_CHECK(cudaStreamDestroy(streams[0])); CUDA_CHECK(cudaStreamDestroy(streams[1]));

    auto bw = [](float t){ return (TOTAL_BYTES/1e9f)/(t/1e3f); };

    std::cout << "=== cp.async vs TMA Pipeline (HBM3) ===\n";
    std::cout << std::string(70,'-') << "\n";
    std::cout << std::left << std::setw(28)<<"Method" << std::setw(14)<<"Time(ms)" << std::setw(14)<<"BW(GB/s)" << std::setw(10)<<"TMA/CPA" << "\n";
    std::cout << std::string(70,'-') << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << std::setw(28)<<"cp.async pipeline" << std::setw(14)<<t_cpa << std::setw(14)<<bw(t_cpa) << std::setw(10)<<1.0f << "\n";
    if (tma_ok)
        std::cout << std::setw(28)<<"TMA pipeline" << std::setw(14)<<t_tma << std::setw(14)<<bw(t_tma) << std::setw(10)<<(t_cpa/t_tma) << "\n";

    std::cout << "\n=== Referans: PCIe ===\n";
    std::cout << std::string(70,'-') << "\n";
    std::cout << std::setw(28)<<"Sequential (cudaMemcpy)" << std::setw(14)<<t_seq << std::setw(14)<<bw(t_seq) << "\n";
    std::cout << std::setw(28)<<"Async overlap" << std::setw(14)<<t_ovlp << std::setw(14)<<bw(t_ovlp) << "\n";
    std::cout << std::string(70,'-') << "\n";

    std::ofstream fcsv(csv); fcsv << "benchmark,method,avg_time_ms,bw_gbs\n";
    fcsv << "overlap,cp.async,"<<t_cpa<<","<<bw(t_cpa)<<"\n";
    if(tma_ok) fcsv << "overlap,TMA,"<<t_tma<<","<<bw(t_tma)<<"\n";
    fcsv << "overlap_ref,sequential,"<<t_seq<<","<<bw(t_seq)<<"\n";
    fcsv << "overlap_ref,async,"<<t_ovlp<<","<<bw(t_ovlp)<<"\n";
    std::cout << "\nCSV: " << csv << "\n";

    CUDA_CHECK(cudaFreeHost(h_pin)); CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_ping[0])); CUDA_CHECK(cudaFree(d_ping[1]));
    CUDA_CHECK(cudaFree(d_full)); CUDA_CHECK(cudaFree(d_res));
    return 0;
}
