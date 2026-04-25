// benchmarks/bench_gemm.cu
// GEMM Tile Fetch: Global Load vs cp.async vs TMA
// Ayni hesaplama (C = A*B), sadece tile yukleme yontemi farkli
// Kare NxN matrisler — anlamli boyutlarda (512-4096)
//
// nvcc -arch=sm_90a -std=c++17 -O3 -lcuda bench_gemm.cu -o bench_gemm -Iinclude

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include "tma_utils.cuh"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr int BM=32, BN=32, BK=32;
static constexpr uint32_t TBM=32, TBN=32, TBK=32;
static constexpr int WARMUP=5, RUNS=20;

// ── Global load GEMM ─────────────────────────────────────────────────────────
__global__ void gemm_global(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int N) {
    int row = blockIdx.y*BM+threadIdx.y, col = blockIdx.x*BN+threadIdx.x;
    if (row>=N||col>=N) return;
    float acc=0.f;
    for (int k=0;k<N;++k) acc += A[row*N+k]*B[k*N+col];
    C[row*N+col] = acc;
}

// ── cp.async GEMM ────────────────────────────────────────────────────────────
__global__ void gemm_cpasync(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int N) {
    __shared__ alignas(16) float sA[BM*BK], sB[BK*BN];
    int tr=blockIdx.y, tc=blockIdx.x, r=threadIdx.y, c=threadIdx.x;
    float acc=0.f;
    for (int kt=0; kt<N/BK; ++kt) {
        uint32_t spA = __cvta_generic_to_shared(&sA[r*BK+c]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(spA), "l"(&A[(tr*BM+r)*N+kt*BK+c]) : "memory");
        uint32_t spB = __cvta_generic_to_shared(&sB[r*BN+c]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(spB), "l"(&B[(kt*BK+r)*N+tc*BN+c]) : "memory");
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        for (int k=0;k<BK;++k) acc += sA[r*BK+k]*sB[k*BN+c];
        __syncthreads();
    }
    int cr=tr*BM+r, cc=tc*BN+c;
    if (cr<N&&cc<N) C[cr*N+cc] = acc;
}

// ── TMA GEMM ─────────────────────────────────────────────────────────────────
__global__ void gemm_tma(const __grid_constant__ CUtensorMap tA, const __grid_constant__ CUtensorMap tB,
                         float* __restrict__ C, int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using bar_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(16) float sA[TBM*TBK], sB[TBK*TBN];
    __shared__ bar_t bA, bB;
    if (threadIdx.x==0&&threadIdx.y==0) { init(&bA,1); init(&bB,1); }
    __syncthreads();
    int tr=blockIdx.y, tc=blockIdx.x; float acc=0.f; int pA=0, pB=0;
    for (int kt=0; kt<N/(int)TBK; ++kt) {
        int ax=kt*(int)TBK, ay=tr*(int)TBM, bx=tc*(int)TBN, by=kt*(int)TBK;
        if (threadIdx.x==0&&threadIdx.y==0) {
            uint32_t bpA=__cvta_generic_to_shared(&bA), bpB=__cvta_generic_to_shared(&bB);
            uint32_t spA=__cvta_generic_to_shared(sA), spB=__cvta_generic_to_shared(sB);
            uint32_t eA=TBM*TBK*4, eB=TBK*TBN*4;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bpA), "r"(eA) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(spA), "l"(&tA), "r"(ax), "r"(ay), "r"(bpA) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bpB), "r"(eB) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(spB), "l"(&tB), "r"(bx), "r"(by), "r"(bpB) : "memory");
        }
        { uint32_t bp=__cvta_generic_to_shared(&bA);
          asm volatile("{\n\t.reg .pred P;\n\tGA_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n\t@!P bra GA_%=;\n\t}" :: "r"(bp), "r"(pA) : "memory"); } pA^=1;
        { uint32_t bp=__cvta_generic_to_shared(&bB);
          asm volatile("{\n\t.reg .pred P;\n\tGB_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n\t@!P bra GB_%=;\n\t}" :: "r"(bp), "r"(pB) : "memory"); } pB^=1;
        __syncthreads();
        int r2=threadIdx.y, c2=threadIdx.x;
        for (int k=0;k<(int)TBK;++k) acc += sA[r2*TBK+k]*sB[k*TBN+c2];
        __syncthreads();
    }
    int cr=tr*(int)TBM+threadIdx.y, cc=tc*(int)TBN+threadIdx.x;
    if (cr<N&&cc<N) C[cr*N+cc] = acc;
#endif
}

static float time_kernel(std::function<void()> fn) {
    for (int r=0;r<WARMUP;++r) fn(); CUDA_CHECK(cudaDeviceSynchronize());
    float tot=0;
    for (int r=0;r<RUNS;++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0=std::chrono::high_resolution_clock::now(); fn(); CUDA_CHECK(cudaDeviceSynchronize());
        tot+=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
    }
    return tot/RUNS;
}

int main(int argc, char** argv) {
    const char* csv = (argc>1)?argv[1]:"results/bench_gemm.csv";
    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::cout << "Device: " << prop.name << "  sm_" << prop.major << prop.minor << "\n\n";
    bool tma_ok = tma_supported();

    // Anlamli boyutlar: kucuk-orta-buyuk kare matrisler
    const int sizes[] = {512, 1024, 2048, 4096};
    const int NS = sizeof(sizes)/sizeof(sizes[0]);

    std::ofstream fcsv(csv); fcsv << "benchmark,N,method,avg_time_ms,gflops\n";

    std::cout << "=== GEMM Tile Fetch: Global vs cp.async vs TMA ===\n";
    std::cout << std::string(110,'-') << "\n";
    std::cout << std::left << std::setw(8)<<"N" << std::setw(16)<<"GLB(ms)" << std::setw(16)<<"CPA(ms)" << std::setw(16)<<"TMA(ms)"
              << std::setw(12)<<"GLB_GF" << std::setw(12)<<"CPA_GF" << std::setw(12)<<"TMA_GF" << std::setw(10)<<"TMA/CPA" << std::setw(10)<<"TMA/GLB" << "\n";
    std::cout << std::string(110,'-') << "\n";

    for (int i=0; i<NS; ++i) {
        int N=sizes[i]; size_t bytes=(size_t)N*N*4;
        float *dA=0,*dB=0,*dC=0;
        if (cudaMalloc(&dA,bytes)!=cudaSuccess||cudaMalloc(&dB,bytes)!=cudaSuccess||cudaMalloc(&dC,bytes)!=cudaSuccess) {
            std::cout << std::setw(8)<<N << "  SKIPPED\n"; cudaFree(dA);cudaFree(dB);cudaFree(dC); continue; }
        CUDA_CHECK(cudaMemset(dA,0x3f,bytes)); CUDA_CHECK(cudaMemset(dB,0x3f,bytes)); CUDA_CHECK(cudaMemset(dC,0,bytes));

        dim3 blk(BN,BM), grd((N+BN-1)/BN,(N+BM-1)/BM);
        float tg = time_kernel([&]{ gemm_global<<<grd,blk>>>(dA,dB,dC,N); });
        float tc = time_kernel([&]{ gemm_cpasync<<<grd,blk>>>(dA,dB,dC,N); });
        float tt = -1.f;
        if (tma_ok) {
            size_t p=(size_t)N*4;
            CUtensorMap mA=make_tma_2d_f32(dA,(uint64_t)N,(uint64_t)N,p,TBK,TBM);
            CUtensorMap mB=make_tma_2d_f32(dB,(uint64_t)N,(uint64_t)N,p,TBN,TBK);
            dim3 blk2(TBN,TBM), grd2((N+TBN-1)/TBN,(N+TBM-1)/TBM);
            tt = time_kernel([&]{ gemm_tma<<<grd2,blk2>>>(mA,mB,dC,N); });
        }
        double ops=2.0*(double)N*N*N;
        float gfg=(float)(ops/(tg/1e3)/1e9), gfc=(float)(ops/(tc/1e3)/1e9), gft=tma_ok?(float)(ops/(tt/1e3)/1e9):-1.f;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8)<<N << std::setw(16)<<tg << std::setw(16)<<tc << std::setw(16)<<(tma_ok?tt:-1.f)
                  << std::setw(12)<<gfg << std::setw(12)<<gfc << std::setw(12)<<(tma_ok?gft:-1.f)
                  << std::setw(10)<<(tma_ok?gft/gfc:-1.f) << std::setw(10)<<(tma_ok?gft/gfg:-1.f) << "\n";
        fcsv << "gemm,"<<N<<",global,"<<tg<<","<<gfg<<"\n";
        fcsv << "gemm,"<<N<<",cp.async,"<<tc<<","<<gfc<<"\n";
        if (tma_ok) fcsv << "gemm,"<<N<<",TMA,"<<tt<<","<<gft<<"\n";
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    std::cout << std::string(110,'-') << "\n";
    std::cout << "\nNotes:\n"
              << "  Uc yol ayni FP32 isini yapar, sadece tile fetch farkli.\n"
              << "  Tile: 32x32. N=4096 -> ~256 MB matris, L2'ye sigmaz.\n"
              << "  Global: register uzerinden. cp.async: DMA. TMA: donanim.\n"
              << "\nCSV: " << csv << "\n";
    return 0;
}
