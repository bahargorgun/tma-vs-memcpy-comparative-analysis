#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

__global__ void computeKernel(float* data, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];
        for (int i = 0; i < 200; i++) {
            x = x * 1.0001f + 0.0001f;
        }
        data[idx] = x;
    }
}

float runSequential(float* h, float* d, size_t size, size_t N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
    computeKernel<<<(N+255)/256, 256>>>(d, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

float runOverlap(float* h, float* d, size_t size, size_t N) {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, s1);
    computeKernel<<<(N+255)/256, 256, 0, s2>>>(d, N);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {

    size_t size = 512ULL * 1024 * 1024; // 512MB
    size_t N = size / sizeof(float);

    float* d;
    cudaMalloc(&d, size);

    float* h;
    cudaHostAlloc(&h, size, cudaHostAllocDefault); // pinned!

    for (size_t i = 0; i < N; i++) h[i] = 1.0f;

    float t_seq = runSequential(h, d, size, N);
    float t_ov  = runOverlap(h, d, size, N);

    std::cout << "--------------------------------------\n";
    std::cout << "Sequential: " << t_seq << " ms\n";
    std::cout << "Overlap   : " << t_ov  << " ms\n";
    std::cout << "Speedup   : " << t_seq / t_ov << "x\n";
    std::cout << "--------------------------------------\n";

    cudaFree(d);
    cudaFreeHost(h);
}