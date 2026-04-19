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

int main() {

    const size_t total_size = 512ULL * 1024 * 1024; // 512MB
    const int NUM_TILES = 4;

    size_t tile_size = total_size / NUM_TILES;
    size_t N = tile_size / sizeof(float);

    float* h;
    cudaHostAlloc(&h, total_size, cudaHostAllocDefault); // pinned

    float* d;
    cudaMalloc(&d, tile_size);

    for (size_t i = 0; i < total_size/sizeof(float); i++)
        h[i] = 1.0f;

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int t = 0; t < NUM_TILES; t++) {

        int s = t % 2; // ping-pong streams

        float* h_tile = h + t * (tile_size / sizeof(float));

        cudaMemcpyAsync(d, h_tile, tile_size,
                        cudaMemcpyHostToDevice,
                        streams[s]);

        computeKernel<<<(N+255)/256, 256, 0, streams[s]>>>(d, N);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "----------------------------------\n";
    std::cout << "Total time (pipeline): " << ms << " ms\n";
    std::cout << "----------------------------------\n";

    cudaFree(d);
    cudaFreeHost(h);
}