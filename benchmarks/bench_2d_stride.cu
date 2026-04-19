#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

float run2D(size_t width, size_t height, size_t pitch) {

    float *h, *d;
    size_t size = pitch * height;

    cudaHostAlloc(&h, size, cudaHostAllocDefault);
    cudaMalloc(&d, size);

    // init
    for (size_t i = 0; i < (pitch/sizeof(float))*height; i++)
        h[i] = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy2D(d, pitch,
                 h, pitch,
                 width * sizeof(float),
                 height,
                 cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d);
    cudaFreeHost(h);

    return ms;
}

int main() {

    std::cout << "------------------------------------------\n";
    std::cout << "| Width | Height | Time (ms)             |\n";
    std::cout << "------------------------------------------\n";

    int widths[]  = {512, 1024, 2048};
    int heights[] = {512, 1024, 2048};

    for (int w : widths) {
        for (int h : heights) {

            size_t pitch = w * sizeof(float) * 2; // add stride!

            float ms = run2D(w, h, pitch);

            std::cout << "| "
                      << std::setw(5) << w << " | "
                      << std::setw(6) << h << " | "
                      << std::setw(10) << ms << " |\n";
        }
    }

    std::cout << "------------------------------------------\n";
}