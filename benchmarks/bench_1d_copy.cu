#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

float runMemcpy(float* h_A, float* d_A, size_t size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {

    size_t sizes[] = {
        16ULL * 1024 * 1024,
        64ULL * 1024 * 1024,
        128ULL * 1024 * 1024,
        256ULL * 1024 * 1024,
        512ULL * 1024 * 1024,
        1ULL * 1024 * 1024 * 1024,
        2ULL * 1024 * 1024 * 1024
    };

    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    const int RUNS = 5;

    // ===== TABLE HEADER =====
    std::cout << "-----------------------------------------------------------\n";
    std::cout << "| Size (MB) | Avg Time (ms) | Bandwidth (GB/s)           |\n";
    std::cout << "-----------------------------------------------------------\n";

    for (int i = 0; i < numSizes; i++) {

        size_t size = sizes[i];
        size_t N = size / sizeof(float);

        float* h_A = (float*)malloc(size);
        float* d_A;

        cudaError_t err = cudaMalloc(&d_A, size);

        if (err != cudaSuccess) {
            std::cout << "| " << std::setw(9) << (size / (1024*1024))
                      << " |   SKIPPED (no VRAM)                |\n";
            free(h_A);
            continue;
        }

        // initialize
        for (size_t j = 0; j < N; j++) {
            h_A[j] = 1.0f;
        }

        // warm-up
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        float total_ms = 0.0f;

        for (int r = 0; r < RUNS; r++) {
            total_ms += runMemcpy(h_A, d_A, size);
        }

        float avg_ms = total_ms / RUNS;

        float gb = size / 1e9f;
        float bandwidth = gb / (avg_ms / 1e3f);

        std::cout << "| "
                  << std::setw(9) << (size / (1024*1024)) << " | "
                  << std::setw(13) << avg_ms << " | "
                  << std::setw(25) << bandwidth << " |\n";

        cudaFree(d_A);
        free(h_A);
    }

    std::cout << "-----------------------------------------------------------\n";

    return 0;
}