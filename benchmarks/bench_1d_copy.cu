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
    const int RUNS = 10;

    std::cout << "-------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| Size(MB) | Time_P(ms) | Time_Pin(ms) | BW_P(GB/s) | BW_Pin(GB/s) | Speedup                                |\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------\n";

    for (int i = 0; i < numSizes; i++) {

        size_t size = sizes[i];
        size_t N = size / sizeof(float);

        float* d_A;
        if (cudaMalloc(&d_A, size) != cudaSuccess) {
            std::cout << "| " << std::setw(8) << (size / (1024*1024))
                      << " | SKIPPED (no VRAM)\n";
            continue;
        }

        // ===== PAGEABLE =====
        float* h_pageable = (float*)malloc(size);
        for (size_t j = 0; j < N; j++) h_pageable[j] = 1.0f;

        cudaMemcpy(d_A, h_pageable, size, cudaMemcpyHostToDevice);

        float total_pageable = 0;
        for (int r = 0; r < RUNS; r++)
            total_pageable += runMemcpy(h_pageable, d_A, size);

        float avg_pageable = total_pageable / RUNS;
        float bw_pageable = (size / 1e9f) / (avg_pageable / 1e3f);

        // ===== PINNED =====
        float* h_pinned;
        cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);

        for (size_t j = 0; j < N; j++) h_pinned[j] = 1.0f;

        cudaMemcpy(d_A, h_pinned, size, cudaMemcpyHostToDevice);

        float total_pinned = 0;
        for (int r = 0; r < RUNS; r++)
            total_pinned += runMemcpy(h_pinned, d_A, size);

        float avg_pinned = total_pinned / RUNS;
        float bw_pinned = (size / 1e9f) / (avg_pinned / 1e3f);

        float speedup = bw_pinned / bw_pageable;

        std::cout << "| "
                  << std::setw(8) << (size / (1024*1024)) << " | "
                  << std::setw(10) << avg_pageable << " | "
                  << std::setw(12) << avg_pinned << " | "
                  << std::setw(11) << bw_pageable << " | "
                  << std::setw(13) << bw_pinned << " | "
                  << std::setw(10) << speedup << "x |\n";

        cudaFree(d_A);
        free(h_pageable);
        cudaFreeHost(h_pinned);
    }

    std::cout << "-------------------------------------------------------------------------------------------------------------\n";

    return 0;
}