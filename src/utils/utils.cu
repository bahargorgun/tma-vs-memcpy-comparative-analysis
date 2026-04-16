#include "utils.h"

float timeMemcpy(void (*func)(float*, float*, size_t, cudaStream_t),
                 float* h, float* d, size_t size) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    func(h, d, size, stream);

    cudaStreamSynchronize(stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return ms;
}