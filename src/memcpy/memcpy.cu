#include <cuda_runtime.h>

void memcpy_impl(float* h, float* d, size_t size, cudaStream_t stream) {
    cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, stream);
}