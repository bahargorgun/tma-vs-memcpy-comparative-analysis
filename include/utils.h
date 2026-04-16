#pragma once
#include <cuda_runtime.h>

float timeMemcpy(void (*func)(float*, float*, size_t, cudaStream_t),float* h, float* d, size_t size);