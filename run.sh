nvcc benchmarks/bench_1d_copy.cu -o bench -Iinclude
./bench
nvcc benchmarks/bench_overlap.cu -o bench -Iinclude
./bench
nvcc benchmarks/bench_2d_stride.cu -o bench -Iinclude
./bench
nvcc benchmarks/bench_gemm.cu -o bench -Iinclude
./bench