# SIMD Inference Engine

This project is a bare-metal C++ implementation of a convolutional neural network (CNN) inference engine. The primary goal is to optimize the forward pass of matrix operations by eliminating high-level abstractions, managing memory directly, and utilizing hardware-specific CPU features.

## How to Use

The engine is built entirely in C++ and uses CMake for its build system. There are no external dependencies like BLAS or Eigen.

### Building
To compile the project, run the following commands in your terminal:

```
cmake .
make
```

### Running
After building, execute the benchmark driver:

```
./simd_engine
```

This will run a series of matrix multiplication (GEMM) benchmarks across different dimensions (128x128, 512x512, and 1024x1024) and print the execution times and correctness validation to the console.

## Implementation Details

The project focuses on three major optimizations for neural network execution:

1. Custom Memory Arena
Standard dynamic allocations (using `new` or `std::vector`) can scatter memory across the heap and cause fragmentation. This project uses a custom `LinearAllocator` that grabs a single contiguous block of memory at startup. Data is aligned to 32-byte boundaries using `posix_memalign`, which is critical for efficient vector loading.

2. Cache-Aware Tiling
A naive matrix multiplication traverses memory in a way that constantly evicts data from the CPU L1 cache before it can be reused. To solve this, the `gemmTiled` function divides the matrices into small blocks (e.g., 32x32) so that the working set fits entirely within the L1 data cache. 

3. SIMD Vectorization
In the `gemmSimd` implementation, standard scalar math is replaced with Advanced Vector Extensions (AVX2). Using Fused Multiply-Add (FMA) intrinsics (`_mm256_fmadd_ps`), the CPU calculates 8 floating-point operations in a single hardware cycle. This dramatically reduces latency for the compute-heavy dot products required in CNN layers.
