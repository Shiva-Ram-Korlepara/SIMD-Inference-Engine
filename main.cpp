#include <iostream>
#include <chrono>
#include "simd_engine.h"

using namespace std;
using namespace std::chrono;

void runBenchmark(size_t M, size_t K, size_t N) {
    cout << "\nRunning GEMM test: " << M << "x" << K << " * " << K << "x" << N << "\n";
    size_t totalFloats = M * K + K * N + 3 * (M * N);
    LinearAllocator allocator(totalFloats + 1024);

    Matrix A(M, K, allocator);
    Matrix B(K, N, allocator);
    Matrix C_naive(M, N, allocator);
    Matrix C_tiled(M, N, allocator);
    Matrix C_simd(M, N, allocator);

    A.fillRandom(-1.0f, 1.0f);
    B.fillRandom(-1.0f, 1.0f);

    auto start = high_resolution_clock::now();
    gemmNaive(A, B, C_naive);
    auto end = high_resolution_clock::now();
    auto durationNaive = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    gemmTiled(A, B, C_tiled, 32);
    end = high_resolution_clock::now();
    auto durationTiled = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    gemmSimd(A, B, C_simd, 32);
    end = high_resolution_clock::now();
    auto durationSimd = duration_cast<microseconds>(end - start).count();

    bool tiledCorrect = C_naive.isEquivalent(C_tiled, 1e-3f);
    bool simdCorrect = C_naive.isEquivalent(C_simd, 1e-3f);

    cout << "Method\t\tTime (us)\tStatus\n";
    
    cout << "Naive\t\t" << durationNaive << "\t\tBaseline\n";
    
    cout << "Tiled\t\t" << durationTiled << "\t\t";
    if (tiledCorrect) {
        cout << "PASS\n";
    }
    else {
        cout << "FAIL\n";
    }

    cout << "SIMD (AVX2)\t" << durationSimd << "\t\t";
    if (simdCorrect) {
        cout << "PASS\n";
    }
    else {
        cout << "FAIL\n";
    }
    
    cout << "\n";
}

int main() {
    cout << "Starting Bare-Metal SIMD Inference Engine Benchmarks\n\n";

    runBenchmark(128, 128, 128);
    runBenchmark(512, 512, 512);
    runBenchmark(1024, 1024, 1024);

    return 0;
}
