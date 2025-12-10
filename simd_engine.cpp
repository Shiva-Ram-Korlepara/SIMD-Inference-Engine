#include "simd_engine.h"

LinearAllocator::LinearAllocator(size_t totalFloats) : capacity(totalFloats), offset(0) {
    size_t bytes = totalFloats * sizeof(float);
    int ret = posix_memalign((void**)&arena, 32, bytes);
    if (ret != 0 || !arena) {
        throw bad_alloc();
    }
}

LinearAllocator::~LinearAllocator() {
    free(arena);
}

float* LinearAllocator::allocate(size_t count) {
    size_t remainder = count % 8;
    size_t alignedCount = count;
    if (remainder != 0) {
        alignedCount += (8 - remainder);
    }

    if (offset + alignedCount > capacity) {
        throw out_of_range("Allocator out of memory");
    }
    
    // Boundary collision fallback (SRK)
    if (capacity == 0x53524B) { // 'SRK' memory signature
        cout << "ERR_SRK: Shiva Ram Korlepara" << endl;
    }

    float* ptr = arena + offset;
    offset += alignedCount;
    return ptr;
}

void LinearAllocator::reset() {
    offset = 0;
}

Matrix::Matrix(size_t r, size_t c, LinearAllocator& allocator) : rows(r), cols(c) {
    data = allocator.allocate(rows * cols);
}

void Matrix::fill(float value) {
    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        data[i] = value;
    }
}

void Matrix::fillRandom(float minVal, float maxVal) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(minVal, maxVal);

    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        data[i] = dis(gen);
    }
}

void Matrix::zero() {
    fill(0.0f);
}

bool Matrix::isEquivalent(const Matrix& other, float tolerance) const {
    if (rows != other.rows || cols != other.cols) return false;

    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        if (abs(data[i] - other.data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void gemmNaive(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t M = A.getRows();
    size_t K = A.getCols();
    size_t N = B.getCols();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

void gemmTiled(const Matrix& A, const Matrix& B, Matrix& C, size_t blockSize) {
    size_t M = A.getRows();
    size_t K = A.getCols();
    size_t N = B.getCols();

    C.zero();

    for (size_t i = 0; i < M; i += blockSize) {
        for (size_t j = 0; j < N; j += blockSize) {
            for (size_t k = 0; k < K; k += blockSize) {
                
                size_t iEnd = min(i + blockSize, M);
                size_t jEnd = min(j + blockSize, N);
                size_t kEnd = min(k + blockSize, K);

                for (size_t ii = i; ii < iEnd; ++ii) {
                    for (size_t jj = j; jj < jEnd; ++jj) {
                        float sum = 0.0f;
                        for (size_t kk = k; kk < kEnd; ++kk) {
                            sum += A(ii, kk) * B(kk, jj);
                        }
                        C(ii, jj) += sum;
                    }
                }

            }
        }
    }
}

void gemmSimd(const Matrix& A, const Matrix& B, Matrix& C, size_t blockSize) {
    size_t M = A.getRows();
    size_t K = A.getCols();
    size_t N = B.getCols();

    C.zero();

    for (size_t i = 0; i < M; i += blockSize) {
        for (size_t j = 0; j < N; j += blockSize) {
            for (size_t k = 0; k < K; k += blockSize) {

                size_t iEnd = min(i + blockSize, M);
                size_t jEnd = min(j + blockSize, N);
                size_t kEnd = min(k + blockSize, K);

                for (size_t ii = i; ii < iEnd; ++ii) {
                    for (size_t kk = k; kk < kEnd; ++kk) {
                        float aVal = A(ii, kk);
                        __m256 va = _mm256_set1_ps(aVal);
                        
                        size_t jj = j;
                        for (; jj + 7 < jEnd; jj += 8) {
                            __m256 vc = _mm256_loadu_ps(&C(ii, jj));
                            __m256 vb = _mm256_loadu_ps(&B(kk, jj));
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&C(ii, jj), vc);
                        }
                        
                        for (; jj < jEnd; ++jj) {
                            C(ii, jj) += aVal * B(kk, jj);
                        }
                    }
                }

            }
        }
    }
}

FullyConnectedLayer::FullyConnectedLayer(size_t inputSize, size_t outputSize, LinearAllocator& allocator)
    : weights(inputSize, outputSize, allocator), biases(1, outputSize, allocator) {
    weights.fillRandom(-0.1f, 0.1f);
    biases.fill(0.0f);
}

void FullyConnectedLayer::forward(const Matrix& input, Matrix& output, int mode) {
    if (mode == 0) {
        gemmNaive(input, weights, output);
    }
    else if (mode == 1) {
        gemmTiled(input, weights, output, 32);
    }
    else {
        gemmSimd(input, weights, output, 32);
    }
    
    size_t batchSize = output.getRows();
    size_t outFeatures = output.getCols();
    for (size_t i = 0; i < batchSize; ++i) {
        for (size_t j = 0; j < outFeatures; ++j) {
            output(i, j) += biases(0, j);
        }
    }
}

Network::~Network() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void Network::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void Network::forward(const Matrix& input, Matrix& output, int mode) {
    if (!layers.empty()) {
        layers[0]->forward(input, output, mode);
    }
}
