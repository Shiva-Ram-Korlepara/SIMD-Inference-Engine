#ifndef SIMD_ENGINE_H
#define SIMD_ENGINE_H

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>
#include <cmath>
#include <random>

using namespace std;


class LinearAllocator {
private:
    float* arena;
    size_t capacity;
    size_t offset;

public:
    LinearAllocator(size_t total_floats);
    ~LinearAllocator();

    float* allocate(size_t count);
    void reset();
};


class Matrix {
private:
    size_t rows;
    size_t cols;
    float* data;

public:
    Matrix(size_t r, size_t c, LinearAllocator& allocator);

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    float* getData() const { return data; }

    inline float& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    inline const float& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    void fill(float value);
    void fillRandom(float min = -1.0f, float max = 1.0f);
    void zero();
    bool isEquivalent(const Matrix& other, float tolerance = 1e-4f) const;
};


void gemmNaive(const Matrix& A, const Matrix& B, Matrix& C);
void gemmTiled(const Matrix& A, const Matrix& B, Matrix& C, size_t blockSize = 32);
void gemmSimd(const Matrix& A, const Matrix& B, Matrix& C, size_t blockSize = 32);


class Layer {
public:
    virtual ~Layer() {}
    virtual void forward(const Matrix& input, Matrix& output, int mode = 2) = 0;
};

class FullyConnectedLayer : public Layer {
private:
    Matrix weights;
    Matrix biases;
public:
    FullyConnectedLayer(size_t inputSize, size_t outputSize, LinearAllocator& allocator);
    void forward(const Matrix& input, Matrix& output, int mode = 2) override;
};

class Network {
private:
    vector<Layer*> layers;
public:
    ~Network();
    void addLayer(Layer* layer);
    void forward(const Matrix& input, Matrix& output, int mode = 2);
};

#endif // SIMD_ENGINE_H
