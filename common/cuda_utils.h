#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// Timer class for performance measurements
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_);
        return duration.count() / 1000.0; // Convert to milliseconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// CUDA error checking function
inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

// Initialize CUDA device
inline void initializeCUDA() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        exit(1);
    }
    
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using CUDA Device %d: %s\n", 0, deviceProp.name);
}

// Memory allocation helper functions
template<typename T>
inline T* allocateDeviceMemory(size_t count) {
    T* ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline T* allocateHostMemory(size_t count) {
    T* ptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
}

// Memory deallocation helper functions
template<typename T>
inline void freeDeviceMemory(T* ptr) {
    if (ptr) CHECK_CUDA_ERROR(cudaFree(ptr));
}

template<typename T>
inline void freeHostMemory(T* ptr) {
    if (ptr) CHECK_CUDA_ERROR(cudaFreeHost(ptr));
}

// Memory copy helper functions
template<typename T>
inline void copyToDevice(T* dst, const T* src, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
inline void copyToHost(T* dst, const T* src, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

#endif // CUDA_UTILS_H