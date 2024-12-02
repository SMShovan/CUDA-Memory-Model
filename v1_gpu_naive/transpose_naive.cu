#include "../common/cuda_utils.h"
#include <iostream>

// Naive GPU kernel for matrix transpose
__global__ void transposeNaive(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

int main() {
    // Initialize CUDA
    initializeCUDA();

    // Matrix dimensions
    const int rows = 1024;
    const int cols = 1024;
    const size_t matrix_size = rows * cols;
    const size_t matrix_bytes = matrix_size * sizeof(float);

    // Allocate host memory
    float* h_input = allocateHostMemory<float>(matrix_size);
    float* h_output = allocateHostMemory<float>(matrix_size);

    // Initialize input matrix
    for (size_t i = 0; i < matrix_size; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input = allocateDeviceMemory<float>(matrix_size);
    float* d_output = allocateDeviceMemory<float>(matrix_size);

    // Copy input data to device
    copyToDevice(d_input, h_input, matrix_size);

    // Set grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                 (rows + blockDim.y - 1) / blockDim.y);

    // Create timer
    Timer timer;

    // Launch kernel
    timer.reset();
    transposeNaive<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double elapsed = timer.elapsed();

    // Copy result back to host
    copyToHost(h_output, d_output, matrix_size);

    // Verify result
    bool correct = true;
    for (int i = 0; i < rows && correct; i++) {
        for (int j = 0; j < cols && correct; j++) {
            if (h_output[j * rows + i] != h_input[i * cols + j]) {
                correct = false;
            }
        }
    }

    // Print results
    printf("Naive GPU Matrix Transpose Results:\n");
    printf("Matrix size: %dx%d\n", rows, cols);
    printf("Block size: %dx%d\n", blockDim.x, blockDim.y);
    printf("Grid size: %dx%d\n", gridDim.x, gridDim.y);
    printf("Elapsed time: %.3f ms\n", elapsed);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    freeHostMemory(h_input);
    freeHostMemory(h_output);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_output);

    return 0;
}