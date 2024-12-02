#include "../common/cuda_utils.h"
#include <iostream>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define UNROLL_FACTOR 4

// Unrolled shared memory kernel for matrix transpose
__global__ void transposeUnrolled(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile into shared memory with unrolled loops
    #pragma unroll UNROLL_FACTOR
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Calculate transposed indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile to output with unrolled loops
    #pragma unroll UNROLL_FACTOR
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && y + j < cols) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
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
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((cols + TILE_DIM - 1) / TILE_DIM,
                 (rows + TILE_DIM - 1) / TILE_DIM);

    // Create timer
    Timer timer;

    // Launch kernel
    timer.reset();
    transposeUnrolled<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
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
    printf("Unrolled Matrix Transpose Results:\n");
    printf("Matrix size: %dx%d\n", rows, cols);
    printf("Tile size: %dx%d\n", TILE_DIM, TILE_DIM);
    printf("Block size: %dx%d\n", blockDim.x, blockDim.y);
    printf("Grid size: %dx%d\n", gridDim.x, gridDim.y);
    printf("Unroll factor: %d\n", UNROLL_FACTOR);
    printf("Elapsed time: %.3f ms\n", elapsed);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    freeHostMemory(h_input);
    freeHostMemory(h_output);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_output);

    return 0;
}