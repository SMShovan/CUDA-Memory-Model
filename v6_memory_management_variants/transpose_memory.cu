#include "../common/cuda_utils.h"
#include <iostream>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Texture memory reference for input matrix
texture<float, 2, cudaReadModeElementType> tex_input;

// Constant memory for tile dimensions
__constant__ int c_rows;
__constant__ int c_cols;

// Kernel using texture memory for reading
__global__ void transposeTexture(float* output) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile from texture memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < c_cols && y + j < c_rows) {
            tile[threadIdx.y + j][threadIdx.x] = tex2D(tex_input, x, y + j);
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile to output with coalesced access
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < c_rows && y + j < c_cols) {
            output[(y + j) * c_rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Kernel using read-only cache
__global__ void transposeReadOnly(const float* __restrict__ input, float* output) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile using read-only cache
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < c_cols && y + j < c_rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * c_cols + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile to output
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < c_rows && y + j < c_cols) {
            output[(y + j) * c_rows + x] = tile[threadIdx.x][threadIdx.y + j];
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

    // Copy dimensions to constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_rows, &rows, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_cols, &cols, sizeof(int)));

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

    // Set up texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cu_array;
    CHECK_CUDA_ERROR(cudaMallocArray(&cu_array, &channelDesc, cols, rows));
    CHECK_CUDA_ERROR(cudaMemcpyToArray(cu_array, 0, 0, h_input, matrix_bytes, cudaMemcpyHostToDevice));
    tex_input.addressMode[0] = cudaAddressModeClamp;
    tex_input.addressMode[1] = cudaAddressModeClamp;
    tex_input.filterMode = cudaFilterModePoint;
    tex_input.normalized = false;
    CHECK_CUDA_ERROR(cudaBindTextureToArray(tex_input, cu_array));

    // Set grid and block dimensions
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((cols + TILE_DIM - 1) / TILE_DIM,
                 (rows + TILE_DIM - 1) / TILE_DIM);

    // Create timer
    Timer timer;
    double elapsed;

    // Test texture memory version
    timer.reset();
    transposeTexture<<<gridDim, blockDim>>>(d_output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    elapsed = timer.elapsed();
    printf("\nTexture Memory Version:\n");
    printf("Elapsed time: %.3f ms\n", elapsed);

    // Test read-only cache version
    timer.reset();
    transposeReadOnly<<<gridDim, blockDim>>>(d_input, d_output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    elapsed = timer.elapsed();
    printf("\nRead-Only Cache Version:\n");
    printf("Elapsed time: %.3f ms\n", elapsed);

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
    printf("\nMemory Management Variants Results:\n");
    printf("Matrix size: %dx%d\n", rows, cols);
    printf("Tile size: %dx%d\n", TILE_DIM, TILE_DIM);
    printf("Block size: %dx%d\n", blockDim.x, blockDim.y);
    printf("Grid size: %dx%d\n", gridDim.x, gridDim.y);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    CHECK_CUDA_ERROR(cudaUnbindTexture(tex_input));
    CHECK_CUDA_ERROR(cudaFreeArray(cu_array));
    freeHostMemory(h_input);
    freeHostMemory(h_output);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_output);

    return 0;
}