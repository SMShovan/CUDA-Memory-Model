#include "../common/cuda_utils.h"
#include <iostream>
#include <vector>

// CPU implementation of matrix transpose
void transposeMatrix(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Verify transpose result
bool verifyResult(const float* input, const float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (output[j * rows + i] != input[i * cols + j]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Matrix dimensions
    const int rows = 1024;
    const int cols = 1024;
    const size_t matrix_size = rows * cols;

    // Allocate host memory
    float* h_input = new float[matrix_size];
    float* h_output = new float[matrix_size];

    // Initialize input matrix
    for (size_t i = 0; i < matrix_size; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Create timer
    Timer timer;

    // Perform transpose
    timer.reset();
    transposeMatrix(h_input, h_output, rows, cols);
    double elapsed = timer.elapsed();

    // Verify result
    bool correct = verifyResult(h_input, h_output, rows, cols);
    
    // Print results
    printf("CPU Matrix Transpose Results:\n");
    printf("Matrix size: %dx%d\n", rows, cols);
    printf("Elapsed time: %.3f ms\n", elapsed);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    delete[] h_input;
    delete[] h_output;

    return 0;
}