#!/bin/bash

# Exit on any error
set -e

# CUDA compiler
NVCC="nvcc"

# Common flags
CUDA_FLAGS="-O3 -std=c++14"

# Build all versions
echo "Building all implementations..."

# Build common utilities
mkdir -p common

# Build CPU baseline version
echo "Building CPU baseline version..."
mkdir -p v0_cpu
${NVCC} ${CUDA_FLAGS} v0_cpu/transpose_cpu.cpp -o v0_cpu/transpose_cpu

# Build naive GPU version
echo "Building naive GPU version..."
mkdir -p v1_gpu_naive
${NVCC} ${CUDA_FLAGS} v1_gpu_naive/transpose_naive.cu -o v1_gpu_naive/transpose_naive

# Build shared memory tiling version
echo "Building shared memory tiling version..."
mkdir -p v2_gpu_sharedmem_tiling
${NVCC} ${CUDA_FLAGS} v2_gpu_sharedmem_tiling/transpose_shared.cu -o v2_gpu_sharedmem_tiling/transpose_shared

# Build bank conflict resolution version
echo "Building bank conflict resolution version..."
mkdir -p v3_gpu_sharedmem_banks
${NVCC} ${CUDA_FLAGS} v3_gpu_sharedmem_banks/transpose_banks.cu -o v3_gpu_sharedmem_banks/transpose_banks

# Build loop unrolling version
echo "Building loop unrolling version..."
mkdir -p v4_gpu_transpose_unrolling
${NVCC} ${CUDA_FLAGS} v4_gpu_transpose_unrolling/transpose_unroll.cu -o v4_gpu_transpose_unrolling/transpose_unroll

# Build diagonal optimization version
echo "Building diagonal optimization version..."
mkdir -p v5_gpu_diagonal_optimization
${NVCC} ${CUDA_FLAGS} v5_gpu_diagonal_optimization/transpose_diagonal.cu -o v5_gpu_diagonal_optimization/transpose_diagonal

# Build memory management variants
echo "Building memory management variants..."
mkdir -p v6_memory_management_variants
${NVCC} ${CUDA_FLAGS} v6_memory_management_variants/transpose_memory.cu -o v6_memory_management_variants/transpose_memory

echo "Build complete!"

# Make the script executable
chmod +x build_and_run.sh

echo "Running performance tests..."

# Run all versions with sample matrices
for version in v*/*; do
    if [ -x "$version" ]; then
        echo "\nRunning $version:"
        ./$version
    fi
done

echo "\nAll tests complete!"