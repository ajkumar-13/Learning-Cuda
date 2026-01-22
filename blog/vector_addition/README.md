# Blog 2: Vector Addition - Your First CUDA Kernel

> **Writing, launching, and benchmarking a real GPU kernel**

## 📖 Overview

Your first hands-on CUDA programming experience. We write a complete vector addition kernel and compare GPU vs CPU performance.

## 📁 Files

| File | Description |
|------|-------------|
| `02_vector_addition.md` | Main blog post |
| `vector_add.cu` | Basic vector addition kernel |
| `vector_add_benchmark.cu` | Full benchmark with timing |
| `Makefile` | Build automation |

## 🎯 Learning Objectives

After completing this blog, you will:
- Write a `__global__` kernel function
- Allocate GPU memory with `cudaMalloc`
- Transfer data with `cudaMemcpy`
- Launch kernels with `<<<blocks, threads>>>`
- Calculate correct thread indices
- Handle arrays larger than thread count
- Time GPU operations with CUDA events
- Implement proper error checking

## 🔧 Build & Run

```bash
# Using Makefile
make all
./vector_add
./vector_add_benchmark

# Manual compilation
nvcc -O3 -o vector_add vector_add.cu
nvcc -O3 -o vector_add_benchmark vector_add_benchmark.cu
```

## 📊 Expected Output

```
Vector size: 1000000
CPU time: 2.34 ms
GPU time: 0.15 ms (including transfer)
GPU kernel only: 0.02 ms
Speedup: 15.6x
Result: PASS
```

## 📚 Prerequisites

- [Blog 1: Introduction to CUDA](../introduction/)

## ➡️ Next

Continue to [Blog 3: Matrix Multiplication](../matrix_multiplication/) - 2D indexing and shared memory!
