# Blog 1: Introduction to CUDA

> **Understanding GPU Architecture and the CUDA Programming Model**

## 📖 Overview

This is the entry point to the CUDA Learning series. No prior GPU programming experience required - just C/C++ knowledge.

## 📁 Files

| File | Description |
|------|-------------|
| `01_introduction.md` | Main blog post |
| `hello_cuda.cu` | Minimal "Hello CUDA" program |
| `device_query.cu` | Query GPU properties |

## 🎯 Learning Objectives

After completing this blog, you will:
- Understand CPU vs GPU architecture differences
- Know the CUDA thread hierarchy (threads → blocks → grids)
- Understand Streaming Multiprocessors (SMs) and warps
- Know what warp divergence is and why it matters
- Understand the GPU memory hierarchy
- Know how CUDA code compiles (PTX → cubin)

## 🔧 Build & Run

```bash
# Compile
nvcc -o hello_cuda hello_cuda.cu
nvcc -o device_query device_query.cu

# Run
./hello_cuda
./device_query
```

## 📚 Prerequisites

- C/C++ programming knowledge
- CUDA Toolkit installed (11.8+)
- NVIDIA GPU with compute capability 5.0+

## ➡️ Next

Continue to [Blog 2: Vector Addition](../vector_addition/) - Your first real CUDA kernel!
