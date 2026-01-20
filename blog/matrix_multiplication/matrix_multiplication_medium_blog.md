# From Loops to Launches: Matrix Multiplication on the GPU

*Your second CUDA kernel—and the foundation of deep learning*

---

## Introduction

In the [previous post](./vector_addition_medium_blog.md), we wrote our first CUDA kernel: vector addition. Each thread computed one element: `C[i] = A[i] + B[i]`. Simple, parallel, and a great starting point.

Now we're stepping up to **matrix multiplication**—the workhorse of linear algebra and the beating heart of deep learning. Every neural network forward pass, every transformer attention layer, every convolution—at the core, they're all matrix multiplications.

Here's the problem: given two matrices **A** (M × K) and **B** (K × N), compute their product **C** (M × N):

```
C[row][col] = Σ (A[row][k] × B[k][col])  for k = 0 to K-1
```

Each element of C is a **dot product** of a row from A and a column from B.

On a CPU, this is an O(M × N × K) operation with three nested loops. For large matrices (say, 4096 × 4096), that's **68 billion** multiply-add operations. Even at 100 GFLOPS, that's nearly a second per multiplication.

A GPU can do this in **milliseconds**—if you structure the computation correctly.

---

## Why Matrix Multiplication is Perfect for GPUs

Matrix multiplication has three properties that make it GPU-friendly:

1. **Massive parallelism**: Each element of C can be computed independently
2. **Regular memory access**: Rows and columns have predictable patterns
3. **High arithmetic intensity**: Lots of math operations per byte loaded (when optimized)

The challenge? Each output element requires reading an entire row of A and an entire column of B. That's a lot of memory traffic—and memory bandwidth is the GPU's bottleneck.

> 💡 **Key insight:** Naive matrix multiplication is **memory-bound**. The real art of GPU optimization is reducing redundant memory loads through **tiling** and **shared memory**. We'll start simple, then show the optimized version.

---

## The Thread Hierarchy for 2D Problems

In vector addition, we used a 1D grid of 1D blocks. For matrices, we use a **2D grid of 2D blocks**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRID (gridDim.x × gridDim.y)                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                │
│  │Block    │ │Block    │ │Block    │ │Block    │ ...            │
│  │(0,0)    │ │(1,0)    │ │(2,0)    │ │(3,0)    │                │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                │
│  │Block    │ │Block    │ │Block    │ │Block    │ ...            │
│  │(0,1)    │ │(1,1)    │ │(2,1)    │ │(3,1)    │                │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                │
│  ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

Each **block** covers a tile of the output matrix C. Each **thread** within a block computes one element of C.

### The 2D Index Formula

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

This is the 2D equivalent of our 1D formula `i = blockIdx.x * blockDim.x + threadIdx.x`.

---

## Visual: The Complete Thread Hierarchy

Here's how everything maps together (recreate this as a diagram):

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║           CUDA MATRIX MULTIPLICATION KERNEL - THREAD HIERARCHY                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   Matrix A (M×K)      Matrix B (K×N)      Matrix C (M×N)                      ║
║   ┌───────────┐       ┌───────────┐       ┌───────────────────┐               ║
║   │           │       │           │       │ ┌─────┬─────┬───┐ │               ║
║   │    M      │   ×   │     K     │   =   │ │Block│Block│...│ │               ║
║   │    ×      │       │     ×     │       │ │(0,0)│(1,0)│   │ │  ← GRID       ║
║   │    K      │       │     N     │       │ ├─────┼─────┼───┤ │               ║
║   │           │       │           │       │ │Block│Block│...│ │               ║
║   └───────────┘       └───────────┘       │ │(0,1)│(1,1)│   │ │               ║
║                                           │ └─────┴─────┴───┘ │               ║
║                                           └───────────────────┘               ║
║                                                                               ║
║   Each cell in grid = 1 Block                                                 ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   BLOCK DETAIL (blockDim.x=16, blockDim.y=16 = 256 threads)                   ║
║                                                                               ║
║   ┌─────────────────────┬─────────────────────┐                               ║
║   │ Warp 0 (threads 0-31)│ Warp 1 (threads 32-63)│                            ║
║   │  threadIdx.x: 0-15   │  threadIdx.x: 0-15    │                            ║
║   │  threadIdx.y: 0-1    │  threadIdx.y: 2-3     │                            ║
║   ├─────────────────────┼─────────────────────┤                               ║
║   │ Warp 2 (threads 64-95)│ Warp 3 (threads 96-127)│                          ║
║   │  threadIdx.x: 0-15   │  threadIdx.x: 0-15    │                            ║
║   │  threadIdx.y: 4-5    │  threadIdx.y: 6-7     │                            ║
║   ├─────────────────────┼─────────────────────┤                               ║
║   │ Warp 4 (128-159)    │ Warp 5 (160-191)     │                              ║
║   ├─────────────────────┼─────────────────────┤                               ║
║   │ Warp 6 (192-223)    │ Warp 7 (224-255)     │                              ║
║   └─────────────────────┴─────────────────────┘                               ║
║                                                                               ║
║   8 Warps × 32 threads = 256 threads/block                                    ║
║   (32 threads execute in lockstep - SIMT)                                     ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   THREAD COMPUTATION (single element of C)                                    ║
║                                                                               ║
║   // Global thread indices                                                    ║
║   row = blockIdx.y * blockDim.y + threadIdx.y                                 ║
║   col = blockIdx.x * blockDim.x + threadIdx.x                                 ║
║                                                                               ║
║   // Dot product for C[row][col]                                              ║
║   float sum = 0.0f;                                                           ║
║   for (int k = 0; k < K; k++)                                                 ║
║       sum += A[row*K + k] * B[k*N + col];                                     ║
║   C[row*N + col] = sum;                                                       ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   HIERARCHY SUMMARY                                    KERNEL LAUNCH          ║
║                                                                               ║
║   GPU                                                  dim3 block(16, 16);    ║
║    └── Grid                                            dim3 grid(N/16, M/16); ║
║         ├── Block[0,0] Block[0,1] ...                                         ║
║         ├── Block[1,0] Block[1,1] ...                  matmul<<<grid, block>>>║
║         │    └── Warp 0: threads 0-31                      (A, B, C);         ║
║         │    └── Warp 1: threads 32-63                                        ║
║         │    └── ... (8 warps total)                   // Total threads =     ║
║         │         └── Thread: C[i][j]                  //   M × N             ║
║         └── ...                                                               ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   KEY CONCEPTS:                                                               ║
║   1. Grid covers output matrix C - each block computes a TILE_SIZE×TILE_SIZE  ║
║   2. Block = 256 threads (16×16) executing on same SM, sharing resources      ║
║   3. Warp = 32 threads executing SIMT (same instruction, multiple threads)    ║
║   4. Thread computes one element: C[row][col] = dot(A[row,:], B[:,col])       ║
║   5. Memory coalescing: adjacent threads access adjacent memory for efficiency║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## The Naive Kernel: One Thread Per Output Element

Let's start with the simplest approach: each thread computes one element of C by performing a full dot product.

```cpp
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < M && col < N)
    {
        float sum = 0.0f;
        
        // Dot product: row of A × column of B
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}
```

### Launch Configuration

```cpp
#define BLOCK_SIZE 16

dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // 256 threads per block
dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
             (M + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Ceiling division

matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
```

### Why 16×16?

- 16 × 16 = 256 threads, a common sweet spot
- Divides evenly into warps (256 / 32 = 8 warps)
- Good balance between occupancy and register usage

---

## The Problem with Naive: Memory Bandwidth

Let's analyze the memory access pattern:

For a matrix multiplication of size 1024 × 1024 × 1024:
- Each thread reads **1024 elements from A** (one row)
- Each thread reads **1024 elements from B** (one column)
- Total reads per thread: 2048 floats = 8 KB
- Total threads: 1024 × 1024 = 1 million
- Total memory reads: **8 TB** of data movement!

But wait—matrix A is only 4 MB and B is only 4 MB. We're reading the same data **millions of times**.

> ⚠️ **The bottleneck:** In naive matmul, every thread re-reads the same rows and columns from global memory. Global memory is slow (~500 GB/s on modern GPUs), so we're completely memory-bound.

---

## The Solution: Tiled Matrix Multiplication with Shared Memory

The key insight: **threads in the same block can share data**.

CUDA provides **shared memory**—a fast, programmer-managed cache that's shared among all threads in a block. Instead of each thread loading its own data from global memory, we:

1. **Load a tile** of A and B into shared memory (one load per thread)
2. **Synchronize** all threads in the block
3. **Compute** partial results using the fast shared memory
4. **Repeat** for all tiles along the K dimension

```
┌────────────────────────────────────────────────────────────────────────┐
│                    TILED MATRIX MULTIPLICATION                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Matrix A                    Matrix B                   Matrix C       │
│  ┌──────────────────┐       ┌──────────────────┐       ┌──────────┐   │
│  │    │Tile│    │   │       │         │        │       │          │   │
│  │    │ 0  │    │   │       │─────────┼────────│       │   Tile   │   │
│  │────┼────┼────┼───│   ×   │  Tile 0 │ Tile 1 │   =   │  Output  │   │
│  │    │Tile│    │   │       │─────────┼────────│       │          │   │
│  │    │ 1  │    │   │       │         │        │       │          │   │
│  └──────────────────┘       └──────────────────┘       └──────────┘   │
│                                                                        │
│  Step 1: Load Tile 0 of A and Tile 0 of B into shared memory          │
│  Step 2: Compute partial dot products (all threads, fast shared mem)  │
│  Step 3: Synchronize                                                   │
│  Step 4: Load Tile 1 of A and Tile 1 of B into shared memory          │
│  Step 5: Accumulate more partial dot products                          │
│  ...repeat for all tiles...                                            │
│  Step N: Write final result to C                                       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### The Tiled Kernel

```cpp
#define TILE_SIZE 16

__global__ void matmulTiled(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C,
                            int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Thread's position in the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Load tile from A: A[row * K + (t*TILE + threadIdx.x)]
        // Adjacent threads (varying threadIdx.x) load adjacent floats → COALESCED
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile from B: B[bRow * N + col], where col = blockIdx.x*TILE + threadIdx.x
        // Adjacent threads (varying threadIdx.x) load adjacent floats → COALESCED
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Wait for all threads to finish loading
        __syncthreads();
        
        // Compute partial dot product for this tile
        // TILE_SIZE is compile-time constant, so compiler can unroll this loop
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Wait before loading next tile (prevent data race)
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}
```

> 💡 **Why `__restrict__`?** This keyword tells the compiler that `A`, `B`, and `C` don't overlap in memory, enabling more aggressive optimizations like caching loads via L1/texture cache (LDG instructions).
```

### Why Tiling Works

| Aspect | Naive | Tiled |
|--------|-------|-------|
| Global memory reads per element | 2K | 2K / TILE_SIZE |
| Memory reuse | None | TILE_SIZE × reuse |
| Shared memory usage | 0 | TILE_SIZE² × 2 × 4 bytes |
| Achieved bandwidth | ~10% peak | ~60-80% peak |

For TILE_SIZE = 16, we reduce global memory traffic by **16×**.

### Why This Loading Pattern is Coalesced

Memory coalescing is crucial for performance. Let's verify our loads are coalesced:

**Loading A:** `A[row * K + (t * TILE_SIZE + threadIdx.x)]`
- `row` is the same for all threads in a warp (same `blockIdx.y` and `threadIdx.y`)
- `threadIdx.x` varies from 0-15 across threads
- Result: Adjacent threads access `A[row*K + 0]`, `A[row*K + 1]`, ... `A[row*K + 15]` → **Coalesced!**

**Loading B:** `B[(t * TILE_SIZE + threadIdx.y) * N + col]` where `col = blockIdx.x * TILE_SIZE + threadIdx.x`
- `bRow` is the same for threads with the same `threadIdx.y`
- `col` varies with `threadIdx.x`
- Result: Adjacent threads access `B[bRow*N + 0]`, `B[bRow*N + 1]`, ... → **Coalesced!**

> ⚠️ **Common mistake:** If you accidentally used `threadIdx.y` as the inner dimension for loading B (like `B[row * N + threadIdx.y]`), you'd get strided access with a stride of `N`—terrible for performance!

---

## The Complete Program

> ⚠️ **Note:** This example uses `new` for simplicity. For benchmarking, use `cudaMallocHost` (pinned memory) to avoid the driver's implicit copy from pageable to pinned memory. See the benchmark code in the repository for production-style `CUDA_CHECK` macros and pinned memory.

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE_SIZE 16

// Tiled matrix multiplication kernel
__global__ void matmulTiled(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C,
                            int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// CPU reference implementation
void matmulCPU(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    // Matrix dimensions
    int M = 1024;  // A is M×K
    int K = 1024;  // B is K×N
    int N = 1024;  // C is M×N
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory (pinned for consistent transfer timing)
    // Note: For a simple demo, you can use new float[], but benchmarks
    // should use pinned memory to avoid implicit pageable→pinned copies
    float *h_A, *h_B, *h_C, *h_C_ref;
    cudaMallocHost(&h_A, sizeA);
    cudaMallocHost(&h_B, sizeB);
    cudaMallocHost(&h_C, sizeC);
    h_C_ref = new float[M * N];  // CPU reference doesn't need pinned
    
    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // CPU reference
    matmulCPU(h_A, h_B, h_C_ref, M, N, K);
    
    // Verify
    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float error = fabs(h_C[i] - h_C_ref[i]);
        if (error > maxError) maxError = error;
    }
    
    printf("Matrix size: %d × %d × %d\n", M, K, N);
    printf("Max error: %e\n", maxError);
    printf("Result: %s\n", maxError < 1e-3 ? "PASS" : "FAIL");
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    delete[] h_C_ref;
    
    return 0;
}
```

---

## Benchmarking: Naive vs Tiled vs cuBLAS

Let's measure the real performance difference.

### Benchmark Setup

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA GTX 1650 Max-Q (14 SMs, CC 7.5) |
| **Memory** | 4.29 GB GDDR6, 160 GB/s bandwidth |
| **Timing** | CUDA Events, averaged over 10 runs after warmup |
| **Memory Type** | Pinned host memory (cudaMallocHost) |
| **Comparison** | Naive kernel, Tiled kernel (TILE=16), cuBLAS |

### Results (1024 × 1024 Matrices)

| Implementation | Time (ms) | GFLOPS | Speedup |
|----------------|-----------|--------|---------|
| **GPU Naive** | 15.90 ms | 135.09 | baseline |
| **GPU Tiled (TILE=16)** | 8.31 ms | 258.33 | 1.9× |
| **cuBLAS** | 0.82 ms | 2616.21 | 19.4× |

### Scaling Across Matrix Sizes

| Size | GPU Naive | GPU Tiled | cuBLAS | Tiled Speedup |
|------|-----------|-----------|--------|---------------|
| 256×256 | 0.25 ms (135 GFLOPS) | 0.16 ms (207 GFLOPS) | 0.06 ms (602 GFLOPS) | 1.5× |
| 512×512 | 2.01 ms (134 GFLOPS) | 1.23 ms (218 GFLOPS) | 0.27 ms (997 GFLOPS) | 1.6× |
| 1024×1024 | 15.90 ms (135 GFLOPS) | 8.31 ms (258 GFLOPS) | 0.82 ms (2616 GFLOPS) | 1.9× |
| 2048×2048 | 75.71 ms (227 GFLOPS) | 49.81 ms (345 GFLOPS) | 6.43 ms (2670 GFLOPS) | 1.5× |

> 💡 **GFLOPS calculation:** For M=N=K=1024, total FLOPs = 2 × 1024³ ≈ 2.15 billion. Divide by time in seconds.

### Key Observations

1. **Tiled is ~2× faster than naive** — shared memory reduces global memory traffic
2. **cuBLAS is 10× faster than our tiled** — it uses advanced optimizations (register blocking, vectorized loads, double buffering, warp-level primitives)
3. **cuBLAS achieves 2.6 TFLOPS** — that's impressive for a laptop GPU with ~4.8 TFLOPS theoretical peak!

---

## Understanding the Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU MEMORY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐   Fastest    ┌──────────────────────────┐ │
│  │     REGISTERS       │◄────────────►│ ~20 TB/s                 │ │
│  │  (per thread)       │              │ Private to each thread   │ │
│  └─────────────────────┘              └──────────────────────────┘ │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐              ┌──────────────────────────┐ │
│  │   SHARED MEMORY     │◄────────────►│ ~10 TB/s                 │ │
│  │  (per block)        │              │ Shared by all threads    │ │
│  │  __shared__ float   │              │ in a block (48-164 KB)   │ │
│  └─────────────────────┘              └──────────────────────────┘ │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐              ┌──────────────────────────┐ │
│  │    L2 CACHE         │◄────────────►│ ~2-4 TB/s                │ │
│  │  (shared by all)    │              │ Automatic caching        │ │
│  └─────────────────────┘              └──────────────────────────┘ │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐   Slowest    ┌──────────────────────────┐ │
│  │   GLOBAL MEMORY     │◄────────────►│ ~500 GB/s (HBM2)         │ │
│  │  (device VRAM)      │              │ Visible to all threads   │ │
│  │  cudaMalloc'd data  │              │ 4-80 GB                  │ │
│  └─────────────────────┘              └──────────────────────────┘ │
│                                                                     │
│  ───────────────── PCIe Bus (~32 GB/s) ─────────────────            │
│                                                                     │
│  ┌─────────────────────┐                                            │
│  │    HOST MEMORY      │  System RAM (CPU side)                     │
│  │  (cudaMemcpy src)   │                                            │
│  └─────────────────────┘                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why tiling works:** We move data from slow global memory (500 GB/s) to fast shared memory (10+ TB/s), then reuse it many times.

---

## Common Pitfalls

### 1. Forgetting `__syncthreads()`

```cpp
// WRONG: Race condition!
tileA[threadIdx.y][threadIdx.x] = A[...];
tileB[threadIdx.y][threadIdx.x] = B[...];
// Some threads start computing before others finish loading
for (int k = 0; k < TILE_SIZE; k++) ...
```

**Fix:** Always synchronize after loading shared memory and before the next load.

### 2. Bank Conflicts in Shared Memory

Shared memory is divided into 32 banks. If multiple threads access the same bank (but different addresses), accesses are serialized.

```cpp
// Potential bank conflict with column-major access
tileA[k][threadIdx.y]  // Threads in a warp hit same bank
```

**Fix:** Pad shared memory or transpose access pattern.

**Why our kernel has NO bank conflicts:**

```cpp
// In the compute loop:
sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
```

- `tileA[threadIdx.y][k]`: All threads in a warp have the same `threadIdx.y` (within a row of the warp). They all read the **same address** → This is a **broadcast**, not a conflict!
- `tileB[k][threadIdx.x]`: Threads have different `threadIdx.x` (0-15). They access different columns → Each thread hits a **different bank** (assuming TILE_SIZE=16 divides evenly into 32 banks).

> 💡 **Bank conflict rule:** Same bank, different address = conflict. Same address = broadcast (free). Different banks = parallel access (free).

### 3. Non-Coalesced Global Memory Access

```cpp
// BAD: Strided access (threads read non-adjacent addresses)
A[threadIdx.x * K + col]  // Stride of K between threads

// GOOD: Coalesced access (adjacent threads read adjacent addresses)  
A[row * K + threadIdx.x]  // Stride of 1 between threads
```

---

## What's Next: Advanced Optimizations

Our tiled kernel is good, but cuBLAS is still 3× faster. Here's what the pros do:

1. **Register tiling**: Each thread computes multiple outputs, reducing shared memory traffic
2. **Vectorized loads**: Use `float4` to load 4 floats at once
3. **Double buffering**: Overlap loading the next tile with computing the current tile
4. **Warp-level primitives**: `wmma` for Tensor Cores (Volta and newer)

These optimizations are beyond beginner level, but understanding them helps appreciate why libraries like cuBLAS and cuDNN exist.

---

## Conclusion

You've now written your second CUDA kernel—and one of the most important algorithms in computing:

1. ✅ **2D grids and blocks** for matrix problems
2. ✅ **Naive kernel**: simple but memory-bound
3. ✅ **Tiled kernel**: uses shared memory for 5-6× speedup
4. ✅ **Memory hierarchy**: registers → shared → L2 → global
5. ✅ **Why libraries win**: cuBLAS uses advanced optimizations we didn't cover

### The Bigger Picture

Matrix multiplication is the foundation of:
- **Deep learning**: Every layer is a matmul (or batched matmul)
- **Computer graphics**: Transformations, projections
- **Scientific computing**: Simulations, linear solvers
- **Recommendation systems**: Embedding lookups

Understanding GPU matmul helps you understand why AI runs on GPUs—and why NVIDIA is worth a trillion dollars.

---

## 🧪 Challenge: Experiment with TILE_SIZE

Before moving on, try this experiment:

1. **Change `TILE_SIZE` from 16 to 32**
2. **Recompile and run the benchmark**
3. **Observe what happens to performance**

```cpp
#define TILE_SIZE 32  // Try this!
```

**Questions to consider:**
- Does performance go up or down?
- Why might larger tiles hurt performance?

<details>
<summary>💡 Click for hints</summary>

**Shared memory usage:** `TILE_SIZE=32` means `32×32×2×4 = 8 KB` of shared memory per block (vs 2 KB for TILE_SIZE=16).

**Threads per block:** `32×32 = 1024` threads, the maximum for most GPUs.

**Occupancy:** Larger shared memory usage means fewer blocks can run concurrently on each SM. This reduces **occupancy** (the ratio of active warps to maximum warps).

**Register pressure:** More threads = more register demand. If you exceed the register file, you get **register spilling** to slow local memory.

**The tradeoff:** Larger tiles reduce global memory traffic but hurt occupancy. The optimal TILE_SIZE depends on your GPU's resources. For GTX 1650, TILE_SIZE=16 is often the sweet spot.

</details>

---

## Further Reading

- [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUTLASS: CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)

---

*If you found this helpful, give it a clap 👏 and follow for more CUDA tutorials.*

---

*Code from this post is available at: [github.com/ajkumar-13/Learning-Cuda](https://github.com/ajkumar-13/Learning-Cuda)*
