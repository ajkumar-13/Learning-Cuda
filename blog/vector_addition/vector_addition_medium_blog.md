# Stop Looping. Start Launching: Vector Addition on the GPU

*A beginner-friendly guide to writing your first CUDA kernel*

---

## Introduction

Here's a simple problem: you have two arrays of numbers, **A** and **B**, each containing **10 million** floating-point values. You want to add them element-by-element to produce a third array **C**:

```text
C[i] = A[i] + B[i]
```

On a CPU, you'd write a `for` loop. On a GPU, you describe the work **per element** and let the GPU spawn millions of parallel threads to do it.

In this post, we'll write your first CUDA kernel (vector addition) and benchmark it properly, because bad benchmarking is the fastest way to fool yourself. You'll finish with a working kernel and a realistic understanding of when the GPU wins and when it doesn't.

> **Prerequisites:** This post assumes you've read [Introduction to CUDA](../introduction/00_introduction_to_cuda.md), where we covered GPU architecture, the thread hierarchy (grids, blocks, threads), warps, and the host-device memory model. If terms like `blockIdx`, `threadIdx`, or "warp divergence" are unfamiliar, start there first.

---

## What We're Building

**Goal:** Implement vector addition on the GPU and benchmark it against CPU implementations (single-threaded and OpenMP).

**What you'll learn:**
- Writing a `__global__` kernel function
- Managing GPU memory (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
- Calculating launch configuration (blocks and threads)
- Proper GPU benchmarking methodology
- When the GPU actually wins (and when it doesn't)

---

## Step 1: The CUDA Kernel

The `__global__` keyword marks a function that runs on the GPU but is called from the CPU:

```cpp
__global__ void vectorAddGPU(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

**Line by line:**
- `int i = blockIdx.x * blockDim.x + threadIdx.x;` - Each thread calculates its unique global index
- `if (i < N)` - Boundary check (we often launch more threads than elements)
- `C[i] = A[i] + B[i];` - The actual work: one addition per thread

![Global Index Calculation](images/Global%20Index%20Calculation.png)

**Why the boundary check?** We launch threads in blocks of fixed size (e.g., 256). If N isn't divisible by 256, we'll have extra threads. The `if` prevents out-of-bounds memory access.

---

## Step 2: Launch Configuration

A common starting point is 256 threads per block:

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // ceiling division

vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

**Why `(N + threadsPerBlock - 1) / threadsPerBlock`?**

This is integer ceiling division. For N = 1000 and threadsPerBlock = 256:
- `(1000 + 255) / 256 = 1255 / 256 = 4` blocks
- 4 blocks × 256 threads = 1024 threads (enough to cover 1000 elements)

![Grid-Block-Thread Hierarchy](images/Grid-Block-Thread%20Hierarchy.png)

> **Note:** You can launch millions of threads even though the GPU has "thousands of cores." The GPU schedules threads in batches called **warps** (typically 32 threads). You're describing parallel work, not manually creating OS threads. The hardware scheduler handles the details.

---

## Step 3: Memory Management

CPU and GPU have **separate memory spaces**. You must explicitly allocate GPU memory and copy data between host and device.

![Host-Device Memory Diagram](images/Host-Device%20Memory%20Diagram.png)

**The workflow:**

```cpp
// 1. Allocate device memory
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, size);
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);

// 2. Copy inputs: Host → Device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// 3. Launch kernel
vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

// 4. Synchronize (wait for kernel to finish)
cudaDeviceSynchronize();

// 5. Copy result: Device → Host
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

// 7. Free device memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

> **Important:** Kernel launches are **asynchronous**. The CPU continues immediately after `<<<...>>>`. Use `cudaDeviceSynchronize()` before timing or reading results.

---

## The Complete Program

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void vectorAddGPU(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 10000000;  // 10 million elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            printf("Mismatch at i=%d: %f != %f\n", i, h_C[i], expected);
            correct = false;
        }
    }
    printf("Result: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
```

**Compile and run:**
```bash
nvcc -O2 vector_add.cu -o vector_add.exe
vector_add.exe
```

---

## Benchmarking: The Right Way

Most "GPU vs CPU" benchmarks are **wrong** for at least one of these reasons:

- Timing only one run (noise/variance)
- Measuring GPU kernels with CPU timers (misses async launches)
- Forgetting kernel launches are asynchronous
- Letting the compiler optimize away CPU work
- Using pageable memory for transfers (slower than pinned)

### Our Methodology

| Aspect | Approach |
|--------|----------|
| **Repetitions** | Multiple runs, averaged (warm-up discarded) |
| **GPU Timing** | CUDA Events (`cudaEventElapsedTime`) |
| **CPU Timing** | `std::chrono::steady_clock` |
| **Memory** | Pinned memory (`cudaMallocHost`) for fair transfer comparison* |
| **Verification** | Checksums prevent dead-code elimination |
| **Correctness** | Results validated against CPU reference |

*Note: The "Complete Program" above uses normal host memory (`new[]`) for simplicity. The benchmark code uses pinned memory for accurate transfer measurements.

### Two Numbers You Should Always Report

1. **GPU kernel-only** - Data already on device. This is the "best case."
2. **GPU with transfer** - Host→Device + Kernel + Device→Host. This is the realistic case for one-off operations.

![Kernel-only vs End-to-end](images/CPU%20(Sequential)%20vs%20GPU%20(Parallel)%20-%20Kernel%20Only.png)

![Realistic GPU Timeline](images/Realistic%20GPU%20Timeline%20(with%20Memory%20Transfer).png)

---

## Benchmark Results

### Test Machine

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA GeForce GTX 1650 Max-Q (14 SMs, Compute Capability 7.5) |
| **CPU** | Intel Core i7-9750H (6 cores / 12 threads, 2.6 GHz base) |
| **RAM** | 16 GB DDR4 |
| **OS** | Windows 11 |
| **CUDA** | 12.6, Driver 560.94 |
| **Compiler** | `nvcc -O2 -Xcompiler "/openmp"` |
| **OpenMP** | 12 threads, `schedule(static)` |
| **PCIe** | Gen 3 x16 |

### Results Table

| N | CPU (1 thread) | CPU (OpenMP 12T) | GPU (with transfer) | GPU (kernel only) |
|---|----------------|------------------|---------------------|-------------------|
| **1K** | **<1 µs** | 2 µs | 149 µs | 3 µs |
| **10K** | 3 µs | **3 µs** | 145 µs | 3 µs |
| **100K** | 33 µs | **9 µs** | 139 µs | 11 µs |
| **1M** | 1.02 ms | **0.39 ms** | 1.20 ms | 0.08 ms |
| **10M** | 13.6 ms | 12.3 ms | 11.5 ms | **0.82 ms** |
| **100M** | 146 ms | 128 ms | 123 ms | **8.28 ms** |

![Benchmark Chart](images/benchmark_chart.png)

---

## Reading the Results

### Small N (< 100K): CPU Wins

GPU overhead (kernel launch + PCIe transfer) dominates. The kernel itself takes microseconds, but you're paying ~140 µs just to start.

**Lesson:** Don't GPU-accelerate tiny workloads.

### Medium N (~1M): It's Close

OpenMP is competitive. Data fits in CPU cache, CPU boosts aggressively. GPU with transfer is slightly slower.

### Large N (10M+): GPU Wins

At 100M elements:
- **Kernel-only:** GPU is **17× faster** than single-threaded CPU
- **With transfer:** GPU still wins by ~20%

As N grows, the fixed overhead matters less, and **memory bandwidth** becomes the bottleneck.

### The Key Insight

Vector addition is **memory-bandwidth bound**, not compute bound.

Each element requires:
- 2 reads (A[i], B[i]) = 8 bytes
- 1 write (C[i]) = 4 bytes
- 1 addition = ~1 cycle

The ratio of memory to compute is huge. The GPU wins on large N because it has higher memory bandwidth (~192 GB/s for GTX 1650 vs ~40 GB/s for DDR4).

---

## When to Use GPU

![When to Use What - Decision Flowchart](images/When%20to%20Use%20What%20(Decision%20Flowchart).png)

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| N < 100K, one-off | CPU | GPU overhead dominates |
| N < 1M, one-off | CPU (OpenMP) | Competitive, simpler |
| N > 10M, one-off | GPU | Bandwidth wins |
| Multi-kernel pipeline | GPU | Pay transfer once, run many kernels |
| Data already on GPU | GPU | Always (no transfer cost) |

> **The real GPU advantage:** In deep learning and simulations, data stays on the GPU across hundreds of operations. You pay the transfer cost once, then run at GPU speed.

---

## Summary

You now know how to:

1. Write a CUDA kernel with proper boundary checking
2. Manage GPU memory (allocate, copy, free)
3. Calculate launch configuration (blocks × threads)
4. Add error checking (`cudaGetLastError`, `CUDA_CHECK` macro)
5. Benchmark correctly (CUDA events, pinned memory, multiple runs)
6. Interpret results (kernel-only vs end-to-end)

**The takeaway:** A GPU doesn't win because it's "faster at math." It wins because it can push **massive parallelism** and **high memory bandwidth**, but only when the problem is large enough to amortize the overhead.

---

## What's Next?

Ready to go deeper? In the next post, we'll tackle **matrix operations**:

- 2D thread indexing
- Shared memory optimization
- Memory coalescing patterns
- Tiled algorithms

Until then, try modifying the vector addition kernel:
- Implement element-wise multiplication
- Implement element-wise subtraction
- Try different block sizes (128, 512, 1024) and measure the impact

Happy coding, and welcome to the parallel world!

---

## Full Source Code

The complete benchmark code with all optimizations is available in the repository:
- [`src/benchmark_vector_add.cu`](../../src/benchmark_vector_add.cu)
