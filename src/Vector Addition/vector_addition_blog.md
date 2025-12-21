# Stop Looping, Start Launching: Vector Addition on the GPU

*A beginner-friendly guide to writing your first CUDA kernel*

---

## Introduction

Here's a simple problem: you have two arrays of numbers, **A** and **B**, each containing 10 million floating-point values. You want to add them together element-by-element to produce a third array **C**.

```
C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
...
C[9,999,999] = A[9,999,999] + B[9,999,999]
```

On a CPU, you'd write a `for` loop and watch your processor churn through 10 million additions **one after the other**. It works, but it's slow.

*"But wait,"* you might say, *"can't I use multithreading on the CPU?"*

Yes! Modern CPUs have multiple cores (typically 4–16), and you could use OpenMP, `std::thread`, or multiprocessing to split the work. With 8 cores, you'd get roughly an 8× speedup. Not bad.

But here's the thing: a modern GPU has **thousands** of cores. An NVIDIA RTX 3080 has 8,704 CUDA cores. An RTX 4090 has 16,384. While each GPU core is simpler and slower than a CPU core, the sheer *quantity* means the GPU can process thousands of independent operations simultaneously—and chew through millions of total operations in a fraction of the time.

> **Wait, if there are only 16K cores, how do we run 10 million threads?**  
> Great question! The GPU doesn't run all 10 million threads at the *exact* same instant. Instead, it **schedules** them in batches called *warps* (32 threads each). While one warp waits for memory, another runs—keeping all cores busy. Think of it like a restaurant with 100 tables but 1,000 reservations: they cycle customers through quickly, so everyone gets served fast.

| Approach | Parallel Workers | Best For |
|----------|------------------|----------|
| CPU (single-threaded) | 1 | Simple tasks, small data |
| CPU (multithreaded) | 4–16 cores | Moderate parallelism |
| GPU | 1,000–16,000+ cores | Massive parallelism |

For 10 million additions where each operation is independent, the GPU wins by a landslide.

Think of it this way: a CPU with 8 cores is like 8 expert chefs sharing a kitchen. A GPU is like 10,000 fast-food workers, each with their own station, all making one sandwich simultaneously.

By the end of this post, you'll write a program that harnesses this power—your very first CUDA kernel.

---

## The Concept: CPU vs. GPU

### The Analogy

**CPU**: Imagine a math genius sitting down to complete a 100-question exam. They're fast, but they solve question 1, then question 2, then question 3... sequentially.

**GPU**: Now imagine 100 elementary school students, each assigned just *one* question. Individually, they're slower than the genius—but they all write their answers at the same time. The exam is finished almost instantly.

### The Timeline

```
CPU (Sequential):
┌───────────────────────────────────────────────────────────────────┐
│ A[0]+B[0] │ A[1]+B[1] │ A[2]+B[2] │ A[3]+B[3] │ ...               │
└───────────────────────────────────────────────────────────────────┘
            Time ──────────────────────────────────────────────────►

GPU (Parallel):
┌───────────┐
│ A[0]+B[0] │
│ A[1]+B[1] │
│ A[2]+B[2] │  ← All happen at the SAME time
│ A[3]+B[3] │
│    ...    │
└───────────┘
```

![CPU vs GPU Timeline - Kernel Only](images/CPU%20(Sequential)%20vs%20GPU%20(Parallel)%20-%20Kernel%20Only.png)

> **Note:** This shows the *kernel-only* time where data is already on GPU.
> For a fair comparison including memory transfer, see below:

![Realistic GPU Timeline with Memory Transfer](images/Realistic%20GPU%20Timeline%20(with%20Memory%20Transfer).png)

> **Key insight:** When including memory transfer, GPU takes ~30ms vs CPU's ~19ms for a one-shot operation. But when data *stays* on GPU (like in deep learning), only the kernel time matters—and that's 13× faster!

The GPU doesn't make each operation faster—it makes *all of them happen simultaneously*.

---

## Understanding the GPU Architecture: Grids, Blocks, and Threads

This is the part that trips up most beginners. You can't just tell the GPU "run everything in parallel." You have to **organize** your parallel workers into a hierarchy.

### The Hierarchy

Think of it like a school:

| CUDA Term | Analogy | Description |
|-----------|---------|-------------|
| **Grid** | The whole school | The entire kernel launch |
| **Block** | A classroom | A group of threads that can cooperate |
| **Thread** | A student | The individual worker doing one task |

When you launch a kernel, you're essentially saying:  
*"Hey GPU, spin up a **grid** of **blocks**, each containing a bunch of **threads**."*

![Grid-Block-Thread Hierarchy](images/Grid-Block-Thread%20Hierarchy.png)

### The Coordinate System

Here's the key question: if you have 10 million threads running simultaneously, how does each thread know *which element* to work on?

Each thread gets a unique ID through built-in variables:

- `threadIdx.x` — The thread's position *within* its block (0, 1, 2, ...)
- `blockIdx.x` — Which block this thread belongs to (0, 1, 2, ...)
- `blockDim.x` — How many threads are in each block (e.g., 256)

### The Magic Formula

To calculate the global index `i` (which element this thread should process):

```
i = (blockIdx.x × blockDim.x) + threadIdx.x
```

**Visual breakdown:**

```
Block 0 (256 threads):  indices 0–255
  └── threadIdx.x = 0   →  i = (0 × 256) + 0   = 0
  └── threadIdx.x = 1   →  i = (0 × 256) + 1   = 1
  └── threadIdx.x = 255 →  i = (0 × 256) + 255 = 255

Block 1 (256 threads):  indices 256–511
  └── threadIdx.x = 0   →  i = (1 × 256) + 0   = 256
  └── threadIdx.x = 1   →  i = (1 × 256) + 1   = 257
  └── threadIdx.x = 255 →  i = (1 × 256) + 255 = 511

Block 2 (256 threads):  indices 512–767
  └── threadIdx.x = 0   →  i = (2 × 256) + 0   = 512
  ...
```

Each thread calculates its own `i`, then works on `A[i] + B[i]`. No loops needed!

![Global Index Calculation](images/Global%20Index%20Calculation.png)

---

## The Code: Implementing Vector Addition

Now let's write actual CUDA code. There are three parts:

1. **Memory management** — moving data between CPU and GPU
2. **The kernel** — the function that runs on the GPU
3. **Launching the kernel** — telling the GPU to go

### Step 1: Memory Management (Host vs. Device)

**Key concept:** The CPU (called the *Host*) and GPU (called the *Device*) have **separate memories**. You can't just pass a pointer from your C++ code to the GPU—you have to explicitly copy data back and forth.

```cpp
// Allocate memory on the GPU
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, size);  // d_ prefix = "device" (GPU)
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);

// Copy data from CPU → GPU
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// ... run kernel ...

// Copy result from GPU → CPU
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```

**Diagram: The Memory Gap**

```
┌─────────────────┐                      ┌─────────────────┐
│      HOST       │   cudaMemcpy         │     DEVICE      │
│      (CPU)      │ ◄──────────────────► │      (GPU)      │
│                 │   (PCI-Express Bus)  │                 │
│  h_A, h_B, h_C  │                      │  d_A, d_B, d_C  │
└─────────────────┘                      └─────────────────┘
```

![Host-Device Memory Diagram](images/Host-Device%20Memory%20Diagram.png)

### Step 2: The Kernel (`__global__`)

The `__global__` keyword tells the compiler: *"This function runs on the GPU but is called from the CPU."*

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    // Calculate global thread ID
    // blockIdx.x * blockDim.x  → skips all previous blocks
    // + threadIdx.x            → adds offset within current block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (safety first!)
    // We might launch more threads than elements
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}
```

**Why the `if (i < N)` check?**

Suppose you have 1000 elements and 256 threads per block. You'd need:  
`ceil(1000 / 256) = 4 blocks = 1024 threads`

But 1024 > 1000! The last 24 threads have nothing to do. Without the boundary check, they'd access memory beyond the array and crash your program.

![Boundary Check Visualization](images/Boundary%20Check%20Visualization.png)

### Step 3: Launching the Kernel (`<<<...>>>`)

Those weird triple angle brackets are CUDA's way of specifying the grid dimensions.

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // ceiling division

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

**Translation:**  
*"Launch `blocksPerGrid` blocks, each with `threadsPerBlock` threads, and have them all execute the `vectorAdd` function."*

---

## The Complete Program

Here's everything together:

```cpp
#include <cuda_runtime.h>
#include <cstdio>

// The GPU kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
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

    // Initialize input vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify (spot check)
    printf("C[0] = %f (expected %f)\n", h_C[0], h_A[0] + h_B[0]);
    printf("C[N-1] = %f (expected %f)\n", h_C[N-1], h_A[N-1] + h_B[N-1]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    printf("Done!\n");
    return 0;
}
```

---

## Performance: When Does the GPU Win?

Let's be honest: for *small* arrays, the GPU might actually be **slower** than the CPU.

Why? Because of the **memory transfer overhead**. Copying data from CPU to GPU (and back) takes time. For a handful of elements, this overhead dominates.

I ran benchmarks on my machine (GTX 1650, 12-thread CPU) comparing:
- **CPU (1 thread)** — simple `for` loop
- **CPU (OpenMP)** — multithreaded with 12 threads
- **GPU (with transfer)** — includes `cudaMemcpy` time
- **GPU (kernel only)** — just the computation, data already on GPU

| N (elements) | CPU (1 thread) | CPU (12 threads) | GPU (w/ transfer) | GPU (kernel only) |
|--------------|----------------|------------------|-------------------|-------------------|
| 1,000 | 0.001 ms ✅ | 1.46 ms ❌ | 0.28 ms | 0.09 ms |
| 10,000 | 0.02 ms | 0.07 ms | 0.23 ms | 0.06 ms |
| 100,000 | 0.40 ms | 0.02 ms ✅ | 0.50 ms | 0.06 ms |
| 1,000,000 | 2.07 ms | 0.95 ms | 2.83 ms | 0.13 ms |
| 10,000,000 | 19.1 ms | 11.2 ms | 29.8 ms ❌ | **0.85 ms** ✅ |

![Benchmark Chart](images/benchmark_chart.png)

### Key Insights

1. **For tiny arrays (N < 10,000)**: Single-threaded CPU wins. The overhead of spinning up OpenMP threads or transferring to GPU isn't worth it.

2. **For medium arrays (N ~ 100,000)**: CPU multithreading shines. OpenMP with 12 threads beats everything here.

3. **For large arrays (N ≥ 1,000,000)**: Look at the "GPU kernel only" column—**0.85 ms** for 10 million elements vs. 11.2 ms for 12-thread CPU. The GPU is **13× faster** at raw computation.

4. **The transfer bottleneck**: Notice that "GPU with transfer" is actually *slower* than single-threaded CPU for 10M elements (29.8 ms vs 19.1 ms). The PCI-Express bus is the bottleneck. This is why real GPU applications keep data on the device across multiple kernel calls.

**Bottom line:** If you're doing a one-off vector addition, CPU multithreading might be enough. But if your data *stays* on the GPU across many operations (like in deep learning), the GPU crushes it.

![When to Use What - Decision Flowchart](images/When%20to%20Use%20What%20(Decision%20Flowchart).png)

---

## Conclusion

You just wrote your first CUDA kernel! Let's recap what you learned:

1. **The mental model**: GPUs execute thousands of threads *simultaneously*, each doing a tiny piece of work.

2. **The hierarchy**: Threads are organized into blocks, and blocks form a grid.

3. **The index formula**: `i = blockIdx.x * blockDim.x + threadIdx.x` gives each thread a unique ID.

4. **Memory management**: CPU and GPU have separate memory spaces—use `cudaMalloc` and `cudaMemcpy` to move data.

5. **The tradeoff**: GPUs shine when `N` is large; small problems suffer from transfer overhead.


Happy coding, and welcome to the parallel world! 🚀

---

*Code from this post is available at: [github.com/ajkumar-13/Learning-Cuda/tree/main/src/Vector%20Addition](https://github.com/ajkumar-13/Learning-Cuda/tree/main/src/Vector%20Addition)*
