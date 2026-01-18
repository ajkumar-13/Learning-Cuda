# Unlocking Supercomputing: An Introduction to CUDA

> **How NVIDIA turned graphics cards into the engines of modern AI**

---

## The GPU Revolution

The Graphics Processing Unit (GPU) was born as a specialized tool for 3D rendering, calculating millions of pixel colors to display video games and graphics. But around 2003, researchers began noticing something remarkable: the massively parallel architecture designed for graphics could solve *other* computational problems too.

In **2006**, NVIDIA introduced **CUDA (Compute Unified Device Architecture)**, revolutionizing high-performance computing by allowing developers to use GPUs for general-purpose processing. Today, CUDA powers everything from scientific simulations to the Large Language Models (LLMs) driving modern AI.

If you're looking to understand how to harness this power, this guide covers the fundamentals of CUDA architecture and programming.

---

## 1. The Core Difference: CPU vs. GPU

To understand CUDA, you must first understand how a GPU differs from a CPU.

### The CPU: Optimized for Latency

The CPU is designed for **latency**, it excels at executing a serial sequence of operations (a single thread) as fast as possible. Modern CPUs devote most of their transistors to:

- **Large caches** (L1, L2, L3) to minimize memory access time
- **Complex flow control** (branch prediction, speculative execution)
- **Out-of-order execution** to maximize single-thread performance

A typical CPU might have 8-16 powerful cores, each optimized to complete individual tasks quickly.

### The GPU: Optimized for Throughput

The GPU is designed for **throughput**, it excels at executing thousands of threads in parallel. It trades off single-thread performance to achieve massive total work done per second by devoting more transistors to:

- **Arithmetic Logic Units (ALUs)** for computation
- **Simple control logic** (no branch prediction, in-order execution)
- **Hardware thread management** to hide memory latency

A modern GPU might have thousands of simpler cores, each weaker than a CPU core, but together capable of vastly more parallel computation.

![GPU CPU Architecture Comparision](\images\cpu_vs_gpu_architecture.png)

### When to Use Each

| Workload Type | Best Processor | Why |
|---------------|----------------|-----|
| Serial algorithms | CPU | Single-thread performance |
| Complex branching | CPU | Sophisticated branch prediction |
| Highly parallel (data-parallel) | GPU | Thousands of threads |
| Matrix operations | GPU | Regular, parallelizable patterns |
| Deep learning | GPU | Massive parallelism in matrix math |

---

## 2. The Programming Model: Heterogeneous Computing

CUDA uses a **heterogeneous model**. This means a system consists of two parts:

| Component | Name | Description |
|-----------|------|-------------|
| **Host** | CPU | The CPU and its memory (System RAM) |
| **Device** | GPU | The GPU and its memory (Video RAM / VRAM) |

A typical CUDA application:

1. **Starts on the CPU** (host)
2. **Allocates memory** on both host and device
3. **Copies data** from host to device
4. **Launches a kernel** (GPU function) to process the data
5. **Copies results** back from device to host
6. **Frees memory** on both sides

![Heterogeneous Computing Model](\images\heterogeneous_computing.png)

### What is a Kernel?

A **kernel** is a function executed on the GPU. When you launch a kernel, you aren't just running a function once—you are launching **millions of threads** that all execute that same function code in parallel.

```cuda
// Kernel definition - runs on the GPU
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];  // Each thread adds ONE element
    }
}

int main() {
    // ... allocate memory, copy data ...
    
    // Launch kernel with N threads
    // Each thread executes VecAdd() simultaneously!
    VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // ... copy results back ...
}
```

The `<<<numBlocks, threadsPerBlock>>>` syntax is the **execution configuration**—it tells CUDA how many threads to launch.

---

## 3. Organizing the Chaos: The Thread Hierarchy

Managing millions of threads requires strict organization. CUDA groups threads into a **three-level hierarchy**:

### The Hierarchy

| Level | Description | Size Limits |
|-------|-------------|-------------|
| **Thread** | The smallest unit of execution | — |
| **Thread Block** | A group of threads that can cooperate and share memory | Up to 1024 threads |
| **Grid** | A collection of thread blocks that execute a kernel | Up to billions of threads |

![Thread Hierarchy](\images\thread_hierarchy.png)

### Built-in Variables

Every thread can identify its unique coordinates within this grid using **built-in variables**:

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | `uint3` | Thread index within the block (x, y, z) |
| `blockIdx` | `uint3` | Block index within the grid (x, y, z) |
| `blockDim` | `dim3` | Dimensions of each block |
| `gridDim` | `dim3` | Dimensions of the grid |

### Calculating Global Thread Index

```cuda
// For a 1D grid of 1D blocks:
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

// For a 2D grid of 2D blocks:
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

This allows each thread to calculate **exactly which piece of data it should process**.

### Why Thread Blocks?

Thread blocks provide important guarantees:

1. **Cooperation**: Threads within a block can share data via **shared memory**
2. **Synchronization**: Threads within a block can synchronize using `__syncthreads()`
3. **Scheduling Unit**: Blocks are the unit of work assigned to hardware processors

> **Key Insight**: Thread blocks must execute **independently**. This allows the GPU to schedule blocks in any order, on any available processor, enabling automatic scalability across different GPU sizes.

---

## 4. Hardware Mapping: SMs and Warps

How does the hardware actually execute this software hierarchy?

### Streaming Multiprocessors (SMs)

The GPU is composed of many **Streaming Multiprocessors (SMs)**. When you launch a grid:

1. The hardware assigns **Thread Blocks** to available SMs
2. Multiple blocks can run on the same SM (if resources permit)
3. Blocks execute independently, in any order



![ GPU Hardware Architecture](\images\streaming_multiprocessor.png)

### Warps: The SIMT Execution Model

Inside an SM, threads are executed in groups of **32** called **warps**.

| Term | Definition |
|------|------------|
| **Warp** | A group of 32 threads that execute together |
| **SIMT** | Single-Instruction, Multiple-Threads |
| **Lane** | A single thread's position within a warp (0-31) |

**SIMT (Single-Instruction, Multiple-Threads)**: All 32 threads in a warp execute the **exact same instruction at the same time**, but on different data.


![Warp Execution](\images\warp_execution.png)

### Warp Divergence: The Performance Killer

What happens if threads in a warp take different code paths?

```cuda
if (threadIdx.x % 2 == 0) {
    // Even threads do this
    doSomething();
} else {
    // Odd threads do this
    doSomethingElse();
}
```

Since all threads in a warp must execute the same instruction, the hardware **serializes** both paths:

1. First, even threads execute while odd threads wait
2. Then, odd threads execute while even threads wait

This is called **warp divergence**, and it can significantly reduce performance.

![Warp Divergence](\images\warp_divergence.png)

> **Best Practice**: Structure your code so threads within a warp follow the same execution path whenever possible.

---

## 5. The Memory Hierarchy

Just like CPUs, GPUs have different levels of memory with different speeds and scopes:

![Memory Hierarchy](\images\memory_hierarchy.png)


### Memory Types

| Memory Type | Scope | Lifetime | Speed | Size |
|-------------|-------|----------|-------|------|
| **Registers** | Thread | Thread | ~20 TB/s | ~255 per thread |
| **Local Memory** | Thread | Thread | Slow (DRAM) | Up to 512 KB |
| **Shared Memory** | Block | Block | ~10 TB/s | Up to 164 KB/SM |
| **Global Memory** | Grid + Host | Application | ~500 GB/s | GBs |
| **Constant Memory** | Grid | Application | Cached | 64 KB |
| **Texture Memory** | Grid | Application | Cached | GBs |

### Using Shared Memory

Shared memory is the key to high-performance CUDA programming. It's **user-managed** and acts like a programmable cache.

```cuda
__global__ void sharedMemExample(float* input, float* output) {
    // Declare shared memory
    __shared__ float tile[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global memory to shared memory
    tile[tid] = input[gid];
    
    // Synchronize to ensure all threads have loaded
    __syncthreads();
    
    // Now all threads can access any element in tile[]
    // Fast access! No global memory trips needed.
    float sum = tile[tid] + tile[(tid + 1) % 256];
    
    output[gid] = sum;
}
```

### L2 Cache

Modern GPUs (Volta+) also have a large **L2 cache** shared across all SMs that automatically caches global memory accesses, improving performance for repeated accesses.

---

## 6. From Code to Execution: PTX and Cubins

When you write CUDA C++, the compiler (NVCC) breaks it down into two stages:

### The Compilation Pipeline

![Compilation pipeline](\images\compilation_pipeline.png)

### PTX: Portable Intermediate Representation

**PTX (Parallel Thread Execution)** is a virtual assembly language that provides a stable abstraction of the GPU hardware.

```ptx
// Example PTX code
.visible .entry VecAdd(
    .param .u64 VecAdd_param_0,
    .param .u64 VecAdd_param_1,
    .param .u64 VecAdd_param_2
)
{
    .reg .f32   %f<3>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<7>;
    
    ld.param.u64    %rd1, [VecAdd_param_0];
    ld.param.u64    %rd2, [VecAdd_param_1];
    // ... more PTX instructions
}
```

**Why PTX matters:**
- **Forward Compatibility**: PTX can be compiled to run on future GPUs that didn't exist when the code was written
- **JIT Compilation**: The driver can optimize PTX for the specific GPU at runtime

### Cubin: Native Binary Code

**Cubin** is the actual binary code for a specific GPU architecture (e.g., `sm_75` for Turing, `sm_86` for Ampere).

**Compilation flags:**

```bash
# Generate PTX (portable)
nvcc -arch=compute_75 -code=sm_75 program.cu

# Generate Cubin (native, faster to load)
nvcc -arch=sm_75 program.cu

# Generate Fatbin (both PTX and Cubin)
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_75,code=compute_75 \
     program.cu
```

---

## 7. Your First CUDA Program: Vector Addition

Let's put it all together with a complete example:

```cuda
// vecadd.cu - Complete Vector Addition Example
#include <stdio.h>

// GPU Kernel: Each thread adds one element
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;  // 1 million elements
    size_t size = N * sizeof(float);
    
    // ====== Step 1: Allocate host memory ======
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // ====== Step 2: Allocate device memory ======
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // ====== Step 3: Copy data to device ======
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // ====== Step 4: Launch kernel ======
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // ====== Step 5: Copy results back to host ======
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    
    // ====== Step 6: Free memory ======
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

### Compilation and Execution

```bash
# Compile
nvcc -arch=sm_75 vecadd.cu -o vecadd

# Run
./vecadd
```

**Output:**
```
C[0] = 0.000000
C[1] = 3.000000
C[2] = 6.000000
C[3] = 9.000000
C[4] = 12.000000
...
```

---

## 8. Key CUDA Keywords and Qualifiers

| Keyword | Meaning |
|---------|---------|
| `__global__` | Kernel function (called from host, runs on device) |
| `__device__` | Device function (called from device, runs on device) |
| `__host__` | Host function (called from host, runs on host) |
| `__shared__` | Shared memory variable |
| `__constant__` | Constant memory variable |
| `__restrict__` | Pointer aliasing hint for optimization |

### Function Qualifiers Combinations

```cuda
__global__ void kernel();           // GPU kernel (entry point)
__device__ void deviceFunc();       // GPU helper function
__host__ void hostFunc();           // CPU function (default)
__host__ __device__ void both();    // Compiles for both CPU and GPU
```

---

## 9. Compute Capability

Every NVIDIA GPU has a **Compute Capability** version that determines its features and limits:

| Compute Capability | Architecture | Key Features |
|-------------------|--------------|--------------|
| 5.x | Maxwell | Dynamic Parallelism, shared atomics |
| 6.x | Pascal | Unified Memory, FP16 |
| 7.x | Volta/Turing | Tensor Cores, Independent Thread Scheduling |
| 8.x | Ampere | TF32, Async Copy, Larger caches |
| 9.x | Hopper | Thread Block Clusters, TMA |
| 10.x | Blackwell | 5th Gen Tensor Cores |

### Key Limits (Varies by Architecture)

| Limit | Typical Value |
|-------|---------------|
| Max threads per block | 1024 |
| Max blocks per SM | 16-32 |
| Warp size | 32 (always) |
| Max warps per SM | 32-64 |
| Max shared memory per SM | 64-228 KB |
| Max registers per thread | 255 |

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **CPU vs GPU** | CPU = latency optimized, GPU = throughput optimized |
| **Heterogeneous Computing** | Host (CPU) + Device (GPU) work together |
| **Kernel** | Function that runs on millions of GPU threads |
| **Thread Hierarchy** | Thread → Block → Grid |
| **Built-in Variables** | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` |
| **Warp** | 32 threads executing in lockstep (SIMT) |
| **Warp Divergence** | Different paths in a warp = serialization |
| **Memory Hierarchy** | Registers → Shared → Global (fast → slow) |
| **PTX** | Portable intermediate representation |
| **Cubin** | Native binary for specific GPU architecture |

### The Optimization Mindset

```
                    GPU Performance Formula
    ────────────────────────────────────────────────────
    
    Performance = (Compute) × (Memory Bandwidth) × (Occupancy)
                  ─────────   ─────────────────   ───────────
                  Use all      Hide latency       Utilize all
                  threads      with parallelism   resources
```

CUDA transforms the GPU from a graphics card into a massive parallel computer. By understanding:
- The **Host/Device** distinction
- The hierarchy of **Grids, Blocks, and Threads**
- The hardware reality of **SMs and Warps**
- The **Memory Hierarchy**

...you can write applications that are orders of magnitude faster than CPU-only code.

---

## What's Next?

This introduction covered the fundamentals. In upcoming posts, we'll dive into:

1. **Vector Addition** - 
2. **Matrix Multiplication** - The workhorse of deep learning
3. **Parallel Reduction** - Summing millions of numbers efficiently
4. **Memory Coalescing** - Maximizing memory bandwidth
5. **Occupancy Optimization** - Keeping the GPU fully utilized
6. **Profiling with NSight** - Finding and fixing bottlenecks

The journey to GPU mastery starts with understanding these fundamentals. Master them, and you'll be ready to tackle the most demanding parallel computing challenges!

---

*Welcome to the world of GPU computing!*
