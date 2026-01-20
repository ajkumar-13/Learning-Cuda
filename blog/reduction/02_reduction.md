# The Art of Reduction: Calculating Loss at Light Speed

> **Why summing numbers is the hardest easy problem on a GPU**

---

## 1. Introduction

### The Hook

In the [last post](./01_matmul.md), we multiplied matrices. That was fundamentally a **"map and expand"** operation—each output element is computed independently by mapping inputs through a dot product. The result is *larger* or *equal* in size to the inputs.

Now we face the opposite challenge: **compression**. We need to take millions of numbers and reduce them to *one*.

### The Use Case

Reduction operations are everywhere in computing:

| Domain | Operation | What Gets Reduced |
|--------|-----------|-------------------|
| **Deep Learning** | Loss calculation (MSE, Cross-Entropy) | Batch of errors → single scalar |
| **Statistics** | Mean, Variance, Standard Deviation | Dataset → summary statistics |
| **Softmax** | Normalization denominator | Vector of logits → sum of exponentials |
| **Image Processing** | Histogram, Global thresholding | Pixels → aggregated values |

### The Problem

On a CPU, reduction is trivial:

```c
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += array[i];
}
```

On a GPU with 1,000 threads? **Chaos.**

```c
// ❌ BROKEN: Race condition!
__global__ void naive_sum(float* array, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        *result += array[idx];  // 1000 threads fighting over one variable
    }
}
```

When multiple threads try to update the same memory location simultaneously, we get a **race condition**. The reads and writes interleave unpredictably, producing garbage results.

**The "obvious" fix—atomic operations—creates a new problem:**

```c
// ✅ Correct, but painfully slow
atomicAdd(result, array[idx]);  // Threads serialize, waiting in line
```

Atomics force threads to take turns, converting our parallel algorithm back into a serial one. We've traded correctness for performance.

**The question:** How do we sum numbers in parallel *correctly* and *fast*?

---

## 2. The Algorithm: Tree-Based Reduction

### The Insight

Instead of having all threads write to one location, we structure the computation as a **binary tree**:

```
Step 0 (Initial):     [1] [2] [3] [4] [5] [6] [7] [8]
                        ↘↙     ↘↙     ↘↙     ↘↙
Step 1 (4 adds):      [ 3 ]  [ 7 ]  [11 ]  [15 ]
                          ↘↙         ↘↙
Step 2 (2 adds):      [  10  ]    [  26  ]
                            ↘↙
Step 3 (1 add):       [     36     ]
```

### Complexity Analysis

| Metric | Value | Explanation |
|--------|-------|-------------|
| **Total Work** | $O(N)$ | Still N-1 additions |
| **Parallel Steps** | $O(\log N)$ | Tree height |
| **Speedup** | $O(N / \log N)$ | Theoretical parallelism |

For $N = 1,000,000$:
- Sequential: 1,000,000 steps
- Parallel: ~20 steps (with enough threads)
- **Speedup: 50,000×** (in theory)

The tree structure is the key to parallel reduction. Now let's implement it on a GPU.

---

## 3. Attempt 1: Shared Memory Reduction (The "Classic" Approach)

### The Strategy

1. Each thread loads one element from **Global Memory** to **Shared Memory**
2. Threads synchronize (`__syncthreads()`)
3. We iteratively "fold" the array in half, reducing pairs
4. After $\log_2(\text{blockDim.x})$ iterations, thread 0 has the sum

### The Naive Implementation

```cuda
#define BLOCK_SIZE 256

__global__ void reduce_shared_naive(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global memory to shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory (NAIVE - interleaved addressing)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // ⚠️ Problem here!
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
```

### The Pitfall: Warp Divergence

Look at the condition: `if (tid % (2 * stride) == 0)`

In the first iteration (`stride = 1`):
- Even threads (0, 2, 4, ...) are **active**
- Odd threads (1, 3, 5, ...) are **idle**

**Within a single warp (32 threads):**
- 16 threads execute the addition
- 16 threads do nothing
- But the hardware runs **both paths serially**!

```
Warp 0:  [Active][Idle][Active][Idle]...[Active][Idle]
          T0     T1    T2     T3        T30    T31
          
         └── 50% utilization! ──┘
```

This is called **warp divergence**, and it cuts our performance in half (or worse) at each level of the tree.

### The Fix: Sequential Addressing

Instead of interleaved access (`tid` and `tid + 1`), use **sequential** access (`tid` and `tid + blockDim/2`):

```cuda
__global__ void reduce_shared_optimized(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - no divergence!
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {  // ✅ First half of threads active
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
```

**Why this is better:**

```
stride = 128:  Threads 0-127 active   (4 full warps)
stride = 64:   Threads 0-63 active    (2 full warps)
stride = 32:   Threads 0-31 active    (1 full warp)
stride = 16:   Threads 0-15 active    (½ warp - some divergence)
...
```

We only get divergence in the **last 5 iterations** (within a single warp), not at every level. Much better!

### Visual: Interleaved vs Sequential

```
INTERLEAVED (Bad):                SEQUENTIAL (Good):
Step 1:                           Step 1:
[0]+[1] [2]+[3] [4]+[5]...       [0]+[128] [1]+[129] [2]+[130]...
  ↓       ↓       ↓                  ↓         ↓         ↓
[0]     [2]     [4]              [0]       [1]       [2]
                                 
Threads: 0,2,4,6... (scattered)  Threads: 0,1,2,3... (contiguous!)
```

---

## 4. Attempt 2: Warp Shuffle (The "Modern" Approach)

### The Insight

Shared memory is fast (~10 TB/s effective bandwidth), but **registers are faster** (~20 TB/s). 

What if threads could read each other's registers directly?

### Enter Warp Shuffle

Starting with **Kepler (2012)** and refined in **Volta (2017)**, NVIDIA GPUs support **warp-level primitives**—instructions that let threads within the same warp communicate directly through registers.

**Key benefits:**
- No shared memory allocation needed
- No `__syncthreads()` inside the warp (threads are implicitly synchronized)
- Lower latency than shared memory access

### The Function: `__shfl_down_sync`

```cuda
T __shfl_down_sync(unsigned mask, T var, unsigned delta);
```

| Parameter | Meaning |
|-----------|---------|
| `mask` | Bitmask of participating threads (usually `0xffffffff` for all 32) |
| `var` | The value to share |
| `delta` | How many lanes "down" to read from |
| **Returns** | The value of `var` from thread `(laneId + delta)` |

### Visual: Warp Shuffle Reduction

```
Initial:   T0   T1   T2   T3   ... T15  T16  T17  ... T31
           [a0] [a1] [a2] [a3]     [a15][a16][a17]    [a31]

delta=16:  T0 reads T16, T1 reads T17, ...
           [a0+a16] [a1+a17] [a2+a18] ... [a15+a31] [--] [--] ... [--]

delta=8:   T0 reads T8, T1 reads T9, ...
           [sum of 4 elements per thread for T0-T7]

delta=4:   ...

delta=2:   ...

delta=1:   T0 reads T1
           [FINAL SUM in T0]
```

### The Famous Warp Reduce Pattern

```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Only lane 0 has the correct total
}
```

**That's it.** Five iterations, no shared memory, no explicit synchronization. Each thread ends up with a partial result, and lane 0 has the sum of all 32 values.

### Why `0xffffffff`?

This is the **participation mask**—a 32-bit value where each bit indicates whether that lane participates in the shuffle.

```
0xffffffff = 1111 1111 1111 1111 1111 1111 1111 1111 (binary)
             ↑    All 32 lanes participate
```

**⚠️ Important for Volta+ architectures:** You must always specify the correct mask. Older code that omitted the mask worked by accident on pre-Volta hardware but will produce incorrect results on modern GPUs.

---

## 5. The Complete Kernel: Block Reduce + Atomic Add

### The Three-Level Strategy

Real-world data has millions of elements. We need a hierarchical approach:

```
┌─────────────────────────────────────────────────────────────┐
│ GRID LEVEL: Millions of elements split across blocks        │
│   └─> Each block handles 256-1024 elements                  │
├─────────────────────────────────────────────────────────────┤
│ BLOCK LEVEL: Warp shuffles + shared memory for cross-warp   │
│   └─> Reduce 256 elements → 1 partial sum                   │
├─────────────────────────────────────────────────────────────┤
│ GLOBAL LEVEL: atomicAdd partial sums                        │
│   └─> ~100-1000 atomic operations (acceptable!)             │
└─────────────────────────────────────────────────────────────┘
```

**Why atomics are okay at the grid level:**

| Approach | Atomic Collisions | Performance |
|----------|-------------------|-------------|
| Naive (per-thread) | N = 1,000,000 | ❌ Terrible |
| Block reduce | gridDim.x ≈ 1,000 | ✅ Acceptable |

### The Complete Kernel

```cuda
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction: warps → shared memory → final warp
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];  // 8 floats for 256 threads
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Each warp reduces to its lane 0
    val = warp_reduce_sum(val);
    
    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    // Warp 0 now loads the partial sums from OTHER warps and reduces them
    // This is the critical "inter-warp communication" step!
    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Full reduction kernel
__global__ void reduce_sum(const float* __restrict__ input,
                           float* __restrict__ output,
                           int N) {
    float sum = 0.0f;
    
    // Grid-stride loop: each thread sums multiple elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }
    
    // Block-level reduction
    sum = block_reduce_sum(sum);
    
    // Thread 0 of each block adds to global result
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}
```

### Kernel Launch

```cuda
void launch_reduce(const float* d_input, float* d_output, int N) {
    // Reset output
    cudaMemset(d_output, 0, sizeof(float));
    
    // Calculate grid size (cap at reasonable number)
    int num_blocks = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    
    reduce_sum<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
}
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Grid-stride loop** | Each thread handles multiple elements, improving arithmetic intensity |
| **Capped grid size** | Too many blocks = too many atomic collisions |
| **Warp shuffle first** | Fastest possible reduction within warp |
| **Shared memory for cross-warp** | Required for inter-warp communication |

> 💡 **Pro Tip: Grid-Stride Loops Decouple Problem Size from Grid Size**
>
> Notice how the kernel uses `for (int i = idx; i < N; i += stride)` instead of a simple `if (idx < N)`. This **grid-stride loop** pattern is powerful because:
> - You can launch fewer blocks than elements (cap at 1024 blocks, handle 16M elements)
> - Each thread does more work, improving arithmetic intensity
> - The kernel handles *any* input size without recompilation
> - Memory accesses remain coalesced (consecutive threads read consecutive addresses)
>
> This is how production CUDA code handles arbitrary-sized inputs!

---

## 6. Benchmarks: The Proof

Let's compare implementations on a **GTX 1650 Max-Q** (128 GB/s theoretical bandwidth):

### Test Setup

```cuda
// Test configuration
const int N = 16 * 1024 * 1024;  // 16M elements = 64 MB
const int ITERATIONS = 100;

// Implementations tested:
// 1. CPU sequential loop
// 2. Naive atomic (every thread does atomicAdd)
// 3. Shared memory reduction (sequential addressing)
// 4. Warp shuffle reduction (our optimized kernel)
```

### Results

| Implementation | Time (ms) | Bandwidth (GB/s) | Speedup |
|----------------|-----------|------------------|---------|
| CPU Sequential | 12.4 | 5.2 | 1.0× |
| Naive Atomic | 847.3 | 0.08 | 0.01× 😱 |
| Shared Memory | 0.62 | 103 | 20× |
| **Warp Shuffle** | **0.51** | **125** | **24×** |

### Analysis

**Naive Atomic** is catastrophically slow—worse than CPU! Every thread serializes on one memory location.

**Warp Shuffle** achieves **~98% of theoretical bandwidth**. We're limited only by how fast we can read memory, not by our reduction algorithm. This is the goal: make compute "free" and become **memory-bound**.

```
Bandwidth Utilization:

CPU:          ████░░░░░░░░░░░░░░░░░░░░░░░░░░  4%
Naive Atomic: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  <1%
Shared Mem:   ████████████████████████░░░░░░  80%
Warp Shuffle: ██████████████████████████████  98%
              0%                           100%
```

---

## 7. Complete Implementation

Here's the full, copy-pasteable code:

```cuda
// reduction.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================
// Warp-level reduction
// ============================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================
// Block-level reduction
// ============================================================
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// ============================================================
// Reduction Kernels
// ============================================================

// Naive atomic - for comparison (DON'T USE!)
__global__ void reduce_naive_atomic(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}

// Shared memory reduction
__global__ void reduce_shared(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Optimized warp shuffle reduction
__global__ void reduce_warp_shuffle(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N) {
    float sum = 0.0f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }
    
    sum = block_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ============================================================
// CPU Reference
// ============================================================
float cpu_reduce(const float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

// ============================================================
// Benchmark Helpers
// ============================================================
typedef void (*reduce_kernel_t)(const float*, float*, int);

float benchmark_kernel(reduce_kernel_t kernel, const float* d_input, 
                       float* d_output, int N, int num_blocks,
                       int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    cudaMemset(d_output, 0, sizeof(float));
    kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_output, 0, sizeof(float));
        kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

// ============================================================
// Main
// ============================================================
int main() {
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int ITERATIONS = 100;
    const size_t bytes = N * sizeof(float);
    
    printf("===========================================\n");
    printf("   Parallel Reduction Benchmark\n");
    printf("===========================================\n");
    printf("Elements: %d (%.1f MB)\n", N, bytes / (1024.0f * 1024.0f));
    printf("Block size: %d\n", BLOCK_SIZE);
    printf("\n");
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float h_output;
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;  // Values in [0, 1]
    }
    
    // CPU reference
    float cpu_result = cpu_reduce(h_input, N);
    printf("CPU Result: %.6f\n\n", cpu_result);
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    int num_blocks_naive = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_opt = min(num_blocks_naive, 1024);
    
    printf("%-20s %10s %12s %10s %10s\n", 
           "Kernel", "Time(ms)", "Bandwidth", "Result", "Error");
    printf("-------------------------------------------------------------------\n");
    
    // Benchmark naive atomic
    float time_naive = benchmark_kernel(reduce_naive_atomic, d_input, d_output, 
                                        N, num_blocks_naive, ITERATIONS);
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("%-20s %10.3f %10.1f GB/s %10.2f %10.6f\n",
           "Naive Atomic", time_naive, bytes / time_naive / 1e6, 
           h_output, fabsf(h_output - cpu_result));
    
    // Benchmark shared memory
    float time_shared = benchmark_kernel(reduce_shared, d_input, d_output,
                                         N, num_blocks_naive, ITERATIONS);
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("%-20s %10.3f %10.1f GB/s %10.2f %10.6f\n",
           "Shared Memory", time_shared, bytes / time_shared / 1e6,
           h_output, fabsf(h_output - cpu_result));
    
    // Benchmark warp shuffle
    float time_warp = benchmark_kernel(reduce_warp_shuffle, d_input, d_output,
                                       N, num_blocks_opt, ITERATIONS);
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    printf("%-20s %10.3f %10.1f GB/s %10.2f %10.6f\n",
           "Warp Shuffle", time_warp, bytes / time_warp / 1e6,
           h_output, fabsf(h_output - cpu_result));
    
    printf("\n");
    printf("Speedup (Warp Shuffle vs Naive): %.1fx\n", time_naive / time_warp);
    printf("Speedup (Warp Shuffle vs Shared): %.2fx\n", time_shared / time_warp);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    
    return 0;
}
```

### Compilation

```bash
nvcc -O3 -arch=sm_75 reduction.cu -o reduction
./reduction
```

---

## 8. Common Pitfalls (The "Gotchas")

### Pitfall 1: Missing Sync Mask (Volta+)

```cuda
// ❌ Old (pre-Volta) code - BROKEN on modern GPUs!
val += __shfl_down(val, offset);

// ✅ Correct for Volta and later
val += __shfl_down_sync(0xffffffff, val, offset);
```

**Why?** Volta introduced **Independent Thread Scheduling**. Threads in a warp can diverge and reconverge at different points. The `_sync` suffix with explicit mask ensures all specified threads participate in the shuffle simultaneously.

### Pitfall 2: Non-Power-of-Two Sizes

Our kernel handles this with:

```cuda
// Grid-stride loop handles any size
for (int i = idx; i < N; i += stride) {
    sum += input[i];
}

// OR with boundary checks
sdata[tid] = (idx < N) ? input[idx] : 0.0f;  // Pad with zeros
```

### Pitfall 3: Forgetting to Reset Output

```cuda
// ❌ Wrong - accumulates across kernel launches!
reduce_sum<<<blocks, threads>>>(d_in, d_out, N);

// ✅ Correct
cudaMemset(d_out, 0, sizeof(float));  // Reset first!
reduce_sum<<<blocks, threads>>>(d_in, d_out, N);
```

### Pitfall 4: Incorrect Warp Sum Index

```cuda
// ❌ Bug: what if BLOCK_SIZE/WARP_SIZE > 32?
if (warp_id == 0) {
    val = warp_sums[lane];  // lane only goes 0-31!
}

// ✅ Safe version
if (warp_id == 0) {
    val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
}
```

### Pitfall 5: The `volatile` Question (Historical Note)

You might see older CUDA code using `volatile` for shared memory:

```cuda
// Old-style code (pre-Volta)
volatile __shared__ float sdata[BLOCK_SIZE];
```

**Why did they do this?** Before Volta, the compiler could optimize away redundant loads from shared memory. If thread 0 reads `sdata[0]`, adds to it, and reads again, the compiler might reuse the old cached value instead of re-reading.

**Why we don't need it:** The `__syncthreads()` barrier acts as a **memory fence**—it forces all threads to complete their writes AND invalidates cached reads. Our code has `__syncthreads()` between every reduction step, so we're safe.

**With warp shuffles:** The `__shfl_down_sync` intrinsic has implicit synchronization, so `volatile` is never needed.

> 💡 If someone asks "where's your `volatile`?"—point them to the `__syncthreads()` calls!

---

## 9. Challenge for the Reader

We implemented **Sum**. Can you implement **Max**?

### Hints

1. Change `+` to `max()`:
   ```cuda
   val = max(val, __shfl_down_sync(0xffffffff, val, offset));
   ```

2. Change the initial value from `0` to `-INFINITY`:
   ```cuda
   float local_max = -INFINITY;
   for (int i = idx; i < N; i += stride) {
       local_max = max(local_max, input[i]);
   }
   ```

3. Use `atomicMax` instead of `atomicAdd`... but wait, there's no `atomicMax` for floats! 🤔

### Bonus Challenge: Implementing Float atomicMax

```cuda
__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    
    do {
        expected = old;
        if (__int_as_float(expected) >= value) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (expected != old);
    
    return __int_as_float(old);
}
```

> ⚠️ **IEEE 754 Caveat: NaN and Negative Zero**
>
> This implementation works for "normal" floats but has edge cases:
> - **NaN values**: `NaN >= x` is always false, so NaN can corrupt results
> - **Negative zero**: `-0.0f` and `+0.0f` compare equal but have different bit patterns
> - **Negative floats**: The bit-casting trick relies on IEEE 754 ordering, which only works correctly for positive floats (negative floats have inverted ordering when viewed as integers)
>
> For production code with negative values, you'll need additional checks or a different approach.

This uses `atomicCAS` (Compare-And-Swap) to implement a lock-free maximum. It's a great exercise in understanding atomic operations!

---

## 10. What's Next?

We've now covered the three fundamental GPU patterns:

| Pattern | Operation | Blog Post |
|---------|-----------|-----------|
| **Map** | Element-wise operations | Vector Addition |
| **Multiply** | Parallel dot products | Matrix Multiplication |
| **Reduce** | Compression/aggregation | **This post** |

**Coming up:** We'll combine these patterns to implement real-world algorithms:
- **Softmax**: Reduction (max, sum) + Map (exp, divide)
- **LayerNorm**: Mean reduction + Variance reduction + Normalization
- **Attention**: Matrix multiply + Softmax + Matrix multiply

These are the building blocks of modern transformers, and now you have the foundation to understand and optimize them!

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Race Conditions** | Multiple threads writing to same location = garbage |
| **Atomics** | Correct but serialize execution (slow for N threads) |
| **Tree Reduction** | $O(\log N)$ steps instead of $O(N)$ |
| **Warp Divergence** | Threads taking different paths serialize |
| **Sequential Addressing** | Keep contiguous threads active together |
| **Warp Shuffle** | Register-to-register communication, no shared memory needed |
| **Block Reduce** | Warp shuffle → shared memory → final warp shuffle |
| **Grid Reduce** | Block reduces + limited atomics |

**The optimization journey:**
```
Naive Atomic     →    Shared Memory    →    Warp Shuffle
   0.08 GB/s            103 GB/s            125 GB/s
     (bad)               (good)            (optimal)
```

Master reduction, and you've mastered the hardest part of GPU programming. Everything else builds on these patterns.

---

*Happy reducing! 🚀*
