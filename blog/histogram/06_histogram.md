# Mastering Memory Contention: High-Performance Histograms

> **How to count millions of items without locking up your GPU**

---

## 1. Introduction

### The Hook

In [Matrix Transpose](../transpose/05_matrix_transpose.md), we learned that the GPU hates it when you access memory in a scattered pattern (strided access).

Today, we face a different enemy: **Popularity**.

What happens when 10,000 threads all try to write to the **same memory address** at the same time? This is the core problem of computing a **Histogram** — counting the frequency of values (e.g., pixel intensities in an image).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Histogram: What We're Counting                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: Array of values (e.g., pixel intensities 0-255)               │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐   │
│   │ 3 │ 7 │ 3 │ 2 │ 7 │ 7 │ 3 │ 1 │ 2 │ 3 │ 7 │ 0 │ 3 │ 2 │ 1 │ 7 │   │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘   │
│                                                                         │
│   Output: Count of each value (Histogram)                              │
│                                                                         │
│   bins[0] = 1    █                                                     │
│   bins[1] = 2    ██                                                    │
│   bins[2] = 3    ███                                                   │
│   bins[3] = 5    █████                                                 │
│   bins[4] = 0                                                          │
│   bins[5] = 0                                                          │
│   bins[6] = 0                                                          │
│   bins[7] = 5    █████                                                 │
│                                                                         │
│   Use cases: Image analysis, statistics, machine learning, etc.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Problem: Race Conditions

Consider what happens when two threads try to increment the same bin:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Race Condition Disaster                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Initial state: bins[7] = 5                                           │
│                                                                         │
│   Thread A                          Thread B                            │
│   ────────                          ────────                            │
│   1. READ bins[7]  → gets 5         1. READ bins[7]  → gets 5          │
│   2. ADD 1         → computes 6     2. ADD 1         → computes 6      │
│   3. WRITE bins[7] ← stores 6       3. WRITE bins[7] ← stores 6        │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   EXPECTED: bins[7] = 7 (two increments)                               │
│   ACTUAL:   bins[7] = 6 (one increment LOST!)                          │
│                                                                         │
│   ⚠️ This is called a "Lost Update" race condition                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The solution is **Atomics** — Read-Modify-Write operations that cannot be interrupted.

```cpp
atomicAdd(&bins[val], 1);  // Guaranteed to be atomic
```

But atomics come with a cost: **Serialization**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Serialization Problem                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Scenario: 1000 threads all want to increment bins[128]               │
│   (e.g., processing a gray image where most pixels = 128)              │
│                                                                         │
│                    Global Memory bins[128]                              │
│                           ┌───┐                                         │
│                           │128│                                         │
│                           └─┬─┘                                         │
│                             │                                           │
│            ┌────────────────┼────────────────┐                         │
│            │                │                │                         │
│            ▼                ▼                ▼                         │
│   Thread 0: atomicAdd  Thread 1: WAIT   Thread 999: WAIT              │
│   Thread 1: atomicAdd  Thread 2: WAIT   ...                           │
│   Thread 2: atomicAdd  Thread 3: WAIT                                 │
│   ...                                                                   │
│                                                                         │
│   The GPU becomes a SINGLE-LANE CHECKOUT LINE!                         │
│   Parallelism → 0                                                      │
│                                                                         │
│   ⚠️ This is why naive histograms are SLOW on "boring" data           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

If 1,000 threads try to `atomicAdd` to Bin 128 simultaneously, they form a queue. The GPU stops being parallel and becomes a single-lane checkout line.

---

## 2. The Naive Approach (Global Atomics)

The simplest implementation: one global array of bins, every thread updates it directly.

```cpp
__global__ void histogram_global(int* input, int* bins, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop: each thread processes multiple elements
    for (int i = tid; i < n; i += stride) {
        int val = input[i];
        atomicAdd(&bins[val], 1);  // Direct atomic to global memory
    }
}
```

### Why It Fails on "Boring" Data

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Data Distribution Matters!                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CASE 1: Random Noise (Uniform Distribution)                          │
│   ════════════════════════════════════════════                          │
│                                                                         │
│   Input: [42, 187, 3, 255, 128, 91, 7, 203, ...]                       │
│                                                                         │
│   Histogram:                                                            │
│   ████████████████████████████████████████████  (fairly uniform)       │
│                                                                         │
│   Collisions per bin: ~N/256 → LOW CONTENTION                          │
│   Performance: ~45 GB/s ✓                                              │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   CASE 2: Solid Color (All Same Value)                                 │
│   ════════════════════════════════════                                  │
│                                                                         │
│   Input: [128, 128, 128, 128, 128, 128, 128, ...]  (white wall photo)  │
│                                                                         │
│   Histogram:                                                            │
│                               ████████████████████████████████████████ │
│   bin 0                       bin 128                          bin 255 │
│                                                                         │
│   Collisions on bin 128: ALL N THREADS → MAXIMUM CONTENTION            │
│   Performance: ~0.8 GB/s ✗ (56× slower!)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

On a "flat" image (e.g., a picture of a white wall), millions of threads hammer `bins[255]`. The memory controller effectively locks, and performance plummets to **near zero**.

---

## 3. The Solution: Privatization

### The Concept

We use a strategy called **Privatization** (similar to how MapReduce works):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Privatization: The MapReduce of Histograms           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   BEFORE: All threads fight for Global Memory                          │
│   ══════════════════════════════════════════                            │
│                                                                         │
│   Block 0  Block 1  Block 2  Block 3  ...  Block N                     │
│      │        │        │        │             │                         │
│      └────────┴────────┴────────┴─────────────┘                         │
│                        │                                                │
│                        ▼                                                │
│               Global bins[256]  ← N×1024 collisions!                   │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   AFTER: Each block has a PRIVATE histogram                            │
│   ═════════════════════════════════════════                             │
│                                                                         │
│   Block 0         Block 1         Block 2         Block N              │
│   ┌───────┐       ┌───────┐       ┌───────┐       ┌───────┐            │
│   │s_bins │       │s_bins │       │s_bins │       │s_bins │            │
│   │[256]  │       │[256]  │       │[256]  │       │[256]  │            │
│   │Shared │       │Shared │       │Shared │       │Shared │            │
│   │Memory │       │Memory │       │Memory │       │Memory │            │
│   └───┬───┘       └───┬───┘       └───┬───┘       └───┬───┘            │
│       │               │               │               │                 │
│       └───────────────┴───────────────┴───────────────┘                 │
│                               │                                         │
│                               ▼                                         │
│                      Global bins[256]  ← Only N collisions!            │
│                                                                         │
│   Collisions reduced from N×1024 to just N (number of blocks)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The three phases:

| Phase | Description | Memory |
|-------|-------------|--------|
| **1. Private Copies** | Each block creates its own mini-histogram in Shared Memory | Fast, on-chip |
| **2. Local Counting** | Threads update only the local histogram | Low latency, fewer collisions |
| **3. Merge** | Block adds its local results to global memory | One atomic per bin per block |

### The Math

Instead of $N$ collisions in slow Global Memory, we have:

- $N$ collisions in **fast Shared Memory** (handled by SM hardware, ~100× lower latency)
- `gridDim.x` collisions in Global Memory (one per block, not per thread!)

For 1M elements with 256 blocks:
- **Before:** 1,000,000 global atomics (worst case: all to one bin)
- **After:** 1,000,000 shared atomics + 256 global atomics

---

## 4. Implementation: Shared Memory Histogram

### Step 1: Initialization

Shared memory is **uninitialized garbage** at startup. We must zero it out.

```cpp
__shared__ int s_bins[256];

// Collaborative clearing: each thread zeros one (or more) bins
// Assumes blockDim.x >= 256
if (threadIdx.x < 256) {
    s_bins[threadIdx.x] = 0;
}
__syncthreads();  // CRITICAL: Wait for all zeros before counting!
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Shared Memory Initialization                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   BEFORE: Shared memory contains GARBAGE                               │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐                            │
│   │????│????│????│????│????│????│????│????│  ...  (256 bins)           │
│   └────┴────┴────┴────┴────┴────┴────┴────┘                            │
│                                                                         │
│   Thread 0: s_bins[0] = 0                                              │
│   Thread 1: s_bins[1] = 0                                              │
│   Thread 2: s_bins[2] = 0                                              │
│   ...                                                                   │
│   Thread 255: s_bins[255] = 0                                          │
│                                                                         │
│   __syncthreads()  ← BARRIER: Everyone must finish zeroing!            │
│                                                                         │
│   AFTER: Shared memory is clean                                        │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐                            │
│   │ 0  │ 0  │ 0  │ 0  │ 0  │ 0  │ 0  │ 0  │  ...  (256 bins)           │
│   └────┴────┴────┴────┴────┴────┴────┴────┘                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Local Aggregation

Threads read global data (coalesced!) and update shared bins (random access, but fast).

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = idx; i < n; i += stride) {
    int val = input[i];                // Coalesced read from global
    atomicAdd(&s_bins[val], 1);        // Atomic to shared (fast!)
}
```

**Why is shared memory atomic faster?**

| Property | Global Memory | Shared Memory |
|----------|--------------|---------------|
| Location | Off-chip DRAM | On-chip SRAM |
| Latency | ~400 cycles | ~20 cycles |
| Bandwidth | ~900 GB/s (total) | ~19 TB/s per SM |
| Atomic throughput | Low (serialized) | Higher (SM handles it) |

### Step 3: The Merge

After processing, the block writes its results to global memory.

```cpp
__syncthreads();  // Wait for everyone to finish counting!

// Each thread merges one bin
if (threadIdx.x < 256) {
    atomicAdd(&global_bins[threadIdx.x], s_bins[threadIdx.x]);
}
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Three-Phase Histogram Dance                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PHASE 1: Initialize                                                   │
│   ═══════════════════                                                   │
│   s_bins[256] = {0, 0, 0, ..., 0}                                      │
│   __syncthreads()                                                       │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   PHASE 2: Count (Grid-Stride Loop)                                    │
│   ═════════════════════════════════                                     │
│                                                                         │
│   Global Input:  [42][187][42][3][255][42][128][42]...                 │
│                    ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓                       │
│                   T0  T1  T2  T3  T4  T5  T6  T7                       │
│                    │   │   │   │   │   │   │   │                       │
│                    ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼                       │
│   s_bins:        [42] [187][42][3][255][42][128][42]                   │
│                  +1   +1   +1  +1  +1  +1   +1  +1                     │
│                                                                         │
│   After counting: s_bins[42] = 4, s_bins[187] = 1, ...                 │
│   __syncthreads()                                                       │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   PHASE 3: Merge                                                        │
│   ══════════════                                                        │
│                                                                         │
│   Thread 0:   atomicAdd(&global_bins[0],   s_bins[0])                  │
│   Thread 1:   atomicAdd(&global_bins[1],   s_bins[1])                  │
│   Thread 42:  atomicAdd(&global_bins[42],  s_bins[42])  → adds 4       │
│   ...                                                                   │
│   Thread 255: atomicAdd(&global_bins[255], s_bins[255])                │
│                                                                         │
│   Only 256 global atomics per block (not millions!)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Complete Kernel (Copy-Paste Ready)

```cpp
#define BIN_COUNT 256

__global__ void histogram_shared(int* input, int* global_bins, int n) {
    // ═══════════════════════════════════════════════════════
    // 1. Allocate Shared Memory for Local Histogram
    // ═══════════════════════════════════════════════════════
    __shared__ int s_bins[BIN_COUNT];

    // ═══════════════════════════════════════════════════════
    // 2. Initialize Shared Memory to 0
    // ═══════════════════════════════════════════════════════
    // Assumes blockDim.x >= BIN_COUNT (e.g., 256 or 512 threads)
    // If blockDim.x < 256, use a loop instead
    if (threadIdx.x < BIN_COUNT) {
        s_bins[threadIdx.x] = 0;
    }
    __syncthreads();  // ⚠️ CRITICAL: Must sync before counting!

    // ═══════════════════════════════════════════════════════
    // 3. Grid-Stride Loop: Count into Shared Memory
    // ═══════════════════════════════════════════════════════
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        int val = input[i];
        // Atomic increment in fast shared memory
        atomicAdd(&s_bins[val], 1);
    }

    __syncthreads();  // ⚠️ CRITICAL: Must sync before merging!

    // ═══════════════════════════════════════════════════════
    // 4. Merge: Add Shared Results to Global Memory
    // ═══════════════════════════════════════════════════════
    if (threadIdx.x < BIN_COUNT) {
        atomicAdd(&global_bins[threadIdx.x], s_bins[threadIdx.x]);
    }
}

// Host launch code
void launch_histogram(int* d_input, int* d_bins, int n) {
    // Zero the global bins first!
    cudaMemset(d_bins, 0, BIN_COUNT * sizeof(int));
    
    int blockSize = 256;
    int gridSize = min(256, (n + blockSize - 1) / blockSize);
    
    histogram_shared<<<gridSize, blockSize>>>(d_input, d_bins, n);
}
```

### Handling Different Block Sizes

If your block has fewer than 256 threads, use a loop for initialization:

```cpp
// Safe initialization for any block size
for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x) {
    s_bins[i] = 0;
}
__syncthreads();
```

---

## 6. Benchmarks: The "White Wall" Test

We test two scenarios to expose the weakness of global atomics:

| Scenario | Description | Contention Level |
|----------|-------------|------------------|
| **Uniform Noise** | Random values 0-255 | Low (spread across 256 bins) |
| **Solid Color** | All values are 128 | Maximum (all hit one bin) |

### Results

| Implementation | Random Data | Solid Color | Robustness |
|----------------|-------------|-------------|------------|
| Global Atomics | 45 GB/s | **0.8 GB/s** 😱 | ❌ Collapses |
| Shared Atomics | 180 GB/s | **120 GB/s** | ✅ Robust |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Histogram Performance Comparison                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   RANDOM DATA (Low Contention):                                        │
│   ─────────────────────────────                                         │
│   Shared Atomics  ████████████████████████████████████████  180 GB/s   │
│   Global Atomics  █████████                                  45 GB/s   │
│                   ↑ 4× faster                                          │
│                                                                         │
│   SOLID COLOR (Max Contention):                                        │
│   ─────────────────────────────                                         │
│   Shared Atomics  ████████████████████████████              120 GB/s   │
│   Global Atomics  █                                         0.8 GB/s   │
│                   ↑ 150× faster! (Global atomics COLLAPSE)             │
│                                                                         │
│   0        20       40       60       80      100      120  (GB/s)     │
│                                                                         │
│   ═══════════════════════════════════════════════════════════          │
│   KEY INSIGHT: Shared memory atomics are ROBUST to data distribution   │
│   ═══════════════════════════════════════════════════════════          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Shared Memory Wins

| Factor | Global Atomics | Shared Atomics |
|--------|----------------|----------------|
| Latency per atomic | ~400 cycles | ~20 cycles |
| Contention scope | All SMs compete | Only threads in one block |
| Hardware support | Memory controller | SM's shared memory unit |
| Worst-case scaling | O(N) serialization | O(N/gridDim) local + O(gridDim) global |

**Insight:** Shared memory atomics are not only faster, they are **robust**. Performance doesn't collapse even in the worst-case data distribution.

---

## 7. Advanced: Beyond 256 Bins

### The Shared Memory Limit

What if we have **4096 bins** (12-bit medical imaging) or **65536 bins** (16-bit HDR)?

| Bins | Shared Memory Needed | Feasibility |
|------|---------------------|-------------|
| 256 | 1 KB | ✅ Easy |
| 1024 | 4 KB | ✅ Fine |
| 4096 | 16 KB | ⚠️ Tight (limits occupancy) |
| 65536 | 256 KB | ❌ Exceeds shared memory! |

### Solutions for Large Bin Counts

**Option 1: Multiple Passes**
Process the histogram in chunks (bins 0-255, then 256-511, etc.).

```cpp
// Pseudo-code for multi-pass histogram
for (int bin_offset = 0; bin_offset < total_bins; bin_offset += 256) {
    histogram_shared_range<<<...>>>(input, bins, n, bin_offset, bin_offset + 255);
}
```

**Option 2: Warp Aggregation**
Before doing an atomic, check if other threads in the warp have the same value. Combine them first.

```cpp
// Advanced: Warp-level aggregation (CUDA 9+)
__device__ void warp_aggregated_atomicAdd(int* bins, int val) {
    // Find threads with same value
    unsigned mask = __match_any_sync(0xFFFFFFFF, val);
    
    // Only one thread per unique value does the atomic
    int leader = __ffs(mask) - 1;
    if (lane_id() == leader) {
        atomicAdd(&bins[val], __popc(mask));  // Add count of matching threads
    }
}
```

> **Note:** For this tutorial, we focus on the 256-bin case as it fits perfectly in shared memory and covers most image processing use cases.

---

## 8. Common Pitfalls

### 1. Forgetting to Initialize Shared Memory

```cpp
// ❌ WRONG: Shared memory is garbage!
__shared__ int s_bins[256];
// Immediately start counting... BOOM! Garbage + 1 = Garbage

// ✅ CORRECT: Zero it first
__shared__ int s_bins[256];
if (threadIdx.x < 256) s_bins[threadIdx.x] = 0;
__syncthreads();
```

### 2. Forgetting `__syncthreads()` After Initialization

```cpp
// ❌ WRONG: Some threads start counting before others finish zeroing
if (threadIdx.x < 256) s_bins[threadIdx.x] = 0;
// Missing __syncthreads()!
atomicAdd(&s_bins[val], 1);  // Thread 300 might run before Thread 0 zeros!

// ✅ CORRECT: Barrier ensures all zeros complete
if (threadIdx.x < 256) s_bins[threadIdx.x] = 0;
__syncthreads();
atomicAdd(&s_bins[val], 1);
```

### 3. Forgetting `__syncthreads()` Before Merge

```cpp
// ❌ WRONG: Merging before all threads finish counting
for (int i = idx; i < n; i += stride) {
    atomicAdd(&s_bins[input[i]], 1);
}
// Missing __syncthreads()!
if (threadIdx.x < 256) {
    atomicAdd(&global_bins[threadIdx.x], s_bins[threadIdx.x]);  // Incomplete count!
}

// ✅ CORRECT: Wait for all counting to complete
for (int i = idx; i < n; i += stride) {
    atomicAdd(&s_bins[input[i]], 1);
}
__syncthreads();
if (threadIdx.x < 256) {
    atomicAdd(&global_bins[threadIdx.x], s_bins[threadIdx.x]);
}
```

### 4. Not Zeroing Global Bins on Host

```cpp
// ❌ WRONG: Global bins might have leftover data
histogram_shared<<<grid, block>>>(input, bins, n);

// ✅ CORRECT: Zero global bins before kernel launch
cudaMemset(d_bins, 0, 256 * sizeof(int));
histogram_shared<<<grid, block>>>(input, bins, n);
```

### 5. Integer Overflow (Rare but Possible)

```cpp
// ⚠️ CAUTION: If a single block counts > 2 billion items
// int will overflow. For huge datasets, consider:
__shared__ unsigned long long s_bins[256];  // 64-bit counters
```

> **Note:** 64-bit shared memory atomics (`atomicAdd` for `unsigned long long`) are fully hardware-supported on **Pascal (SM 6.0) and later**. On older architectures (Maxwell, Kepler), 64-bit atomics may be emulated and significantly slower. Also note that each 64-bit counter uses two 32-bit registers worth of shared memory (2 KB total for 256 bins instead of 1 KB).

---

## 9. Challenge for the Reader

### Challenge 1: Cumulative Distribution Function (CDF)

A histogram tells you **frequency**. A CDF tells you the **percentile** (essential for Histogram Equalization in image processing).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Histogram vs. CDF                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Histogram (frequency):                                               │
│   bins = [2, 5, 3, 8, 1, ...]                                          │
│                                                                         │
│   CDF (cumulative sum):                                                │
│   cdf  = [2, 7, 10, 18, 19, ...]                                       │
│           ↑  ↑   ↑   ↑   ↑                                             │
│           │  │   │   │   └── 2+5+3+8+1                                 │
│           │  │   │   └────── 2+5+3+8                                   │
│           │  │   └────────── 2+5+3                                     │
│           │  └────────────── 2+5                                       │
│           └──────────────── 2                                          │
│                                                                         │
│   This is a PREFIX SUM (Scan) on the histogram!                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Challenge:** Take the output `global_bins` and perform a **Prefix Sum (Scan)** on it to generate the CDF.

**Hint:** You already wrote a Scan kernel in the [previous post](../scan/04_parallel_prefix_sum.md)!

### Challenge 2: Multi-Channel Histogram

For RGB images, you need 3 separate histograms (one per channel).

**Challenge:** Modify the kernel to compute histograms for all 3 channels in a single kernel launch.

**Hints:**
- Use `__shared__ int s_bins[3][256]` or `__shared__ int s_bins[768]`
- Each thread processes R, G, B values from one pixel

### Challenge 3: Sparse Histograms

What if you only care about a few bins (e.g., counting specific error codes)?

**Challenge:** Instead of a 256-bin array, use a hash map approach with fewer bins.

---

## Summary

### Key Takeaways

| Concept | Lesson |
|---------|--------|
| **Race Conditions** | Multiple threads writing to same address → lost updates |
| **Atomics** | Solve correctness but introduce serialization |
| **Privatization** | Give each block its own copy, merge at the end |
| **Data Distribution** | Performance depends on how "boring" your data is |
| **Robustness** | Shared memory atomics maintain performance regardless of distribution |

### The Optimization Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Histogram Optimization Path                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GLOBAL ATOMICS              SHARED ATOMICS                           │
│   ──────────────              ──────────────                           │
│                                                                         │
│   Correctness:                Correctness:                             │
│   ✅ Atomics prevent races    ✅ Atomics prevent races                 │
│                                                                         │
│   Random Data:                Random Data:                             │
│   ⚠️ 45 GB/s (OK)             ✅ 180 GB/s (4× faster)                  │
│                                                                         │
│   Worst-Case Data:            Worst-Case Data:                         │
│   ❌ 0.8 GB/s (COLLAPSE)      ✅ 120 GB/s (ROBUST)                     │
│                                                                         │
│   Contention:                 Contention:                              │
│   ❌ All threads compete      ✅ Local competition only                │
│                                                                         │
│         ─────────────────────────────────────────────►                 │
│                     Optimization Progress                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What's Next?

In the next post, we'll explore **CUDA Streams and Asynchronous Execution** — how to overlap memory transfers with computation, run multiple kernels concurrently, and hide latency. We've mastered algorithms (Reduction, Scan) and memory patterns (Transpose, Histogram). Now it's time to master **system-level optimization**.

---

## References

1. [NVIDIA CUDA C++ Programming Guide — Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
2. [NVIDIA CUDA C++ Best Practices Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
3. Sengupta, S., Harris, M., Zhang, Y., & Owens, J. D. — *Scan Primitives for GPU Computing* (Graphics Hardware 2007)
4. [GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
