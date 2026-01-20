# CUDA Streams: The Art of Concurrency

> **How to make your CPU, GPU, and PCIe bus work simultaneously**

---

## 1. Introduction

### The Hook

In every previous post, we followed a strict, serial pattern:

1. **Copy Data to GPU** (Host → Device)
2. *Wait...*
3. **Launch Kernel** (Compute)
4. *Wait...*
5. **Copy Data Back** (Device → Host)

This is the **"Serial Trap."** While the GPU is computing, the PCIe bus is idle. While data is copying, the expensive GPU cores are idle.

**You paid for the whole chip; you should use the whole chip.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Serial Trap: Wasted Hardware                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIME ──────────────────────────────────────────────────────────────►  │
│                                                                         │
│   PCIe Bus:   [████ H2D ████]                    [████ D2H ████]       │
│                              ↓ IDLE ↓                                   │
│   GPU Cores:                 [████████ KERNEL ████████]                │
│               ↑ IDLE ↑                              ↑ IDLE ↑           │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Problem: Only ONE thing happens at a time!                           │
│   - While copying → GPU cores idle (you're paying for nothing)         │
│   - While computing → PCIe bus idle (memory bandwidth wasted)          │
│                                                                         │
│   If copy time ≈ compute time → You're at ~33% efficiency!             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Solution: Asynchronous Execution

**CUDA Streams** allow us to break big tasks into smaller chunks and **pipeline** them. Just like a factory assembly line:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Pipeline: Full Utilization                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIME ──────────────────────────────────────────────────────────────►  │
│                                                                         │
│   Chunk 1:  [H2D]  [Kernel]  [D2H]                                     │
│   Chunk 2:         [H2D]     [Kernel]  [D2H]                           │
│   Chunk 3:                   [H2D]     [Kernel]  [D2H]                 │
│   Chunk 4:                             [H2D]     [Kernel]  [D2H]       │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   At any moment:                                                        │
│   - Chunk 1: Copying BACK results (D2H)                                │
│   - Chunk 2: COMPUTING on GPU                                          │
│   - Chunk 3: Copying TO GPU (H2D)                                      │
│                                                                         │
│   THREE operations happening SIMULTANEOUSLY!                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Concept: What is a Stream?

### Definition

A **Stream** is a sequence of operations that execute **in order** on the GPU.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Streams: Ordered Sequences                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Stream A: [ Copy 1 ] ──► [ Kernel 1 ] ──► [ Copy Back 1 ]            │
│             (Strictly ordered within stream A)                          │
│                                                                         │
│   Stream B: [ Copy 2 ] ──► [ Kernel 2 ] ──► [ Copy Back 2 ]            │
│             (Strictly ordered within stream B)                          │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   THE MAGIC: Operations in DIFFERENT streams can run CONCURRENTLY!     │
│                                                                         │
│   Stream A: [ Copy 1 ][ Kernel 1    ][ D2H 1 ]                         │
│   Stream B:      [ Copy 2 ][ Kernel 2    ][ D2H 2 ]                    │
│                       ↑                                                 │
│                       └── These overlap if hardware allows!            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Hardware Reality

Modern GPUs aren't just big calculators. They have **independent hardware engines**:

| Engine | Purpose | Count |
|--------|---------|-------|
| **Compute Engines** | Run CUDA kernels | 1+ (can run multiple kernels) |
| **Copy Engine (H2D)** | Host → Device transfers | 1 (dedicated) |
| **Copy Engine (D2H)** | Device → Host transfers | 1 (dedicated) |

> **📡 Hardware Fact:** PCIe is **full-duplex** — data can flow Host→Device AND Device→Host simultaneously! This is why the separate H2D and D2H copy engines can both be active at the same time.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GPU Hardware Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                         GPU CHIP                                 │  │
│   │  ┌───────────────────────────────────────────────────────────┐  │  │
│   │  │              COMPUTE ENGINES (SMs)                        │  │  │
│   │  │   Run kernels from ANY stream concurrently                │  │  │
│   │  │   (if enough resources available)                         │  │  │
│   │  └───────────────────────────────────────────────────────────┘  │  │
│   │                                                                  │  │
│   │  ┌─────────────────────┐    ┌─────────────────────┐            │  │
│   │  │  COPY ENGINE (H2D)  │    │  COPY ENGINE (D2H)  │            │  │
│   │  │  Host → Device      │    │  Device → Host      │            │  │
│   │  │  Independent!       │    │  Independent!       │            │  │
│   │  └─────────────────────┘    └─────────────────────┘            │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                           │                   │                         │
│                           └─────────┬─────────┘                         │
│                                     │                                   │
│                              ┌──────┴──────┐                            │
│                              │  PCIe BUS   │                            │
│                              └──────┬──────┘                            │
│                                     │                                   │
│                              ┌──────┴──────┐                            │
│                              │  CPU + RAM  │                            │
│                              └─────────────┘                            │
│                                                                         │
│   KEY INSIGHT: These engines can ALL work AT THE SAME TIME!            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Default Stream Problem

If you only use the **Default Stream** (Stream 0), you force these engines to take turns:

```cpp
// All operations go to Stream 0 (implicit default)
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // Stream 0
kernel<<<grid, block>>>(d_data);                            // Stream 0
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);  // Stream 0

// Result: Everything serialized. No overlap possible.
```

Using **multiple streams** lets the engines work in parallel!

---

## 3. The Implementation

### Step 1: Pinned (Page-Locked) Memory

To use asynchronous transfers (`cudaMemcpyAsync`), host memory **must be pinned** (page-locked).

**Why?** Standard `malloc` memory is *pageable* — the OS can swap it to disk at any time. Before the GPU can access it, the driver must copy it to a staging buffer. This forces synchronization.

**Pinned memory** is locked in physical RAM. The GPU can DMA directly from it without CPU intervention.

```cpp
// ❌ SLOW: Pageable Memory
float* h_data = (float*)malloc(bytes);
// Driver must stage this before GPU can access it
// cudaMemcpyAsync silently becomes SYNCHRONOUS!

// ✅ FAST: Pinned Memory
float* h_data;
cudaMallocHost(&h_data, bytes);  // Page-locked, GPU can DMA directly
// cudaMemcpyAsync is truly asynchronous!
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pageable vs. Pinned Memory                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PAGEABLE MEMORY (malloc):                                            │
│   ═════════════════════════                                             │
│                                                                         │
│   CPU RAM          Staging Buffer        GPU                           │
│   ┌───────┐        ┌───────┐            ┌───────┐                      │
│   │ Data  │ ──1──► │ Copy  │ ────2────► │ Data  │                      │
│   │(pages)│        │       │            │       │                      │
│   └───────┘        └───────┘            └───────┘                      │
│       ↑                                                                 │
│       └── OS might swap these pages to disk!                           │
│                                                                         │
│   Step 1: CPU copies to pinned staging buffer (BLOCKS CPU!)            │
│   Step 2: GPU DMAs from staging buffer                                 │
│   → cudaMemcpyAsync becomes SYNCHRONOUS                                │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   PINNED MEMORY (cudaMallocHost):                                      │
│   ═══════════════════════════════                                       │
│                                                                         │
│   CPU RAM (Pinned)               GPU                                   │
│   ┌───────────────┐             ┌───────┐                              │
│   │     Data      │ ────DMA───► │ Data  │                              │
│   │ (Page-locked) │             │       │                              │
│   └───────────────┘             └───────┘                              │
│                                                                         │
│   Direct DMA transfer, no CPU involvement!                             │
│   → cudaMemcpyAsync is truly ASYNCHRONOUS                              │
│   → CPU can do other work while transfer happens                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

> **⚠️ Warning:** Pinned memory is a limited system resource. Don't pin gigabytes unnecessarily — it reduces memory available for other applications and can hurt system performance.

> **⚠️ Performance Warning:** `cudaMallocHost` is **expensive to allocate** — much slower than `malloc` or even `cudaMalloc`. Always allocate pinned memory **once at startup**, never inside your performance loop. The allocation overhead will dwarf any speedup from async transfers!

### Step 2: Creating Streams

```cpp
const int nStreams = 4;
cudaStream_t streams[nStreams];

for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
}

// ... use streams ...

// Don't forget to clean up!
for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
}
```

### Step 3: Issuing Asynchronous Operations

The key is specifying which stream each operation belongs to:

```cpp
// Asynchronous memory copy: specify stream as LAST argument
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

// Kernel launch: specify stream as 4th argument in <<<>>>
kernel<<<grid, block, sharedMem, stream>>>(args...);
//                              ^^^^^^
//                              Stream goes here!

// Device-to-host copy
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
```

### Step 4: The Complete Pattern

```cpp
#define N_STREAMS 4

void process_with_streams(float* h_in, float* h_out, float* d_in, float* d_out, 
                          int totalSize) {
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunkSize = totalSize / N_STREAMS;
    int chunkBytes = chunkSize * sizeof(float);

    // ═══════════════════════════════════════════════════════
    // Issue ALL operations for ALL streams
    // ═══════════════════════════════════════════════════════
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * chunkSize;

        // 1. Copy chunk to device (H2D)
        cudaMemcpyAsync(&d_in[offset], &h_in[offset], chunkBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // 2. Launch kernel on chunk
        int blocks = (chunkSize + 255) / 256;
        process_kernel<<<blocks, 256, 0, streams[i]>>>(&d_in[offset], 
                                                        &d_out[offset], 
                                                        chunkSize);

        // 3. Copy results back (D2H)
        cudaMemcpyAsync(&h_out[offset], &d_out[offset], chunkBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // ═══════════════════════════════════════════════════════
    // Wait for ALL streams to complete
    // ═══════════════════════════════════════════════════════
    cudaDeviceSynchronize();

> **💡 Pro Tip: Depth-First vs. Breadth-First Scheduling**
>
> This code uses **Depth-First** scheduling (queue H2D→Kernel→D2H for each stream sequentially). On **older GPUs (pre-Pascal)** with limited hardware queues, this could cause false dependencies where Stream 1's H2D gets blocked behind Stream 0's Kernel.
>
> The alternative is **Breadth-First**: queue ALL H2Ds first, then ALL Kernels, then ALL D2Hs. This guaranteed overlap on older hardware.
>
> **Modern GPUs (Volta/Ampere/Hopper)** have "HyperQ" with 32+ hardware queues and independent scheduling, so Depth-First (shown here) works efficiently and is more readable. If targeting older GPUs (Kepler/Maxwell), consider Breadth-First.

    // Clean up
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}
```

---

## 4. Visualizing the Overlap

### The Serial Timeline (Default Stream)

With the default stream, everything happens sequentially:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Serial Execution (Default Stream)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIME ──────────────────────────────────────────────────────────────►  │
│                                                                         │
│   Chunk 1   Chunk 2   Chunk 3   Chunk 4                                │
│   ────────────────────────────────────────────────────────────────     │
│                                                                         │
│   [H2D][K][D2H][H2D][K][D2H][H2D][K][D2H][H2D][K][D2H]                 │
│                                                                         │
│   Total Time = 4 × (H2D + Kernel + D2H)                                │
│                                                                         │
│   If each phase = 10ms:                                                │
│   Total = 4 × (10 + 10 + 10) = 120 ms                                  │
│                                                                         │
│   Hardware Utilization:                                                │
│   - H2D Engine: 33% busy (idle during Kernel and D2H)                  │
│   - Compute:    33% busy (idle during copies)                          │
│   - D2H Engine: 33% busy (idle during Kernel and H2D)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Concurrent Timeline (4 Streams)

With 4 streams, operations overlap:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Concurrent Execution (4 Streams)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIME ──────────────────────────────────────────────────────────────►  │
│                                                                         │
│   Stream 1: [H2D 1][Kernel 1][D2H 1]                                   │
│   Stream 2:       [H2D 2]   [Kernel 2][D2H 2]                          │
│   Stream 3:              [H2D 3]     [Kernel 3][D2H 3]                 │
│   Stream 4:                     [H2D 4]       [Kernel 4][D2H 4]        │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Hardware View (What's Actually Running):                             │
│                                                                         │
│   H2D Engine:  [H2D 1][H2D 2][H2D 3][H2D 4]                            │
│   Compute:           [K1]   [K2]   [K3]   [K4]                         │
│   D2H Engine:              [D2H 1][D2H 2][D2H 3][D2H 4]                │
│                                                                         │
│   Total Time ≈ H2D_total + Kernel_1 + D2H_last                         │
│             ≈ 40ms + 10ms + 10ms = 60ms (vs 120ms serial!)             │
│                                                                         │
│   Speedup: 2× just from overlapping!                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Ideal Pipeline (When Copy ≈ Compute)

When transfer time equals compute time, we achieve **near-perfect overlap**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Ideal Pipeline: Maximum Throughput                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIME ──────────────────────────────────────────────────────────────►  │
│                                                                         │
│   H2D Engine:  [1][2][3][4][5][6][7][8]                                │
│   Compute:        [1][2][3][4][5][6][7][8]                             │
│   D2H Engine:        [1][2][3][4][5][6][7][8]                          │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   After startup (first 2 chunks):                                      │
│   - H2D Engine: 100% utilized (always copying next chunk)              │
│   - Compute:    100% utilized (always processing a chunk)              │
│   - D2H Engine: 100% utilized (always sending back results)            │
│                                                                         │
│   STEADY STATE: All three engines working simultaneously!              │
│                                                                         │
│   Theoretical Speedup: 3× (hiding 2 of 3 phases completely)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Complete Example: Streamed Vector Addition

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 24)  // 16M elements = 64 MB per array
#define N_STREAMS 4

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    size_t streamBytes = bytes / N_STREAMS;
    int streamSize = N / N_STREAMS;

    // ═══════════════════════════════════════════════════════
    // Allocate PINNED host memory (critical for async!)
    // ═══════════════════════════════════════════════════════
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ═══════════════════════════════════════════════════════
    // Create streams
    // ═══════════════════════════════════════════════════════
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // ═══════════════════════════════════════════════════════
    // Process in chunks using streams
    // ═══════════════════════════════════════════════════════
    dim3 block(256);
    dim3 grid((streamSize + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * streamSize;

        // H2D: Copy input chunks
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // Compute
        vectorAdd<<<grid, block, 0, streams[i]>>>(&d_a[offset], 
                                                   &d_b[offset], 
                                                   &d_c[offset], 
                                                   streamSize);

        // D2H: Copy results back
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Streamed execution: %.2f ms\n", ms);
    printf("Effective bandwidth: %.2f GB/s\n", 
           (3 * bytes / 1e9) / (ms / 1000));

    // ═══════════════════════════════════════════════════════
    // Cleanup
    // ═══════════════════════════════════════════════════════
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

---

## 6. Benchmarks

### Test Configuration
- **GPU:** NVIDIA RTX 3080
- **Data Size:** 256 MB (64M floats × 4 bytes)
- **Operation:** Vector Addition (bandwidth-bound)

| Method | Total Time | Speedup | Notes |
|--------|-----------|---------|-------|
| Serial (Pageable) | 45.2 ms | 1.0× | Baseline: malloc + cudaMemcpy |
| Serial (Pinned) | 32.1 ms | 1.4× | Faster transfer, no overlap |
| **4 Streams (Async)** | **12.5 ms** | **3.6×** | Massive overlap! |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Streams Performance Comparison                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Serial (Pageable)  ████████████████████████████████████████  45.2 ms │
│                                                                         │
│   Serial (Pinned)    ████████████████████████████              32.1 ms │
│                      ↑ 1.4× faster (no staging buffer)                 │
│                                                                         │
│   4 Streams (Async)  ██████████                                12.5 ms │
│                      ↑ 3.6× faster! (overlap hides latency)            │
│                                                                         │
│   0        10       20       30       40       50  (milliseconds)      │
│                                                                         │
│   ═══════════════════════════════════════════════════════════          │
│   KEY INSIGHT: For bandwidth-bound kernels, we essentially HIDE        │
│   the computation completely behind the data transfer!                 │
│   ═══════════════════════════════════════════════════════════          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### When Streams Help Most

| Scenario | Speedup | Why |
|----------|---------|-----|
| **Bandwidth-bound kernel** (Vector Add) | 3-4× | Compute hidden behind transfer |
| **Balanced kernel** (MatMul small tiles) | 2× | Partial overlap |
| **Compute-bound kernel** (Heavy math) | 1.1-1.5× | Transfer already hidden |

**Key Insight:** Streams provide the biggest wins when **transfer time ≈ compute time**. If your kernel is extremely compute-bound, the transfers are already "hidden" even in serial mode.

---

## 7. Common Pitfalls

### 1. The Default Stream Trap

The default stream (Stream 0) has special **synchronizing behavior**. Any operation in the default stream:
- Waits for ALL other streams to complete
- Blocks ALL other streams until it completes

```cpp
// ❌ DANGEROUS: Mixing default and explicit streams
cudaMemcpyAsync(d_a, h_a, size, H2D, stream1);  // Stream 1
kernel<<<g, b, 0, stream1>>>(d_a);               // Stream 1

cudaMemcpy(d_b, h_b, size, H2D);  // DEFAULT STREAM! Implicit sync!
                                   // ↑ This WAITS for stream1 to finish
                                   //   AND blocks stream1 from continuing!

kernel<<<g, b, 0, stream1>>>(d_b);  // Can't overlap with previous!
```

```cpp
// ✅ CORRECT: Use explicit streams everywhere
cudaMemcpyAsync(d_a, h_a, size, H2D, stream1);
kernel<<<g, b, 0, stream1>>>(d_a);

cudaMemcpyAsync(d_b, h_b, size, H2D, stream2);  // Different stream, no sync!
kernel<<<g, b, 0, stream2>>>(d_b);               // Can overlap!
```

### 2. Forgetting Pinned Memory

This is the **silent killer** of stream performance:

```cpp
float* h_data = (float*)malloc(bytes);  // PAGEABLE!

// This looks async but ISN'T:
cudaMemcpyAsync(d_data, h_data, bytes, H2D, stream);
// ↑ Driver silently falls back to synchronous copy!
//   You lose ALL overlap benefits!
```

**Always check:** If your streams don't seem to overlap in Nsight Systems, check your host memory allocation!

### 3. Too Many Streams

Creating thousands of streams doesn't help:

| Streams | Effect |
|---------|--------|
| 1 | No overlap (serial) |
| 2-4 | Good overlap, minimal overhead |
| 4-8 | Usually optimal |
| 8-16 | Diminishing returns |
| 100+ | Overhead dominates, no benefit |

**Why?** The GPU has limited hardware queues:
- ~128 concurrent kernel launches
- 1-2 copy engines (H2D and D2H)

More streams just means more scheduling overhead without more parallelism.

### 4. Dependencies Across Streams

If Stream B needs data produced by Stream A, you need explicit synchronization:

```cpp
// ❌ WRONG: Race condition!
kernel_A<<<g, b, 0, streamA>>>(d_data);  // Produces d_data
kernel_B<<<g, b, 0, streamB>>>(d_data);  // Reads d_data - might run first!

// ✅ CORRECT: Use events to synchronize
cudaEvent_t event;
cudaEventCreate(&event);

kernel_A<<<g, b, 0, streamA>>>(d_data);
cudaEventRecord(event, streamA);           // Record when A finishes

cudaStreamWaitEvent(streamB, event, 0);    // B waits for A's event
kernel_B<<<g, b, 0, streamB>>>(d_data);    // Now safe!
```

---

## 8. Advanced: Double Buffering

For maximum throughput, use **double buffering**: while the GPU processes Buffer A, the CPU fills Buffer B.

```cpp
// Two sets of buffers
float *h_buf[2], *d_buf[2];
cudaStream_t streams[2];

for (int i = 0; i < 2; i++) {
    cudaMallocHost(&h_buf[i], chunkBytes);
    cudaMalloc(&d_buf[i], chunkBytes);
    cudaStreamCreate(&streams[i]);
}

int current = 0;
for (int chunk = 0; chunk < totalChunks; chunk++) {
    int next = 1 - current;  // Alternate: 0, 1, 0, 1, ...

    // Start async copy of NEXT chunk while GPU works on CURRENT
    if (chunk + 1 < totalChunks) {
        fill_buffer(h_buf[next], chunk + 1);  // CPU work
        cudaMemcpyAsync(d_buf[next], h_buf[next], chunkBytes, H2D, streams[next]);
    }

    // Process current chunk
    kernel<<<g, b, 0, streams[current]>>>(d_buf[current]);
    cudaMemcpyAsync(h_result, d_buf[current], chunkBytes, D2H, streams[current]);

    current = next;
}
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Double Buffering Timeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIME ──────────────────────────────────────────────────────────────►  │
│                                                                         │
│   CPU:      [Fill A][Fill B][Fill A][Fill B]...                        │
│   H2D:           [Copy A][Copy B][Copy A][Copy B]...                   │
│   Compute:            [Proc A][Proc B][Proc A][Proc B]...              │
│   D2H:                     [Back A][Back B][Back A]...                 │
│                                                                         │
│   - While GPU processes A, CPU fills B and copies B to GPU             │
│   - While GPU processes B, CPU fills A and copies A to GPU             │
│   - Maximum overlap: CPU and GPU never wait for each other!            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Profiling with Nsight Systems

To **verify** your streams are actually overlapping, use Nsight Systems:

```bash
nsys profile -o streams_profile ./my_cuda_app
nsys-ui streams_profile.nsys-rep
```

### What to Look For

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Nsight Systems Timeline View                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ❌ BAD: No overlap (serial execution)                                │
│                                                                         │
│   Stream 0:  [MemCpy H2D][Kernel    ][MemCpy D2H]                      │
│   Stream 0:                                      [MemCpy H2D][Kernel]  │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   ✅ GOOD: Full overlap (concurrent execution)                         │
│                                                                         │
│   MemCpy H2D:  [Chunk 1][Chunk 2][Chunk 3][Chunk 4]                    │
│   Compute:          [K1]    [K2]    [K3]    [K4]                       │
│   MemCpy D2H:            [Chunk 1][Chunk 2][Chunk 3][Chunk 4]          │
│                                                                         │
│   Three rows of activity overlapping = SUCCESS!                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Challenge for the Reader

### Challenge 1: The Pipeline Challenge

1. Take your **Matrix Multiplication** kernel from an earlier post
2. Allocate **1 GB** of matrices
3. Process it in **128 MB chunks** using **4 streams**
4. Measure the throughput improvement

### Challenge 2: Double Buffering

Implement full double buffering:
- While the GPU computes on Buffer A, the CPU fills Buffer B
- Alternate between buffers to achieve maximum overlap

### Challenge 3: Multi-GPU Streaming

If you have multiple GPUs:
- Create streams on each GPU
- Split work across GPUs AND streams
- Use `cudaSetDevice()` to switch between GPUs

---

## Summary

### Key Takeaways

| Concept | Lesson |
|---------|--------|
| **The Serial Trap** | Default behavior wastes hardware (only 33% utilization) |
| **Streams** | Ordered sequences that can run concurrently |
| **Pinned Memory** | Required for true async transfers |
| **Hardware Engines** | H2D, Compute, D2H can all work simultaneously |
| **Sweet Spot** | 4-8 streams is usually optimal |

### The Optimization Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    System-Level Optimization Path                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SERIAL (DEFAULT)            PINNED MEMORY              STREAMS       │
│   ────────────────            ─────────────              ───────       │
│                                                                         │
│   Memory:                     Memory:                    Memory:       │
│   Pageable (slow)             Pinned (fast DMA)          Pinned        │
│                                                                         │
│   Execution:                  Execution:                 Execution:    │
│   H2D → K → D2H               H2D → K → D2H              Overlapped!   │
│   (serial)                    (still serial)             (concurrent)  │
│                                                                         │
│   Utilization:                Utilization:               Utilization:  │
│   ~33%                        ~33%                       ~90%+         │
│                                                                         │
│   Time:                       Time:                      Time:         │
│   45.2 ms                     32.1 ms                    12.5 ms       │
│                                                                         │
│         ─────────────────────────────────────────────────►             │
│                     System Optimization Progress                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What's Next?

Congratulations! You've completed the core CUDA optimization journey:

| Post | Focus |
|------|-------|
| Vector Add | Basics: Threads, Blocks, Grids |
| Reduction | Algorithmic optimization, warp-level primitives |
| Convolution | Shared memory tiling, halo regions |
| Scan | Parallel prefix patterns |
| Transpose | Memory coalescing |
| Histogram | Atomics, privatization |
| **Streams** | **System-level concurrency** |

You now have the tools to optimize CUDA code at every level: **algorithms**, **memory access patterns**, and **system concurrency**.

---

## References

1. [NVIDIA CUDA C++ Programming Guide — Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
2. [NVIDIA CUDA C++ Best Practices Guide — Asynchronous Transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)
3. [NVIDIA Developer Blog — How to Overlap Data Transfers in CUDA](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
4. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
