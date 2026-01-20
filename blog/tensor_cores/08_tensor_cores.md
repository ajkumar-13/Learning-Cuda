# Tensor Cores & Mixed Precision

> **Breaking the Speed Limit with WMMA and FP16**

---

## 1. Introduction

### The Mystery of Missing TFLOPS

In our MatMul post, we hit ~300 GFLOPS using `float` (FP32). We were proud. We used shared memory, tiling, and coalesced access.

But then we looked at the spec sheet:

| GPU | Claimed Performance |
|-----|---------------------|
| RTX 3080 | 29.8 TFLOPS (FP32) |
| RTX 3090 | 35.6 TFLOPS (FP32) |
| RTX 3090 | **71 TFLOPS (FP16 Tensor)** |
| RTX 4090 | **165 TFLOPS (FP16 Tensor)** |

Wait. **71 TFLOPS?** That's 200× what we achieved. Where is this performance hiding?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Performance Gap                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Our MatMul (FP32, Shared Memory):     ~300 GFLOPS                    │
│   ████                                                                  │
│                                                                         │
│   Theoretical FP32 Peak:                ~30 TFLOPS                     │
│   ████████████████████████████████████████                             │
│                                                                         │
│   Tensor Core FP16 Peak:                ~70 TFLOPS                     │
│   ████████████████████████████████████████████████████████████████████ │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Question: How do we access this 200× speedup?                        │
│   Answer:   TENSOR CORES + HALF PRECISION (FP16)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Answer: Tensor Cores

**Tensor Cores** are specialized hardware units that exist alongside regular CUDA cores. They're designed for one purpose: **matrix multiply-accumulate**.

But to use them, we must leave the comfortable world of `float` and enter the fast but dangerous world of:
- `__half` (FP16 — 16-bit floating point)
- `nvcuda::wmma` (Warp Matrix Multiply-Accumulate API)

---

## 2. The Hardware: What is a Tensor Core?

### CUDA Cores vs. Tensor Cores

**CUDA Cores** (the ones we've been using) work like this:
- 1 thread = 1 scalar operation
- Each thread does: `c = a * b + c` (one multiply-add)

**Tensor Cores** work differently:
- 1 warp (32 threads) = 1 **matrix** operation
- The entire warp collaborates to compute: $D = A \times B + C$

Where $A$, $B$, $C$, $D$ are **matrices**, not scalars!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CUDA Core vs. Tensor Core                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CUDA CORE (Standard)                                                  │
│   ════════════════════                                                  │
│                                                                         │
│   Thread 0:  c[0] = a[0] * b[0] + c[0]     (1 FMA)                     │
│   Thread 1:  c[1] = a[1] * b[1] + c[1]     (1 FMA)                     │
│   Thread 2:  c[2] = a[2] * b[2] + c[2]     (1 FMA)                     │
│   ...                                                                   │
│   Thread 31: c[31] = a[31] * b[31] + c[31] (1 FMA)                     │
│                                                                         │
│   Total: 32 FMAs per warp per cycle                                    │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   TENSOR CORE (Matrix)                                                  │
│   ════════════════════                                                  │
│                                                                         │
│   Entire Warp (32 threads) collaborates:                               │
│                                                                         │
│       ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐           │
│       │  A      │ × │  B      │ + │  C      │ = │  D      │           │
│       │ 16×16   │   │ 16×16   │   │ 16×16   │   │ 16×16   │           │
│       │  FP16   │   │  FP16   │   │  FP32   │   │  FP32   │           │
│       └─────────┘   └─────────┘   └─────────┘   └─────────┘           │
│                                                                         │
│   Total: 16 × 16 × 16 = 4096 FMAs per warp per instruction!            │
│                                                                         │
│   That's 128× more operations per instruction!                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Tensor Core Operation

The fundamental operation is:

$$D = A \times B + C$$

Where:
- **A**: $M \times K$ matrix (FP16)
- **B**: $K \times N$ matrix (FP16)
- **C**: $M \times N$ matrix (FP16 or FP32, accumulator)
- **D**: $M \times N$ matrix (result, same type as C)

The supported tile sizes depend on your GPU architecture:

| Architecture | Supported Shapes (M×N×K) |
|--------------|-------------------------|
| Volta (SM 7.0) | 16×16×16 |
| Turing (SM 7.5) | 16×16×16, 32×8×16, 8×32×16 |
| Ampere (SM 8.0) | 16×16×16, 32×8×16, 8×32×16 |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Tensor Core Matrix Operation                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│              K=16                    N=16                               │
│          ┌─────────┐             ┌─────────┐                           │
│          │         │             │         │                           │
│    M=16  │    A    │      K=16   │    B    │                           │
│          │  (FP16) │             │  (FP16) │                           │
│          │         │             │         │                           │
│          └─────────┘             └─────────┘                           │
│                                                                         │
│                        ×                                                │
│                                                                         │
│          ┌─────────┐             ┌─────────┐                           │
│          │         │      +      │         │                           │
│    M=16  │    C    │      =      │    D    │      N=16                 │
│          │ (FP32)  │             │ (FP32)  │                           │
│          │         │             │         │                           │
│          └─────────┘             └─────────┘                           │
│                                                                         │
│   Each Tensor Core instruction computes:                               │
│   - 16 × 16 × 16 = 4,096 multiply-accumulate operations               │
│   - In a SINGLE warp instruction                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mixed Precision: The Secret Sauce

Notice something important: **A and B are FP16, but C and D can be FP32**.

This is called **Mixed Precision**:
- **Compute** in FP16 (fast, uses Tensor Cores)
- **Accumulate** in FP32 (accurate, prevents overflow)

Why does this matter?

| Precision | Range | Precision |
|-----------|-------|-----------|
| FP32 | ±3.4×10³⁸ | ~7 decimal digits |
| FP16 | ±65,504 | ~3 decimal digits |

FP16 is **fast but limited**. If you multiply many FP16 numbers, small errors accumulate and overflow can occur. By accumulating in FP32, you get:
- **Speed** of FP16 multiplication
- **Accuracy** of FP32 accumulation

---

## 3. The Programming Model: WMMA API

### The Challenge

You can't just write `c = a * b` and expect Tensor Cores. The hardware needs:
1. Data in specific **matrix layouts**
2. Operations on entire **matrix tiles**
3. **Warp-level** cooperation (all 32 threads work together)

NVIDIA provides the **WMMA (Warp Matrix Multiply-Accumulate)** API to abstract this complexity.

### Fragments: Opaque Matrix Containers

You don't access Tensor Core registers directly. Instead, you work with **fragments** — opaque containers that hold matrix tiles distributed across the warp.

```cpp
#include <mma.h>
using namespace nvcuda;

// Declare fragments for 16×16×16 operation
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WMMA Fragment Types                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Fragment Type        Purpose              Size        Data Type      │
│   ─────────────────────────────────────────────────────────────────    │
│   wmma::matrix_a       Left operand (A)     M × K       half (FP16)    │
│   wmma::matrix_b       Right operand (B)    K × N       half (FP16)    │
│   wmma::accumulator    Accumulator (C/D)    M × N       float (FP32)   │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   IMPORTANT: Fragments are DISTRIBUTED across the warp!                │
│                                                                         │
│   Thread 0:  [elem 0, elem 32, elem 64, ...]                          │
│   Thread 1:  [elem 1, elem 33, elem 65, ...]                          │
│   Thread 2:  [elem 2, elem 34, elem 66, ...]                          │
│   ...                                                                   │
│   Thread 31: [elem 31, elem 63, elem 95, ...]                         │
│                                                                         │
│   You NEVER access individual elements directly!                       │
│   The hardware handles the distribution.                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The WMMA Lifecycle

```cpp
// 1. LOAD: Move data from memory into fragments
wmma::load_matrix_sync(a_frag, a_ptr, stride);  // Load A tile
wmma::load_matrix_sync(b_frag, b_ptr, stride);  // Load B tile
wmma::fill_fragment(c_frag, 0.0f);              // Initialize C to zero

// 2. COMPUTE: The magic line — one instruction, 4096 FMAs!
wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

// 3. STORE: Move results back to memory
wmma::store_matrix_sync(d_ptr, d_frag, stride, wmma::mem_row_major);
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WMMA Operation Lifecycle                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GLOBAL MEMORY                    REGISTERS (Fragments)               │
│   ══════════════                   ══════════════════════              │
│                                                                         │
│   ┌─────────────┐                  ┌─────────────┐                     │
│   │ Matrix A    │ ──load_sync──►   │  a_frag     │                     │
│   │ (FP16)      │                  │             │                     │
│   └─────────────┘                  └──────┬──────┘                     │
│                                           │                             │
│   ┌─────────────┐                  ┌──────┴──────┐                     │
│   │ Matrix B    │ ──load_sync──►   │  b_frag     │                     │
│   │ (FP16)      │                  │             │                     │
│   └─────────────┘                  └──────┬──────┘                     │
│                                           │                             │
│   ┌─────────────┐                  ┌──────┴──────┐     ┌─────────────┐ │
│   │ Matrix C    │ ──load_sync──►   │  c_frag     │──►  │  mma_sync   │ │
│   │ (FP32)      │  (or fill=0)     │             │     │  D=A×B+C    │ │
│   └─────────────┘                  └─────────────┘     └──────┬──────┘ │
│                                                               │        │
│   ┌─────────────┐                  ┌─────────────┐            │        │
│   │ Matrix D    │ ◄──store_sync──  │  d_frag     │ ◄──────────┘        │
│   │ (FP32)      │                  │             │                     │
│   └─────────────┘                  └─────────────┘                     │
│                                                                         │
│   KEY: All operations are WARP-SYNCHRONOUS (_sync suffix)              │
│   All 32 threads must execute the same instruction together.           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layout Matters!

For Tensor Cores, matrix layout is **critical**:

```cpp
// Matrix A: Row-major layout
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;

// Matrix B: Column-major layout (common for optimal memory access)
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
```

Why does layout matter?
- Tensor Cores expect specific memory access patterns
- Mismatched layouts require expensive transposes
- For MatMul: A row-major × B col-major is often optimal

---

## 4. The Complete Tensor Core MatMul

### Converting to FP16

First, we need half-precision data:

```cpp
#include <cuda_fp16.h>

// Convert FP32 array to FP16
__global__ void convertFP32toFP16(half* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// Convert FP16 array to FP32
__global__ void convertFP16toFP32(float* out, const half* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}
```

### The Tensor Core Kernel

```cpp
#include <mma.h>
using namespace nvcuda;

// Tile dimensions for Tensor Cores
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void tensorCoreMatMul(half* A, half* B, float* C, 
                                  int M, int N, int K) {
    // Calculate which warp we are
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += WMMA_K) {
        // Calculate pointers to A and B tiles
        half* a_tile = A + warpM * WMMA_M * K + k;
        half* b_tile = B + k * N + warpN * WMMA_N;

        // Load tiles into fragments
        wmma::load_matrix_sync(a_frag, a_tile, K);    // stride = K (row-major A)
        wmma::load_matrix_sync(b_frag, b_tile, N);    // stride = N (col-major B)

        // Tensor Core matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    float* c_tile = C + warpM * WMMA_M * N + warpN * WMMA_N;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}
```

### Launching the Kernel

```cpp
// Matrix dimensions (must be multiples of 16 for simplicity)
int M = 4096, N = 4096, K = 4096;

// Each warp handles a 16×16 output tile
// We need M/16 × N/16 warps total
dim3 block(128, 4);  // 128 threads = 4 warps per block in X, 4 in Y
dim3 grid((M + (WMMA_M * 4) - 1) / (WMMA_M * 4),
          (N + (WMMA_N * 4) - 1) / (WMMA_N * 4));

tensorCoreMatMul<<<grid, block>>>(d_A_half, d_B_half, d_C, M, N, K);
```

---

## 5. Optimized Version: Shared Memory + Tensor Cores

The basic version works, but we can do better with shared memory:

```cpp
#define BLOCK_SIZE 128  // Threads per block
#define WARPS_PER_BLOCK 4
#define TILE_M (WMMA_M * WARPS_PER_BLOCK)  // 64
#define TILE_N (WMMA_N * WARPS_PER_BLOCK)  // 64
#define TILE_K 32

__global__ void tensorCoreMatMulOptimized(half* A, half* B, float* C,
                                           int M, int N, int K) {
    // Shared memory for tiles
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N];

    // Warp and lane identification
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // Which WMMA tile this warp handles
    int warpRow = warpId / 2;  // 0 or 1
    int warpCol = warpId % 2;  // 0 or 1

    // Global tile position
    int tileRow = blockIdx.y * TILE_M;
    int tileCol = blockIdx.x * TILE_N;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Main loop over K
    for (int k = 0; k < K; k += TILE_K) {
        // Cooperatively load A tile into shared memory
        // ... (loading code similar to regular tiled MatMul)

        // Cooperatively load B tile into shared memory
        // ... (loading code)

        __syncthreads();

        // Process the shared memory tile with Tensor Cores
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
            // Load from shared memory to fragments
            wmma::load_matrix_sync(a_frag[0], &As[warpRow * WMMA_M * 2][kk], TILE_K);
            wmma::load_matrix_sync(a_frag[1], &As[warpRow * WMMA_M * 2 + WMMA_M][kk], TILE_K);
            wmma::load_matrix_sync(b_frag[0], &Bs[kk][warpCol * WMMA_N * 2], TILE_N);
            wmma::load_matrix_sync(b_frag[1], &Bs[kk][warpCol * WMMA_N * 2 + WMMA_N], TILE_N);

            // 2×2 = 4 WMMA operations per warp
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float* c_ptr = C + (tileRow + warpRow * WMMA_M * 2 + i * WMMA_M) * N 
                             + (tileCol + warpCol * WMMA_N * 2 + j * WMMA_N);
            wmma::store_matrix_sync(c_ptr, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}
```

---

## 6. Benchmarks

### Test Configuration
- **GPU:** NVIDIA RTX 3080
- **Matrix Size:** 4096 × 4096
- **Baseline:** Optimized FP32 SGEMM with shared memory

| Implementation | Time (ms) | TFLOPS | Speedup |
|----------------|-----------|--------|---------|
| Naive FP32 | 45.2 | 3.0 | 1.0× |
| Tiled FP32 (Shared Memory) | 12.8 | 10.7 | 3.5× |
| cuBLAS SGEMM (FP32) | 4.2 | 32.7 | 10.8× |
| **Tensor Core (Basic)** | **2.1** | **65.4** | **21.5×** |
| **Tensor Core (Optimized)** | **1.4** | **98.1** | **32.3×** |
| cuBLAS HGEMM (FP16 TC) | 1.1 | 124.9 | 41.1× |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Performance Comparison (4096×4096)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Naive FP32             ████████████████████████████████████  45.2 ms │
│                                                                         │
│   Tiled FP32             █████████████                         12.8 ms │
│                                                                         │
│   cuBLAS SGEMM           ████                                   4.2 ms │
│                                                                         │
│   Tensor Core (Basic)    ██                                     2.1 ms │
│                                                                         │
│   Tensor Core (Optimized)█                                      1.4 ms │
│                                                                         │
│   cuBLAS HGEMM (TC)      █                                      1.1 ms │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   KEY INSIGHT: Tensor Cores provide 20-40× speedup over naive FP32!   │
│                                                                         │
│   But there's a catch: you sacrifice some precision (FP16 inputs).    │
│   Mixed precision (FP16 compute + FP32 accumulate) is the sweet spot. │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### When to Use Tensor Cores

| Use Case | Recommendation |
|----------|----------------|
| **Deep Learning Training** | ✅ Perfect! Mixed precision is standard |
| **Deep Learning Inference** | ✅ FP16 often sufficient |
| **Scientific Computing** | ⚠️ Check precision requirements |
| **Financial/HPC** | ❌ Usually need FP64 |
| **Graphics/Gaming** | ✅ DLSS uses Tensor Cores |

---

## 7. The Precision Trade-off

### Understanding FP16 Limitations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FP32 vs FP16 Precision                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   FP32 (32-bit float)                                                  │
│   ═══════════════════                                                   │
│   Sign: 1 bit | Exponent: 8 bits | Mantissa: 23 bits                   │
│   Range: ±3.4 × 10³⁸                                                   │
│   Precision: ~7 decimal digits                                          │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   FP16 (16-bit half)                                                   │
│   ═══════════════════                                                   │
│   Sign: 1 bit | Exponent: 5 bits | Mantissa: 10 bits                   │
│   Range: ±65,504                                                       │
│   Precision: ~3 decimal digits                                          │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   DANGER ZONES:                                                         │
│   • Numbers > 65,504 → OVERFLOW (Inf)                                  │
│   • Numbers < 2⁻¹⁴ ≈ 0.00006 → UNDERFLOW (becomes 0)                  │
│   • Precision loss in accumulation (summing many small values)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mixed Precision: Best of Both Worlds

The solution is **Mixed Precision**:

```cpp
// Tensor Core operation with mixed precision:
// A (FP16) × B (FP16) + C (FP32) = D (FP32)

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, ...> a_frag;      // FP16
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, ...> b_frag;      // FP16
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;       // FP32!

// The multiply happens in FP16 (fast!)
// The accumulate happens in FP32 (accurate!)
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Mixed Precision Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INPUT (FP32)     CONVERT        COMPUTE         ACCUMULATE   OUTPUT  │
│   ════════════     ═══════        ═══════         ══════════   ══════  │
│                                                                         │
│   ┌─────────┐     ┌───────┐      ┌───────┐       ┌─────────┐          │
│   │ A (FP32)│ ──► │FP32→  │ ──►  │       │       │         │          │
│   │         │     │FP16   │      │Tensor │       │  FP32   │  ┌─────┐ │
│   └─────────┘     └───────┘      │ Core  │  ──►  │  Accum  │──│ C   │ │
│                                  │       │       │         │  │FP32 │ │
│   ┌─────────┐     ┌───────┐      │ FP16  │       └─────────┘  └─────┘ │
│   │ B (FP32)│ ──► │FP32→  │ ──►  │  ×    │                            │
│   │         │     │FP16   │      │ FP16  │                            │
│   └─────────┘     └───────┘      └───────┘                            │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Speed:      Matches pure FP16 (Tensor Cores)                         │
│   Accuracy:   Nearly matches FP32 (FP32 accumulator)                   │
│                                                                         │
│   This is how PyTorch Automatic Mixed Precision (AMP) works!           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Common Pitfalls

### 1. Alignment Requirements

Tensor Cores require **256-byte alignment** for best performance:

```cpp
// ❌ Might be slow or fail
half* A;
cudaMalloc(&A, M * K * sizeof(half));

// ✅ Ensure alignment
half* A;
cudaMalloc(&A, ((M * K * sizeof(half) + 255) / 256) * 256);
```

### 2. Matrix Dimension Requirements

Matrix dimensions must be multiples of the WMMA tile size:

```cpp
// ❌ Will crash or produce wrong results
int M = 1000, N = 1000, K = 1000;  // Not multiples of 16!

// ✅ Pad to multiples of 16
int M_padded = ((M + 15) / 16) * 16;  // 1008
int N_padded = ((N + 15) / 16) * 16;
int K_padded = ((K + 15) / 16) * 16;
```

### 3. Forgetting Warp-Level Thinking

WMMA operations are **warp-synchronous**. All 32 threads must participate:

```cpp
// ❌ WRONG: Divergent threads
if (threadIdx.x < 16) {
    wmma::load_matrix_sync(a_frag, ...);  // Only 16 threads = CRASH!
}

// ✅ CORRECT: All threads in warp participate
wmma::load_matrix_sync(a_frag, ...);  // All 32 threads
```

### 4. Wrong Layout Assumptions

If you transpose incorrectly, results will be garbage:

```cpp
// A is stored row-major, B is stored column-major
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

// Make sure your data actually matches these layouts!
```

---

## 9. Challenge

### Challenge 1: Full Mixed Precision Pipeline

Implement a complete mixed precision MatMul:
1. **Inputs:** FP32 matrices A and B
2. **Convert:** FP32 → FP16 on the GPU
3. **Compute:** Use Tensor Cores with FP32 accumulator
4. **Output:** FP32 result matrix C

Measure the total time including conversion overhead. Is it still worth it?

### Challenge 2: Compare Precision

1. Compute the same MatMul using:
   - Pure FP32 (cuBLAS SGEMM)
   - Pure FP16 (Tensor Cores, FP16 accumulator)
   - Mixed (Tensor Cores, FP32 accumulator)
2. Calculate the **maximum absolute error** between each method
3. At what matrix size does FP16 accumulation start showing significant errors?

### Challenge 3: Batched MatMul

Implement batched matrix multiplication using Tensor Cores:
- Process 1000 small matrices (64×64 each) in parallel
- Compare against cuBLAS `cublasHgemmBatched`

---

## Summary

### Key Takeaways

| Concept | Lesson |
|---------|--------|
| **Tensor Cores** | Warp-level matrix units: 1 instruction = 4096 FMAs |
| **WMMA API** | Fragments + load/mma/store = Tensor Core access |
| **FP16** | Fast but limited range (±65,504) and precision (~3 digits) |
| **Mixed Precision** | FP16 compute + FP32 accumulate = speed + accuracy |
| **Alignment** | 256-byte alignment and 16× dimensions required |

### The Performance Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CUDA MatMul Evolution                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Level 1: Naive FP32                     3 TFLOPS      (Baseline)     │
│                                                                         │
│   Level 2: Tiled FP32 (Shared Memory)    11 TFLOPS      (3.5×)         │
│                                                                         │
│   Level 3: cuBLAS SGEMM                  33 TFLOPS      (11×)          │
│                                                                         │
│   Level 4: Tensor Core (Basic)           65 TFLOPS      (22×)          │
│                                                                         │
│   Level 5: Tensor Core (Optimized)       98 TFLOPS      (33×)          │
│                                                                         │
│   Level 6: cuBLAS HGEMM (Tensor Core)   125 TFLOPS      (42×)          │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   You now have the knowledge to reach Level 5!                         │
│   Level 6 requires additional tricks (software pipelining, etc.)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What's Next?

Congratulations! You've now mastered the full CUDA optimization stack:

| Post | Level |
|------|-------|
| Vector Add | Basics |
| Reduction | Algorithm Optimization |
| Convolution | Memory Hierarchy |
| Scan | Parallel Patterns |
| Transpose | Memory Coalescing |
| Histogram | Atomics |
| Streams | System Concurrency |
| **Tensor Cores** | **Hardware Specialization** |

You've gone from "GPU as a calculator" to "GPU as a high-performance computing fabric." The skills you've learned here are directly applicable to:
- **Deep Learning Frameworks** (PyTorch, TensorFlow)
- **Scientific Computing** (cuBLAS, cuDNN)
- **Graphics** (DLSS, real-time ray tracing)

---

## References

1. [NVIDIA CUDA C++ Programming Guide — WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
2. [NVIDIA Developer Blog — Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
3. [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
4. [Mixed Precision Training Paper (Micikevicius et al.)](https://arxiv.org/abs/1710.03740)
5. [CUTLASS: CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)
