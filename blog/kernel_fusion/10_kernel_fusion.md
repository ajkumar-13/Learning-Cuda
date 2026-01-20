# Kernel Fusion

> **The fastest kernel is the one you don't launch**

---

## 1. Introduction

### The Hidden Cost of Clean Code

PyTorch code looks beautiful:

```python
x = F.relu(F.conv2d(x, weight))
```

One line. Elegant. Easy to read.

But on the GPU, this is a **disaster**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    What Actually Happens                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Python: x = relu(conv(x))                                            │
│                                                                         │
│   GPU Reality:                                                          │
│   ════════════                                                          │
│                                                                         │
│   ┌─────────┐      ┌─────────────────┐      ┌─────────┐                │
│   │   X     │─────►│   Conv Kernel   │─────►│   Y     │                │
│   │  (HBM)  │ READ │   (Compute)     │WRITE │  (HBM)  │                │
│   └─────────┘      └─────────────────┘      └────┬────┘                │
│                                                  │                      │
│                    Kernel launch overhead        │                      │
│                    + synchronization             │                      │
│                                                  ▼                      │
│   ┌─────────┐      ┌─────────────────┐      ┌─────────┐                │
│   │   Y     │─────►│   ReLU Kernel   │─────►│   Z     │                │
│   │  (HBM)  │ READ │   (Compute)     │WRITE │  (HBM)  │                │
│   └─────────┘      └─────────────────┘      └─────────┘                │
│                                                                         │
│   COST:                                                                 │
│   • 2 kernel launches (~5-10μs each)                                   │
│   • 2 HBM reads (X for conv, Y for relu)                               │
│   • 2 HBM writes (Y from conv, Z from relu)                            │
│   • Y exists only to be immediately read and discarded!                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Memory Traffic Problem

Let's do the math. For a tensor with 1 million FP32 elements:

| Operation | Reads | Writes | Total Traffic |
|-----------|-------|--------|---------------|
| Conv (separate) | 4 MB | 4 MB | 8 MB |
| ReLU (separate) | 4 MB | 4 MB | 8 MB |
| **Total** | **8 MB** | **8 MB** | **16 MB** |

But `Y` is just an intermediate! We read and write it for **no reason**.

### The Solution: Kernel Fusion

What if we did both operations in **one kernel**?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fused Kernel                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐      ┌─────────────────────────┐      ┌─────────┐        │
│   │   X     │─────►│   Fused Conv+ReLU       │─────►│   Z     │        │
│   │  (HBM)  │ READ │                         │WRITE │  (HBM)  │        │
│   └─────────┘      │  ┌─────────────────┐    │      └─────────┘        │
│                    │  │ Conv result     │    │                         │
│                    │  │ (in registers!) │    │                         │
│                    │  └────────┬────────┘    │                         │
│                    │           │             │                         │
│                    │           ▼             │                         │
│                    │  ┌─────────────────┐    │                         │
│                    │  │ ReLU            │    │                         │
│                    │  │ (in registers!) │    │                         │
│                    │  └─────────────────┘    │                         │
│                    └─────────────────────────┘                         │
│                                                                         │
│   COST:                                                                 │
│   • 1 kernel launch                                                    │
│   • 1 HBM read (X only)                                                │
│   • 1 HBM write (Z only)                                               │
│   • Y never touches HBM — it lives and dies in registers!             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Operation | Reads | Writes | Total Traffic |
|-----------|-------|--------|---------------|
| Fused Conv+ReLU | 4 MB | 4 MB | 8 MB |

**50% less memory traffic!** And that's just for two operations.

---

## 2. Compute-Bound vs Memory-Bound

### The Roofline Model Refresher

Every kernel is limited by one of two things:

| Bottleneck | Limited By | Characteristic |
|------------|-----------|----------------|
| **Compute-bound** | FLOPS | High arithmetic intensity |
| **Memory-bound** | Bandwidth | Low arithmetic intensity |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Arithmetic Intensity Analysis                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Arithmetic Intensity = FLOPs / Bytes Moved                           │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                                                               │    │
│   │   Operation          FLOPs/Element    Bytes/Element    AI     │    │
│   │   ─────────────────────────────────────────────────────────  │    │
│   │   ReLU               1 (compare)      8 (read+write)   0.125 │    │
│   │   Vector Add         1                8                0.125 │    │
│   │   LayerNorm          ~10              8                1.25  │    │
│   │   MatMul (large)     ~2N              ~2               ~N    │    │
│   │   Convolution        ~2K²C            ~2               high  │    │
│   │                                                               │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   A100 GPU:                                                            │
│   • Peak Compute: 312 TFLOPS (FP16)                                   │
│   • Peak Bandwidth: 2 TB/s                                             │
│   • Balance Point: 312/2 = 156 FLOPs/byte                             │
│                                                                         │
│   If AI < 156: Memory-bound (most elementwise ops!)                    │
│   If AI > 156: Compute-bound (matmul, conv)                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### ReLU: The Ultimate Memory-Bound Operation

ReLU does **one comparison** per element:

```cpp
output[i] = input[i] > 0 ? input[i] : 0;
```

- **FLOPs**: 1 (one comparison)
- **Memory**: 8 bytes (4-byte read + 4-byte write)
- **Arithmetic Intensity**: 0.125

This is **1,248× below** the A100's balance point. ReLU is so memory-bound it's almost embarrassing.

### The "Free" Operation Insight

When you fuse ReLU onto a compute-bound kernel like Conv:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Why Fusion Makes ReLU "Free"                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SEPARATE KERNELS:                                                     │
│   ─────────────────                                                     │
│                                                                         │
│   Conv Kernel:                                                          │
│   [████████████████████████████]  ← Compute time                       │
│   [░░░░░░░]                       ← Memory stalls (waiting for data)   │
│                                                                         │
│   ReLU Kernel:                                                          │
│   [█]                             ← Tiny compute                        │
│   [░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ← Almost all memory stalls!           │
│                                                                         │
│   Total time = Conv + ReLU launch + ReLU execution                     │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
│   FUSED KERNEL:                                                         │
│   ──────────────                                                        │
│                                                                         │
│   Conv+ReLU Kernel:                                                     │
│   [████████████████████████████]  ← Same compute time as conv alone!   │
│   [░░░░░░░]                       ← Same memory stalls                 │
│                                    + 1 extra instruction for ReLU      │
│                                    (hidden in the pipeline!)            │
│                                                                         │
│   ReLU is computed WHILE Conv is waiting for memory!                   │
│   The ReLU is literally FREE.                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The key insight: **Memory-bound ops can "piggyback" on compute-bound ops for free**.

---

## 3. Implementation

### Naive: Two Separate Kernels

```cpp
// Kernel 1: Convolution
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,  // Intermediate result written to HBM
    int N, int C, int H, int W, int K
) {
    // ... convolution logic ...
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                sum += input[...] * weight[...];
            }
        }
    }
    output[idx] = sum;  // Write to HBM (expensive!)
}

// Kernel 2: ReLU
__global__ void relu_kernel(
    const float* input,  // Read from HBM (expensive!)
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);  // Trivial compute
    }
}

// Launch both kernels
conv2d_kernel<<<grid1, block1>>>(x, weight, y, ...);  // Writes Y
relu_kernel<<<grid2, block2>>>(y, z, size);           // Reads Y, writes Z
```

**Problem**: `Y` is written to HBM by conv, then immediately read by relu. Pure waste.

### Fused: One Kernel

```cpp
__global__ void conv2d_relu_bias_fused(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int K, int out_channels
) {
    // ... same convolution logic ...
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                sum += input[...] * weight[...];
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // FUSION MAGIC: These ops happen in registers, NOT HBM!
    // ═══════════════════════════════════════════════════════════════
    
    float val = sum;              // Conv result (already in register)
    val += bias[out_channel];     // Bias add (1 register op, cheap!)
    val = fmaxf(val, 0.0f);       // ReLU (1 instruction, FREE!)
    
    output[idx] = val;            // Single write to HBM
}
```

The ReLU and bias add are **one instruction each**. They execute while the conv is stalled waiting for memory.

### Common Fusion Patterns

```cpp
// ═══════════════════════════════════════════════════════════════════════
// Pattern 1: MatMul + Bias + Activation (90% of neural networks!)
// ═══════════════════════════════════════════════════════════════════════

__device__ float apply_activation(float x, int activation_type) {
    switch (activation_type) {
        case RELU:  return fmaxf(x, 0.0f);
        case GELU:  return x * 0.5f * (1.0f + tanhf(0.7978845608f * 
                           (x + 0.044715f * x * x * x)));
        case SILU:  return x / (1.0f + expf(-x));  // x * sigmoid(x)
        default:    return x;
    }
}

__global__ void matmul_bias_activation_fused(
    const float* A, const float* B, const float* bias,
    float* C, int M, int N, int K, int activation
) {
    // ... tiled matmul logic ...
    float sum = /* result of dot product */;
    
    // All fused in registers:
    sum += bias[col];                          // Bias
    sum = apply_activation(sum, activation);   // Activation
    
    C[row * N + col] = sum;
}

// ═══════════════════════════════════════════════════════════════════════
// Pattern 2: Residual Add + LayerNorm (Transformer blocks!)
// ═══════════════════════════════════════════════════════════════════════

__global__ void residual_layernorm_fused(
    const float* x,           // Input
    const float* residual,    // Skip connection
    const float* gamma,       // LayerNorm weight
    const float* beta,        // LayerNorm bias
    float* output,
    int N, int D
) {
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Step 1: Residual add and compute mean (fused!)
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x[row * D + i] + residual[row * D + i];  // Residual add
        sdata[i] = val;  // Store for variance computation
        sum += val;
    }
    // ... parallel reduction for mean ...
    float mean = /* reduced sum */ / D;
    
    // Step 2: Variance and normalize (fused!)
    float var_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = sdata[i] - mean;
        var_sum += diff * diff;
    }
    // ... parallel reduction for variance ...
    float var = /* reduced var_sum */ / D;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // Step 3: Normalize and apply affine transform
    for (int i = tid; i < D; i += blockDim.x) {
        float normalized = (sdata[i] - mean) * inv_std;
        output[row * D + i] = gamma[i] * normalized + beta[i];
    }
}
```

---

## 4. Operator Fusion in the Real World

### PyTorch 2.0: torch.compile

```python
import torch

@torch.compile  # Automatic fusion!
def transformer_block(x, weight, bias):
    x = F.linear(x, weight, bias)
    x = F.gelu(x)
    return x

# torch.compile analyzes the graph and fuses:
# - Linear + GELU into a single kernel
# - Uses Triton to generate optimized code
```

### cuDNN Fusion

NVIDIA's cuDNN library automatically fuses common patterns:

```cpp
// cuDNN operation graph API
cudnnBackendDescriptor_t convDesc, biasDesc, reluDesc;
// ... create descriptors ...

// Create fused operation
cudnnBackendDescriptor_t fusedOps[] = {convDesc, biasDesc, reluDesc};
cudnnBackendFinalize(fusedOps, 3);  // cuDNN fuses these automatically
```

### FlashAttention: The Ultimate Fusion

Remember FlashAttention? It's kernel fusion taken to the extreme:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FlashAttention = Extreme Fusion                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   UNFUSED (Standard Attention):                                         │
│   ─────────────────────────────                                         │
│   Kernel 1: S = Q × Kᵀ         (Write N×N to HBM)                      │
│   Kernel 2: P = softmax(S)     (Read N×N, write N×N)                   │
│   Kernel 3: O = P × V          (Read N×N)                              │
│                                                                         │
│   Memory traffic: O(N²)                                                │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
│   FUSED (FlashAttention):                                              │
│   ───────────────────────                                               │
│   Single Kernel:                                                        │
│     - Compute Sᵢⱼ = Qᵢ × Kⱼᵀ   (in SRAM)                              │
│     - Compute softmax          (in SRAM)                               │
│     - Accumulate Pᵢⱼ × Vⱼ      (in SRAM)                               │
│     - Write final O            (to HBM once!)                          │
│                                                                         │
│   Memory traffic: O(N)                                                 │
│                                                                         │
│   Speedup: 2-4× (and enables 100K+ context lengths!)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Trade-off: Register Pressure

### The Danger of Over-Fusion

Fusion isn't always free. Every fused operation uses more registers:

```cpp
// Over-fused kernel (bad!)
__global__ void everything_fused(
    const float* input,
    float* output,
    /* 20 other parameters */
) {
    float a = /* conv result */;
    float b = /* batch norm: mean */;
    float c = /* batch norm: variance */;
    float d = /* batch norm: normalized */;
    float e = /* relu */;
    float f = /* another conv */;
    float g = /* more intermediate values */;
    // ... pattern continues ...
    
    // Each intermediate uses registers!
    // Compiler might spill to local memory (slow!)
}
```

### Register Pressure → Lower Occupancy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Register Pressure Trade-off                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GPU has limited registers per SM (65,536 on A100)                    │
│                                                                         │
│   Scenario A: Simple Kernel                                             │
│   ─────────────────────────                                             │
│   Registers per thread: 32                                              │
│   Max threads per SM: 65536/32 = 2048                                  │
│   Occupancy: 2048/2048 = 100%                                          │
│                                                                         │
│   Scenario B: Over-Fused Kernel                                        │
│   ─────────────────────────────                                         │
│   Registers per thread: 128                                             │
│   Max threads per SM: 65536/128 = 512                                  │
│   Occupancy: 512/2048 = 25%  ← PROBLEM!                                │
│                                                                         │
│   Lower occupancy = less latency hiding = slower execution!            │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
│   WORSE: Register Spilling                                              │
│   ────────────────────────                                              │
│   If kernel needs more registers than available:                       │
│   → Compiler "spills" to local memory (actually DRAM!)                 │
│   → Local memory access: ~100x slower than registers                   │
│   → Fusion made things SLOWER!                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Check Register Usage

```bash
# Compile with register info
nvcc -Xptxas=-v my_kernel.cu

# Output:
# ptxas info    : Used 64 registers, 8192 bytes smem, ...
```

```cpp
// Limit registers per thread (compiler hint)
__global__ __launch_bounds__(256, 4)  // 256 threads, 4 blocks per SM
void my_kernel(...) {
    // Compiler will try to use ≤ 65536/(256*4) = 64 registers
}
```

### The Sweet Spot

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fusion Strategy                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ✓ FUSE: Elementwise ops onto compute-bound kernels                   │
│     • MatMul + Bias + ReLU                                             │
│     • Conv + BatchNorm + Activation                                     │
│     • Attention components (Q×Kᵀ + softmax + ×V)                       │
│                                                                         │
│   ✓ FUSE: Consecutive memory-bound ops                                 │
│     • Residual Add + LayerNorm                                         │
│     • Multiple elementwise ops (add, multiply, activation)             │
│                                                                         │
│   ✗ DON'T FUSE: When register pressure gets too high                   │
│     • Profile first! Check occupancy with Nsight Compute               │
│     • If occupancy drops below 50%, reconsider                         │
│                                                                         │
│   ✗ DON'T FUSE: When it prevents parallelism                           │
│     • Two independent convolutions should run in parallel              │
│     • Fusion serializes them!                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Benchmarks

### Separate vs Fused Operations

| Operation | Separate (μs) | Fused (μs) | Speedup |
|-----------|--------------|------------|---------|
| MatMul + ReLU | 45 | 42 | 1.07× |
| MatMul + Bias + ReLU | 52 | 42 | 1.24× |
| MatMul + Bias + GELU | 58 | 43 | 1.35× |
| Conv + BN + ReLU | 120 | 85 | 1.41× |
| Residual + LayerNorm | 35 | 18 | 1.94× |
| Attention (3 ops) | 150 | 45 | 3.33× |

### Memory Bandwidth Savings

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Memory Traffic Comparison                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Operation: Linear(4096 → 4096) + GELU                                │
│   Tensor size: 1024 × 4096 = 4M elements = 16 MB (FP32)               │
│                                                                         │
│   SEPARATE:                                                             │
│   ─────────                                                             │
│   Linear:  Read 16MB (input) + Write 16MB (output)     = 32 MB        │
│   GELU:    Read 16MB (input) + Write 16MB (output)     = 32 MB        │
│   Total:   64 MB                                                       │
│                                                                         │
│   FUSED:                                                                │
│   ──────                                                                │
│   Linear+GELU: Read 16MB (input) + Write 16MB (output) = 32 MB        │
│   Total:   32 MB                                                       │
│                                                                         │
│   Savings: 50% less memory traffic!                                    │
│                                                                         │
│   For 3+ operations, savings compound:                                 │
│   • 3 ops: 67% savings                                                 │
│   • 4 ops: 75% savings                                                 │
│   • N ops: (N-1)/N savings                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Challenge

### Challenge 1: Fused LayerNorm

Write a single CUDA kernel that computes LayerNorm in **one pass**:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Requirements:
1. Use shared memory reductions for mean and variance
2. Handle arbitrary hidden dimensions using parallel reduction
3. Apply gamma/beta scaling in the same kernel

```cpp
__global__ void fused_layernorm(
    const float* __restrict__ input,    // (batch, hidden_dim)
    const float* __restrict__ gamma,    // (hidden_dim,)
    const float* __restrict__ beta,     // (hidden_dim,)
    float* __restrict__ output,         // (batch, hidden_dim)
    int batch_size,
    int hidden_dim,
    float epsilon
) {
    // Your implementation here!
    // 
    // Hints:
    // 1. Each block handles one row (one sample in the batch)
    // 2. Use Welford's online algorithm for numerical stability
    // 3. Two reductions: one for mean, one for variance
    // 4. Final pass: normalize and apply gamma/beta
}
```

### Challenge 2: Fused Softmax + TopK

Implement a fused kernel that computes softmax and finds the top-K values in a single pass:

```cpp
__global__ void fused_softmax_topk(
    const float* input,    // (batch, vocab_size)
    float* probs,          // (batch, vocab_size) - softmax output
    float* topk_vals,      // (batch, K)
    int* topk_indices,     // (batch, K)
    int batch_size,
    int vocab_size,
    int K
);
```

Hint: Compute softmax using online algorithm while maintaining a min-heap of size K.

### Challenge 3: Fused Attention Score + Mask + Softmax

Write a fused kernel for the attention score computation:

$$\text{Attention}(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)$$

```cpp
__global__ void fused_attention_scores(
    const half* Q,         // (batch, heads, seq_len, head_dim)
    const half* K,         // (batch, heads, seq_len, head_dim)
    const float* mask,     // (seq_len, seq_len) - causal mask
    float* attention,      // (batch, heads, seq_len, seq_len)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale            // 1/sqrt(head_dim)
);
```

---

## Summary

### Key Takeaways

| Concept | Lesson |
|---------|--------|
| **The Problem** | Separate kernels = redundant memory traffic |
| **The Solution** | Fusion = compute in registers, write once |
| **Free Ops** | Memory-bound ops piggyback on compute-bound |
| **The Trap** | Over-fusion → register pressure → slower! |
| **The Rule** | Profile, fuse wisely, verify speedup |

### The Fusion Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Should I Fuse These Operations?                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Start: Two consecutive operations A → B                              │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────────────────────────┐                             │
│   │ Is B elementwise (relu, add, etc.)? │                             │
│   └──────────────────┬───────────────────┘                             │
│                      │                                                  │
│          ┌───────────┴───────────┐                                     │
│          │ YES                   │ NO                                  │
│          ▼                       ▼                                      │
│   ┌──────────────┐      ┌────────────────────────┐                     │
│   │ FUSE IT!     │      │ Is A → B a known       │                     │
│   │ Almost       │      │ pattern (attention,    │                     │
│   │ always wins  │      │ layernorm, etc.)?      │                     │
│   └──────────────┘      └────────────┬───────────┘                     │
│                                      │                                  │
│                          ┌───────────┴───────────┐                     │
│                          │ YES                   │ NO                  │
│                          ▼                       ▼                      │
│                   ┌──────────────┐      ┌────────────────────┐         │
│                   │ FUSE with    │      │ Check register     │         │
│                   │ specialized  │      │ pressure first!    │         │
│                   │ implementation│      │ Profile before     │         │
│                   │ (cuDNN, etc.)│      │ fusing.            │         │
│                   └──────────────┘      └────────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Golden Rule

> **Eliminate intermediate tensors. Keep values in registers. Write to memory once.**

This is why:
- `torch.compile` generates fused Triton kernels
- FlashAttention fuses QK^T + softmax + @V
- cuDNN provides fused conv+bn+relu
- Every serious ML framework has a fusion pass

Master fusion, and you master GPU optimization.

---

## References

1. [NVIDIA cuDNN Fused Operations](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#fused-ops)
2. [PyTorch 2.0 torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
3. [Triton: An Intermediate Language for Blocked Algorithms](https://triton-lang.org/)
4. [FlashAttention: Fast and Memory-Efficient Attention](https://arxiv.org/abs/2205.14135)
5. [Roofline Model for GPU Performance Analysis](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
