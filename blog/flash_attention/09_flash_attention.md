# FlashAttention Explained

> **How tiling and math tricks solved the sequence length bottleneck**

---

## 1. Introduction

### The Quadratic Wall

Transformers have a dirty secret: they scale **quadratically**.

| Sequence Length | Attention Matrix Size | Memory (FP16) |
|-----------------|----------------------|---------------|
| 512 tokens | 512 × 512 = 262K | 0.5 MB |
| 2,048 tokens | 2K × 2K = 4M | 8 MB |
| 8,192 tokens | 8K × 8K = 67M | 134 MB |
| 32,768 tokens | 32K × 32K = 1B | 2 GB |
| 128,000 tokens | 128K × 128K = 16B | **32 GB** |

Double the sequence length → **4× the memory**.

This is why GPT-2 was limited to 1,024 tokens. How does GPT-4 handle 128K? How does Claude handle 200K?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Quadratic Memory Problem                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Memory Usage                                                          │
│   ▲                                                                     │
│   │                                              ╱                      │
│   │                                           ╱                        │
│   │                                        ╱                           │
│   │                                     ╱    Standard                  │
│   │                                  ╱       Attention                 │
│   │                               ╱          O(N²)                     │
│   │                            ╱                                       │
│   │                         ╱                                          │
│   │                      ╱                                             │
│   │                   ╱                                                │
│   │                ╱                                                   │
│   │             ╱   ─────────────────────────────                      │
│   │          ╱      FlashAttention: O(N) memory!                       │
│   │       ╱                                                            │
│   │    ╱                                                               │
│   └──────────────────────────────────────────────────► Sequence Length │
│                                                                         │
│   The secret: Never materialize the N×N attention matrix!              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Problem: Materializing the Attention Matrix

Standard Attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The naive implementation does this in **three separate GPU kernels**:

```python
# Standard PyTorch Attention (simplified)
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)  # Kernel 1: N×N matrix!
attention = torch.softmax(scores, dim=-1)                   # Kernel 2: Read N×N, write N×N
output = torch.matmul(attention, V)                         # Kernel 3: Read N×N again
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Standard Attention Memory Flow                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   HBM (Global Memory)              GPU Compute                          │
│   ═══════════════════              ═══════════                          │
│                                                                         │
│   ┌─────────────┐                                                       │
│   │ Q (N × d)   │ ───────────┐                                         │
│   └─────────────┘            │     ┌─────────────┐                     │
│                              ├───► │  MatMul     │                     │
│   ┌─────────────┐            │     │  QKᵀ        │                     │
│   │ K (N × d)   │ ───────────┘     └──────┬──────┘                     │
│   └─────────────┘                         │                             │
│                                           ▼                             │
│   ┌─────────────┐              ┌─────────────────┐                     │
│   │ Scores      │ ◄────────── │ WRITE N×N       │  ← BOTTLENECK!      │
│   │ (N × N)     │              │ to HBM          │                     │
│   └──────┬──────┘              └─────────────────┘                     │
│          │                                                              │
│          ▼                                                              │
│   ┌──────┴──────┐              ┌─────────────────┐                     │
│   │ READ N×N    │ ───────────► │   Softmax       │                     │
│   │ from HBM    │              └──────┬──────────┘                     │
│   └─────────────┘                     │                                 │
│                                       ▼                                 │
│   ┌─────────────┐              ┌─────────────────┐                     │
│   │ Attention   │ ◄────────── │ WRITE N×N       │  ← BOTTLENECK!      │
│   │ (N × N)     │              │ to HBM          │                     │
│   └──────┬──────┘              └─────────────────┘                     │
│          │                                                              │
│          ▼                                                              │
│   ┌──────┴──────┐              ┌─────────────────┐                     │
│   │ READ N×N    │ ───────────► │  MatMul × V     │                     │
│   │ from HBM    │              └──────┬──────────┘                     │
│   └─────────────┘                     │                                 │
│                                       ▼                                 │
│   ┌─────────────┐              ┌─────────────────┐                     │
│   │ Output      │ ◄────────── │ Write (N × d)   │                     │
│   │ (N × d)     │              └─────────────────┘                     │
│   └─────────────┘                                                       │
│                                                                         │
│   PROBLEM: We read/write the N×N matrix THREE times!                   │
│   For N=32K: 3 × 2GB = 6GB of memory traffic per attention layer       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

Each intermediate result (the $N \times N$ scores and attention matrices) must be:
1. **Written** to HBM (slow)
2. **Read back** from HBM for the next operation (slow again)

This is the bottleneck. Not compute — **memory bandwidth**.

### The Solution: FlashAttention

**FlashAttention** computes the exact same mathematical result, but:
- **Never writes** the $N \times N$ matrix to HBM
- **Fuses** all three operations into a single kernel
- Uses **tiling** and **online softmax** to compute block-by-block

The result? **2-4× faster** and uses **O(N)** memory instead of **O(N²)**.

---

## 2. The Math Trick: Online Softmax

### The Problem with Standard Softmax

Softmax over a vector $\mathbf{x}$ of length $N$:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$

To compute this, you need to:
1. Compute **all** $e^{x_j}$ values
2. Sum them up
3. Divide each by the sum

This requires seeing **all** $N$ elements before producing **any** output. How can we compute this block-by-block?

### The "Safe Softmax" Trick

First, a numerical stability trick. Computing $e^{1000}$ overflows. So we subtract the max:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^{N} e^{x_j - \max(\mathbf{x})}}$$

This is mathematically identical but numerically stable.

### Online Softmax: The Key Insight

Here's the magic. We can compute softmax **incrementally** by maintaining running statistics:

1. **Running max** $m$: The maximum value seen so far
2. **Running sum** $\ell$: The sum of $e^{x_i - m}$ seen so far

When we see a new block of values, we **update** these statistics:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Online Softmax Algorithm                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Process Block 1: [x₁, x₂, x₃, x₄]                                    │
│   ─────────────────────────────────                                     │
│   m₁ = max(x₁, x₂, x₃, x₄)                                             │
│   ℓ₁ = Σ exp(xᵢ - m₁)                                                  │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Process Block 2: [x₅, x₆, x₇, x₈]                                    │
│   ─────────────────────────────────                                     │
│   m₂_local = max(x₅, x₆, x₇, x₈)                                       │
│   m₂ = max(m₁, m₂_local)           ← New global max!                   │
│                                                                         │
│   # Correction: Previous sum was computed with old max                 │
│   ℓ₁_corrected = ℓ₁ × exp(m₁ - m₂)  ← Scale down if max increased     │
│                                                                         │
│   ℓ₂_local = Σ exp(xᵢ - m₂)         ← Sum for new block               │
│   ℓ₂ = ℓ₁_corrected + ℓ₂_local      ← Total running sum               │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   After all blocks:                                                     │
│   softmax(xᵢ) = exp(xᵢ - m_final) / ℓ_final                           │
│                                                                         │
│   KEY INSIGHT: We never needed the full N×N matrix in memory!          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Update Equations

When processing a new block with local max $\tilde{m}$ and local sum $\tilde{\ell}$:

$$m_{\text{new}} = \max(m_{\text{old}}, \tilde{m})$$

$$\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \tilde{\ell} \cdot e^{\tilde{m} - m_{\text{new}}}$$

And for the running output (which we'll use for attention):

$$O_{\text{new}} = O_{\text{old}} \cdot \frac{\ell_{\text{old}}}{\ell_{\text{new}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \frac{\tilde{O} \cdot e^{\tilde{m} - m_{\text{new}}}}{\ell_{\text{new}}}$$

Where $\tilde{O}$ is the partial output from the current block.

---

## 3. The Algorithm: Tiling Q, K, V

### The Memory Hierarchy Strategy

FlashAttention exploits the GPU memory hierarchy:

| Memory | Size | Bandwidth | Latency |
|--------|------|-----------|---------|
| **HBM** (Global) | 40-80 GB | 2 TB/s | High |
| **SRAM** (Shared) | ~200 KB per SM | 19 TB/s | Low |
| **Registers** | ~256 KB per SM | — | Instant |

The goal: Keep the hot $N \times N$ computation in SRAM, only read/write $Q$, $K$, $V$, $O$ from HBM.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FlashAttention Tiling Strategy                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Q (N × d)              K^T (d × N)              V (N × d)            │
│   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐        │
│   │ Q₁ │ Q₂ │...│        │K₁│K₂│K₃│...│        │ V₁ │ V₂ │...│        │
│   │────│────│   │        │──│──│──│   │        │────│────│   │        │
│   │ Q₃ │ Q₄ │   │        │  │  │  │   │        │ V₃ │ V₄ │   │        │
│   │────│────│   │        │  │  │  │   │        │────│────│   │        │
│   │ ...│    │   │        │  │  │  │   │        │ ...│    │   │        │
│   └─────────────┘        └─────────────┘        └─────────────┘        │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
│   OUTER LOOP: For each block Qᵢ (rows of Q)                            │
│   ────────────────────────────────────────                              │
│       Load Qᵢ into SRAM                                                │
│       Initialize: Oᵢ = 0, mᵢ = -∞, ℓᵢ = 0                              │
│                                                                         │
│       INNER LOOP: For each block Kⱼ, Vⱼ (columns of K, rows of V)      │
│       ─────────────────────────────────────────────────────────        │
│           Load Kⱼ, Vⱼ into SRAM                                        │
│                                                                         │
│           ┌───────────────────────────────────────────┐                │
│           │  IN SRAM (Fast!)                          │                │
│           │  ─────────────────                        │                │
│           │  Sᵢⱼ = Qᵢ × Kⱼᵀ / √d    (Block scores)   │                │
│           │  m̃ = max(Sᵢⱼ)                             │                │
│           │  P̃ᵢⱼ = exp(Sᵢⱼ - m̃)     (Block softmax)  │                │
│           │  ℓ̃ = sum(P̃ᵢⱼ)                             │                │
│           │  Õ = P̃ᵢⱼ × Vⱼ           (Block output)   │                │
│           │                                           │                │
│           │  # Update running statistics              │                │
│           │  m_new = max(mᵢ, m̃)                       │                │
│           │  ℓ_new = ℓᵢ × exp(mᵢ - m_new) + ...      │                │
│           │  Oᵢ = rescale(Oᵢ) + rescale(Õ)           │                │
│           └───────────────────────────────────────────┘                │
│                                                                         │
│       Write Oᵢ to HBM (only once per Qᵢ block!)                        │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
│   Result: We NEVER store the full N×N matrix in HBM!                   │
│   Only the block-sized tiles exist in SRAM at any time.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Works: The IO Complexity Analysis

**Standard Attention:**
- Read Q, K: $O(Nd)$
- Write Scores: $O(N^2)$ ← **Bottleneck!**
- Read Scores: $O(N^2)$
- Write Attention: $O(N^2)$
- Read Attention: $O(N^2)$
- Total: $O(N^2)$ memory IO

**FlashAttention:**
- Read Q, K, V: $O(Nd)$ each, but multiple passes...
- Total: $O(N^2 d / M)$ where $M$ is SRAM size

For typical $M$ (SRAM) >> $d$ (head dimension), this is **much smaller**.

The key insight: **We trade extra computation for less memory IO**. Since:
- Compute is fast (Tensor Cores: 300+ TFLOPS)
- Memory is slow (HBM: 2 TB/s)

This trade is a **huge win**.

---

## 4. The Implementation

### Pseudocode

```python
def flash_attention(Q, K, V, block_size):
    """
    Q, K, V: (N, d) matrices in HBM
    Returns: O (N, d) output matrix
    """
    N, d = Q.shape
    O = zeros(N, d)  # Output accumulator
    m = full(N, -inf)  # Running max per row
    l = zeros(N)  # Running sum per row

    # Outer loop: iterate over blocks of Q
    for i in range(0, N, block_size):
        # Load Q block into SRAM
        Qi = Q[i:i+block_size]  # (block_size, d)
        
        Oi = zeros(block_size, d)
        mi = full(block_size, -inf)
        li = zeros(block_size)

        # Inner loop: iterate over blocks of K, V
        for j in range(0, N, block_size):
            # Load K, V blocks into SRAM
            Kj = K[j:j+block_size]  # (block_size, d)
            Vj = V[j:j+block_size]  # (block_size, d)

            # ═══════════════════════════════════════════
            # All of this happens in SRAM (fast!)
            # ═══════════════════════════════════════════
            
            # Compute block scores
            Sij = Qi @ Kj.T / sqrt(d)  # (block_size, block_size)
            
            # Block-wise max and softmax
            mij = Sij.max(dim=-1)  # (block_size,)
            Pij = exp(Sij - mij[:, None])  # Safe softmax numerator
            lij = Pij.sum(dim=-1)  # (block_size,)
            
            # Update running max
            mi_new = maximum(mi, mij)
            
            # Rescale previous accumulator
            alpha = exp(mi - mi_new)
            beta = exp(mij - mi_new)
            
            li_new = alpha * li + beta * lij
            
            # Update output accumulator  
            Oi = (alpha[:, None] * li[:, None] * Oi + 
                  beta[:, None] * Pij @ Vj) / li_new[:, None]
            
            mi = mi_new
            li = li_new

        # Write output block to HBM (only once!)
        O[i:i+block_size] = Oi
        m[i:i+block_size] = mi
        l[i:i+block_size] = li

    return O
```

### The CUDA Kernel (Simplified)

```cpp
#include <cuda_fp16.h>
#include <mma.h>

// Block sizes tuned for A100 (192KB shared memory per SM)
#define BLOCK_M 64  // Rows of Q per block
#define BLOCK_N 64  // Rows of K/V per block
#define BLOCK_D 64  // Head dimension

// ┌─────────────────────────────────────────────────────────────────────┐
// │ PRO TIP: Block Size Tuning                                         │
// │                                                                     │
// │ These 64×64 blocks are optimized for A100/H100 SRAM (~192-228KB).  │
// │ Smaller GPUs need smaller blocks to avoid register spilling:       │
// │   • T4 / RTX 3060: Try 32×32 blocks                                │
// │   • RTX 3090 / A6000: 64×64 usually works                          │
// │   • H100: Can potentially go larger (96×96)                        │
// │                                                                     │
// │ Rule of thumb: Total SRAM usage ≈ 2×(BLOCK_M + BLOCK_N)×BLOCK_D×2  │
// │ (×2 for FP16). Must fit in shared memory with room for scores.    │
// └─────────────────────────────────────────────────────────────────────┘

__global__ void flash_attention_kernel(
    // __restrict__ tells the compiler these pointers don't alias (point to
    // overlapping memory). This enables aggressive optimizations:
    //   1. Better L1/L2 cache utilization (no false sharing concerns)
    //   2. Instruction reordering (loads can be hoisted)
    //   3. Vectorized loads (compiler can batch memory accesses)
    // Without __restrict__, compiler must assume Q, K, V, O might overlap!
    const half* __restrict__ Q,  // (N, d)
    const half* __restrict__ K,  // (N, d)
    const half* __restrict__ V,  // (N, d)
    half* __restrict__ O,        // (N, d)
    float* __restrict__ L,       // (N,) logsumexp for backward
    int N, int d
) {
    // Shared memory for tiles
    __shared__ half Qi[BLOCK_M][BLOCK_D];
    __shared__ half Kj[BLOCK_N][BLOCK_D];
    __shared__ half Vj[BLOCK_N][BLOCK_D];
    __shared__ float Sij[BLOCK_M][BLOCK_N];  // Scores

    // Thread block handles one block of Q rows
    int block_row = blockIdx.x;
    int row_start = block_row * BLOCK_M;

    // Initialize running statistics
    float mi[BLOCK_M];  // Max per row (in registers)
    float li[BLOCK_M];  // Sum per row (in registers)
    float Oi[BLOCK_M][BLOCK_D];  // Output accumulator

    for (int i = 0; i < BLOCK_M; i++) {
        mi[i] = -INFINITY;
        li[i] = 0.0f;
        for (int j = 0; j < BLOCK_D; j++) {
            Oi[i][j] = 0.0f;
        }
    }

    // Load Q block into shared memory (cooperatively)
    // ... (standard tiled loading)

    // Inner loop: iterate over K, V blocks
    for (int block_col = 0; block_col < (N + BLOCK_N - 1) / BLOCK_N; block_col++) {
        int col_start = block_col * BLOCK_N;

        // Load Kj, Vj into shared memory
        // ... (standard tiled loading)
        __syncthreads();

        // Compute Sij = Qi × Kj^T (using Tensor Cores ideally)
        // ... (matrix multiply in shared memory)
        __syncthreads();

        // Compute block-wise softmax and update running stats
        for (int i = threadIdx.y; i < BLOCK_M; i += blockDim.y) {
            // Find max in this row of Sij
            float mij = -INFINITY;
            for (int j = 0; j < BLOCK_N; j++) {
                mij = fmaxf(mij, Sij[i][j]);
            }

            // Compute exp and sum
            float lij = 0.0f;
            float Pij[BLOCK_N];
            for (int j = 0; j < BLOCK_N; j++) {
                Pij[j] = expf(Sij[i][j] - mij);
                lij += Pij[j];
            }

            // Update running statistics
            float mi_new = fmaxf(mi[i], mij);
            float alpha = expf(mi[i] - mi_new);
            float beta = expf(mij - mi_new);
            float li_new = alpha * li[i] + beta * lij;

            // Rescale and accumulate output
            float scale_old = alpha * li[i] / li_new;
            float scale_new = beta / li_new;

            for (int k = 0; k < BLOCK_D; k++) {
                float pv = 0.0f;
                for (int j = 0; j < BLOCK_N; j++) {
                    pv += Pij[j] * __half2float(Vj[j][k]);
                }
                Oi[i][k] = scale_old * Oi[i][k] + scale_new * pv;
            }

            mi[i] = mi_new;
            li[i] = li_new;
        }
        __syncthreads();
    }

    // Write output to HBM
    for (int i = threadIdx.y; i < BLOCK_M; i += blockDim.y) {
        int global_row = row_start + i;
        if (global_row < N) {
            for (int k = threadIdx.x; k < BLOCK_D; k += blockDim.x) {
                O[global_row * d + k] = __float2half(Oi[i][k]);
            }
            // Save logsumexp for backward pass
            L[global_row] = mi[i] + logf(li[i]);
        }
    }
}
```

---

## 5. FlashAttention vs Standard: Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Memory Access Pattern Comparison                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   STANDARD ATTENTION                                                    │
│   ══════════════════                                                    │
│                                                                         │
│   HBM         SRAM        HBM         SRAM        HBM                  │
│   ┌───┐       ┌───┐       ┌───┐       ┌───┐       ┌───┐               │
│   │ Q │──────►│   │       │   │       │   │       │   │               │
│   │ K │──────►│QKᵀ│──────►│S  │──────►│sft│──────►│ A │               │
│   └───┘       └───┘ write └───┘ read  └───┘ write └───┘               │
│                     N×N         N×N         N×N                         │
│                                                                         │
│                           ┌───┐       ┌───┐       ┌───┐               │
│                           │ A │──────►│   │──────►│ O │               │
│                           │ V │──────►│A×V│       │   │               │
│                           └───┘ read  └───┘       └───┘               │
│                                 N×N                                     │
│                                                                         │
│   Total HBM traffic: 3 × N² (write scores, write attn, read attn)     │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   FLASH ATTENTION                                                       │
│   ═══════════════                                                       │
│                                                                         │
│   HBM         SRAM (stays here!)                            HBM        │
│   ┌───┐       ┌─────────────────────────────────────────┐   ┌───┐     │
│   │ Q │──────►│  Load Qᵢ                                │   │   │     │
│   │   │       │    ┌─────┐                              │   │   │     │
│   │ K │──────►│    │Sᵢⱼ =│──►softmax──►×Vⱼ──►accumulate│──►│ O │     │
│   │   │       │    │QᵢKⱼᵀ│     ↑         ↑              │   │   │     │
│   │ V │──────►│    └─────┘     │         │              │   │   │     │
│   │   │       │          (never leaves SRAM!)           │   │   │     │
│   └───┘       └─────────────────────────────────────────┘   └───┘     │
│                                                                         │
│   Total HBM traffic: O(N × d) — NO N² terms!                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Benchmarks

### Performance Comparison

| Sequence Length | Standard Attention | FlashAttention | Speedup |
|-----------------|-------------------|----------------|---------|
| 512 | 0.5 ms | 0.3 ms | 1.7× |
| 2,048 | 4.2 ms | 1.1 ms | 3.8× |
| 8,192 | 68 ms | 8 ms | 8.5× |
| 16,384 | 280 ms | 18 ms | 15.6× |
| 32,768 | OOM | 42 ms | ∞* |

> **\*** The ∞ speedup indicates that FlashAttention enables workloads that were previously **impossible** due to memory constraints. You can't measure speedup against a baseline that crashes!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FlashAttention Speedup                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Sequence     Standard         FlashAttention                         │
│   Length       Attention                                                │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   512          ██                █                           1.7×      │
│                                                                         │
│   2,048        █████████         ██                          3.8×      │
│                                                                         │
│   8,192        ████████████████████████████████████  ████    8.5×      │
│                                                                         │
│   16,384       [================OOM================]  ████   15.6×      │
│                (Out of Memory!)                                         │
│                                                                         │
│   32,768       [================OOM================]  █████    ∞       │
│                                                                         │
│   65,536       [================OOM================]  ██████   ∞       │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│   KEY: FlashAttention enables sequence lengths that were IMPOSSIBLE!   │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Usage

| Sequence Length | Standard Attention | FlashAttention | Reduction |
|-----------------|-------------------|----------------|-----------|
| 2,048 | 16 MB | 0.5 MB | 32× |
| 8,192 | 256 MB | 2 MB | 128× |
| 32,768 | 4 GB | 8 MB | 512× |
| 131,072 | 64 GB (impossible) | 32 MB | ∞ |

---

## 7. The Backward Pass: Recomputation

### The Problem

For training, we need gradients. Standard attention saves the $N \times N$ attention matrix for the backward pass. But FlashAttention doesn't store it!

### The Solution: Recomputation

We **recompute** the attention matrix during the backward pass, block by block. This sounds expensive, but:

1. **Memory saved** >> **Compute cost**
2. We're still IO-bound, so extra compute is "free"
3. Modern GPUs have massive compute capacity

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Recomputation vs. Storage Trade-off                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   STANDARD ATTENTION BACKWARD                                           │
│   ═══════════════════════════                                           │
│                                                                         │
│   Forward:  Save attention matrix A (N×N) to HBM                       │
│   Backward: Load A from HBM, compute gradients                         │
│                                                                         │
│   Memory: O(N²)                                                        │
│   IO:     Write N² (forward) + Read N² (backward)                      │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   FLASH ATTENTION BACKWARD                                              │
│   ════════════════════════                                              │
│                                                                         │
│   Forward:  Save only Q, K, V, O, and L (logsumexp)                    │
│   Backward: RECOMPUTE attention matrix block-by-block                  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  For each block:                                                 │  │
│   │    1. Recompute Sij = Qi × Kj^T                                 │  │
│   │    2. Recompute Pij = softmax(Sij) using saved L                │  │
│   │    3. Compute gradients dQ, dK, dV                              │  │
│   │    4. Accumulate gradients                                       │  │
│   │                                                                  │  │
│   │  All done in SRAM — no N×N HBM access!                          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Memory: O(N)                                                         │
│   IO:     O(N²d/M) same as forward                                     │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│   Trade: 2× compute (recompute in backward)                            │
│   Win:   N²× less memory, enabling longer sequences!                   │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Recomputation Works

The key insight: **Memory is the bottleneck, not compute**.

| Resource | Capability | Utilization |
|----------|-----------|-------------|
| A100 Compute | 312 TFLOPS | Often 20-50% |
| A100 Memory BW | 2 TB/s | Often 90%+ |

We have **spare compute cycles**. Using them to recompute the attention matrix instead of storing it is a **net win**.

---

## 8. FlashAttention-2 and Beyond

### FlashAttention-2 Improvements

1. **Better parallelism**: Parallelize over sequence length, not just batch/heads
2. **Reduced non-matmul FLOPs**: Minimize the softmax overhead
3. **Better work partitioning**: Balance work across warps

Result: **2× faster** than FlashAttention-1!

### FlashAttention-3 (Hopper)

Leverages new Hopper GPU features:
- **TMA**: Tensor Memory Accelerator for async loads
- **WGMMA**: Warp Group Matrix Multiply-Accumulate
- **FP8**: Even lower precision for more speed

---

## 9. Challenge

### Challenge 1: Implement Online Softmax

Write a CUDA kernel that computes softmax using the online algorithm:
1. Process input in blocks of 256 elements
2. Maintain running max and sum
3. Produce the correct softmax output

Verify against `torch.softmax`.

### Challenge 2: Masked Attention

Extend the FlashAttention algorithm to support:
- **Causal masking** (for autoregressive models)
- **Custom attention masks**

Hint: When computing $S_{ij}$, set masked positions to $-\infty$ before the softmax.

### Challenge 3: Multi-Head Attention

Implement a complete multi-head attention layer using FlashAttention:
1. Linear projections for Q, K, V
2. Split into heads
3. FlashAttention per head
4. Concatenate and project output

Benchmark against `torch.nn.MultiheadAttention`.

---

## Summary

### Key Takeaways

| Concept | Lesson |
|---------|--------|
| **Quadratic Bottleneck** | Standard attention is O(N²) memory |
| **Online Softmax** | Compute softmax incrementally with running stats |
| **Tiling** | Process Q, K, V in blocks that fit in SRAM |
| **Fusion** | Combine matmul + softmax + matmul in one kernel |
| **Recomputation** | Trade compute for memory in backward pass |

### The Memory Hierarchy Insight

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FlashAttention Core Principle                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   "Make the algorithm fit the hardware, not vice versa"                │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
│   Hardware Reality:                                                     │
│   • GPU compute: 300+ TFLOPS (FAST)                                    │
│   • GPU HBM:     2 TB/s (SLOW relative to compute)                     │
│   • GPU SRAM:    19 TB/s but tiny (~200KB)                             │
│                                                                         │
│   Algorithm Adaptation:                                                 │
│   • Keep hot data (attention scores) in SRAM                           │
│   • Never write N×N matrix to HBM                                      │
│   • Trade extra compute (recomputation) for less memory IO             │
│                                                                         │
│   Result:                                                               │
│   • 2-4× faster than standard attention                                │
│   • O(N) memory instead of O(N²)                                       │
│   • Enables 100K+ context lengths                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What This Enables

| Before FlashAttention | After FlashAttention |
|----------------------|---------------------|
| GPT-2: 1K context | GPT-4: 128K context |
| BERT: 512 tokens | Long-context models |
| OOM on long docs | Process entire books |
| Memory-bound | Compute-bound (good!) |

---

## References

1. [FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2: Faster Attention with Better Parallelism (Dao, 2023)](https://arxiv.org/abs/2307.08691)
3. [Online Softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
4. [FlashAttention GitHub Repository](https://github.com/Dao-AILab/flash-attention)
5. [NVIDIA Cutlass Attention](https://github.com/NVIDIA/cutlass/tree/main/examples/41_fused_multi_head_attention)
