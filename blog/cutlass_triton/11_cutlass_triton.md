# CUTLASS & Triton

> **Why nobody writes raw CUDA for MatMul anymore**

---

## 1. Introduction

### The Journey So Far

You've learned raw CUDA C++. You understand:
- Thread hierarchies and memory coalescing
- Shared memory tiling and bank conflicts
- Tensor Cores and WMMA
- Double buffering and software pipelining

You can write a matrix multiplication kernel from scratch. **But should you?**

### The Maintenance Nightmare

Here's what a production-grade GEMM kernel needs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    What a "Real" GEMM Kernel Needs                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ✓ Tiling for shared memory (multiple levels)                         │
│   ✓ Tensor Core integration (WMMA or newer MMA)                        │
│   ✓ Double/triple buffering for latency hiding                         │
│   ✓ Bank-conflict-free shared memory layouts                           │
│   ✓ Register-level tiling within warps                                 │
│   ✓ Software pipelining with async copies (cp.async)                   │
│   ✓ Epilogue fusion (bias, activation, residual add)                   │
│   ✓ Split-K for skinny matrices                                        │
│   ✓ Stream-K for better load balancing                                 │
│   ✓ Mixed precision (FP16 compute, FP32 accumulate)                    │
│   ✓ Different layouts (row-major, column-major, mixed)                 │
│   ✓ Batched and grouped GEMM                                           │
│   ✓ Different GPU architectures (Volta, Ampere, Hopper, Blackwell)     │
│                                                                         │
│   Lines of code: 5,000 - 50,000                                        │
│   Time to write: Months                                                │
│   Time to debug: Eternity                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Evolution of GPU Programming

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Abstraction Ladder                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Level 5: PyTorch / JAX                                               │
│            torch.matmul(A, B)                                          │
│            ↑ Easiest, calls optimized libraries                        │
│            │                                                            │
│   Level 4: cuBLAS / cuDNN                                              │
│            cublasSgemm(...)                                            │
│            ↑ Black box, peak performance, no customization             │
│            │                                                            │
│   Level 3: Triton (Python DSL)                                         │
│            @triton.jit with block-level programming                    │
│            ↑ Easy to write, good performance, customizable             │
│            │                                                            │
│   Level 2: CUTLASS (C++ Templates)                                     │
│            cutlass::gemm::device::Gemm<...>                            │
│            ↑ Near-peak performance, steep learning curve               │
│            │                                                            │
│   Level 1: Raw CUDA C++                                                │
│            __global__ void gemm_kernel(...)                            │
│            ↑ Maximum control, maximum pain                             │
│                                                                         │
│   This blog: Levels 2 and 3                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CUTLASS: NVIDIA's Template Library

### What is CUTLASS?

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source C++ template library. It provides:

- **Composable components** for building GEMM and convolution kernels
- **Architecture-specific optimizations** for each GPU generation
- **Template-based design** for compile-time specialization

Think of it as "LEGO blocks for GPU linear algebra."

### The Hierarchy Model

CUTLASS organizes computation into a strict hierarchy:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CUTLASS Hierarchy                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                         DEVICE                                   │  │
│   │  The entire GEMM: C = α(A × B) + βC                             │  │
│   │                                                                  │  │
│   │  ┌───────────────────────────────────────────────────────────┐  │  │
│   │  │                    THREADBLOCK TILE                        │  │  │
│   │  │  Typically 128×128 or 256×128                             │  │  │
│   │  │  Uses shared memory for A and B tiles                     │  │  │
│   │  │                                                            │  │  │
│   │  │  ┌─────────────────────────────────────────────────────┐  │  │  │
│   │  │  │                   WARP TILE                          │  │  │  │
│   │  │  │  Typically 64×64 or 32×32                           │  │  │  │
│   │  │  │  Each warp computes a portion                       │  │  │  │
│   │  │  │                                                      │  │  │  │
│   │  │  │  ┌───────────────────────────────────────────────┐  │  │  │  │
│   │  │  │  │              MMA INSTRUCTION                   │  │  │  │  │
│   │  │  │  │  Tensor Core operation (e.g., 16×8×16)        │  │  │  │  │
│   │  │  │  │  The actual hardware instruction              │  │  │  │  │
│   │  │  │  └───────────────────────────────────────────────┘  │  │  │  │
│   │  │  └─────────────────────────────────────────────────────┘  │  │  │
│   │  └───────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   You configure each level via template parameters!                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### CUTLASS 2.x: The Classic API

```cpp
#include <cutlass/gemm/device/gemm.h>

// Define the GEMM operation via template parameters
// Yes, this is intimidating. But each parameter has a purpose.

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // Element type of A
    cutlass::layout::RowMajor,                 // Layout of A
    cutlass::half_t,                           // Element type of B  
    cutlass::layout::RowMajor,                 // Layout of B
    cutlass::half_t,                           // Element type of C
    cutlass::layout::RowMajor,                 // Layout of C
    float,                                     // Accumulator type
    cutlass::arch::OpClassTensorOp,            // Use Tensor Cores!
    cutlass::arch::Sm80,                       // Target Ampere (A100)
    cutlass::gemm::GemmShape<128, 128, 32>,    // Threadblock tile (M, N, K)
    cutlass::gemm::GemmShape<64, 64, 32>,      // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,       // MMA instruction shape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8,                    // Output type, alignment
        float, float                           // Accumulator, compute type
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                          // Pipeline stages
>;

int main() {
    // Problem size
    int M = 4096, N = 4096, K = 4096;
    
    // Allocate matrices (device memory)
    cutlass::half_t *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&B, K * N * sizeof(cutlass::half_t));
    cudaMalloc(&C, M * N * sizeof(cutlass::half_t));
    
    // Initialize Gemm arguments
    Gemm gemm_op;
    Gemm::Arguments args{
        {M, N, K},           // Problem size
        {A, K},              // TensorRef for A (ptr, stride)
        {B, N},              // TensorRef for B
        {C, N},              // TensorRef for C (input)
        {C, N},              // TensorRef for D (output)
        {1.0f, 0.0f}         // alpha, beta
    };
    
    // Launch!
    cutlass::Status status = gemm_op(args);
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed!" << std::endl;
    }
    
    return 0;
}
```

### CUTLASS 3.x: The Modern API (CuTe)

CUTLASS 3.x introduces **CuTe** (Cute Tensor), a more flexible tensor abstraction:

```cpp
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

using namespace cute;

// CuTe uses "Layouts" to describe memory organization
// Much more readable than the 2.x template soup!

// Define the problem shape
auto M = Int<128>{};
auto N = Int<128>{};
auto K = Int<32>{};

// Define a layout for a row-major matrix
auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));

// Create a tensor from pointer + layout
Tensor A = make_tensor(make_gmem_ptr(ptr_A), layout_A);

// Tile the tensor for threadblock-level processing
auto tiled_A = zipped_divide(A, make_tile(
    make_layout(Int<64>{}),   // Tile M
    make_layout(Int<32>{})    // Tile K
));

// CuTe handles the indexing automatically!
// Much cleaner than raw pointer arithmetic
```

### When to Use CUTLASS

| Use Case | CUTLASS? |
|----------|----------|
| Need absolute peak performance | ✅ Yes |
| Fused epilogues (GEMM + bias + relu) | ✅ Yes |
| Custom matrix layouts | ✅ Yes |
| Rapid prototyping | ❌ No (use Triton) |
| Need to ship tomorrow | ❌ No (use cuBLAS) |

---

## 3. Triton: OpenAI's Python DSL

### What is Triton?

**Triton** is a Python-based domain-specific language (DSL) for writing GPU kernels. Key features:

- **Block-level programming**: You think in terms of tiles, not threads
- **Automatic optimization**: Compiler handles memory coalescing, shared memory
- **Python syntax**: Easy to read, easy to iterate
- **JIT compilation**: Compiles to optimized PTX at runtime

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Triton's Magic                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   What you write (Python):                                             │
│   ─────────────────────────                                             │
│   offs = pid * BLOCK + tl.arange(0, BLOCK)                             │
│   x = tl.load(X + offs)                                                │
│   y = x * x                                                            │
│   tl.store(Y + offs, y)                                                │
│                                                                         │
│   What Triton generates:                                               │
│   ──────────────────────                                                │
│   • Coalesced global memory loads                                      │
│   • Shared memory staging (if beneficial)                              │
│   • Optimal thread-to-data mapping                                     │
│   • Bank-conflict-free layouts                                         │
│   • Tensor Core usage for matrix ops                                   │
│                                                                         │
│   You don't see threadIdx.x, blockIdx.x, or __syncthreads()!          │
│   Triton handles all of that for you.                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Triton Vector Add

Compare this to our raw CUDA vector add from Blog 1:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def vector_add_kernel(
    X_ptr,           # Pointer to input X
    Y_ptr,           # Pointer to input Y  
    Z_ptr,           # Pointer to output Z
    N,               # Number of elements
    BLOCK: tl.constexpr  # Block size (compile-time constant)
):
    # Which block am I?
    pid = tl.program_id(0)  # Like blockIdx.x, but simpler!
    
    # Compute offsets for this block
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # Vector of offsets!
    
    # Mask for boundary handling
    mask = offs < N
    
    # Load data (automatically coalesced!)
    x = tl.load(X_ptr + offs, mask=mask)
    y = tl.load(Y_ptr + offs, mask=mask)
    
    # Compute
    z = x + y
    
    # Store result
    tl.store(Z_ptr + offs, z, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper function that launches the Triton kernel."""
    assert x.shape == y.shape
    z = torch.empty_like(x)
    
    N = x.numel()
    BLOCK = 1024
    
    # Calculate grid size
    grid = ((N + BLOCK - 1) // BLOCK,)
    
    # Launch kernel
    vector_add_kernel[grid](x, y, z, N, BLOCK)
    
    return z


# Usage
x = torch.randn(1000000, device='cuda')
y = torch.randn(1000000, device='cuda')
z = vector_add(x, y)
```

**Notice:**
- No `threadIdx.x` — Triton uses `tl.program_id` (block-level)
- No `__syncthreads()` — Triton handles synchronization
- `tl.arange` creates a vector — you program at the block level
- Boundary handling is clean with `mask`

### Triton Matrix Multiplication

Here's where Triton really shines:

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B
    
    This kernel:
    - Uses tiling (BLOCK_M × BLOCK_N output tile per program)
    - Accumulates in registers
    - Handles arbitrary matrix sizes
    """
    # Which output tile am I computing?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Pointers to the start of this output tile's input rows/cols
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointer arithmetic for A and B tiles
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Initialize accumulator (in registers!)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles (Triton handles shared memory staging!)
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k))
        b = tl.load(B_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N))
        
        # Matrix multiply-accumulate (uses Tensor Cores if available!)
        acc += tl.dot(a, b)
        
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Store result
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=mask)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for Triton matmul."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    
    # Tune these for your GPU!
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return C
```

### Triton's Autotuning

One of Triton's killer features: **automatic tuning**!

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # Retune when these change
)
@triton.jit
def matmul_kernel_autotuned(...):
    # Same kernel code as before!
    # Triton will try all configs and pick the fastest
    pass
```

---

## 4. Comparison: CUDA vs CUTLASS vs Triton

### Side-by-Side: Vector Add

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Vector Add Comparison                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   RAW CUDA:                                                             │
│   ─────────                                                             │
│   __global__ void add(float* x, float* y, float* z, int n) {           │
│       int i = blockIdx.x * blockDim.x + threadIdx.x;                   │
│       if (i < n) z[i] = x[i] + y[i];                                   │
│   }                                                                     │
│   // Launch: add<<<(n+255)/256, 256>>>(x, y, z, n);                    │
│                                                                         │
│   TRITON:                                                               │
│   ───────                                                               │
│   @triton.jit                                                          │
│   def add(X, Y, Z, N, BLOCK: tl.constexpr):                           │
│       pid = tl.program_id(0)                                           │
│       offs = pid * BLOCK + tl.arange(0, BLOCK)                        │
│       mask = offs < N                                                  │
│       tl.store(Z + offs, tl.load(X + offs, mask) +                    │
│                          tl.load(Y + offs, mask), mask)               │
│                                                                         │
│   Difference: Triton thinks in BLOCKS, not threads!                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Approach | Dev Time | Performance | Flexibility | Learning Curve |
|----------|----------|-------------|-------------|----------------|
| **cuBLAS** | Minutes | 100% | None | Easy |
| **Triton** | Hours | 90-98% | High | Medium |
| **CUTLASS** | Days | 99-100% | Highest | Hard |
| **Raw CUDA** | Weeks | Variable | Total | Hardest |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GEMM Performance (A100, 4096×4096×4096, FP16)       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   cuBLAS          ████████████████████████████████████████  275 TFLOPS │
│   (Baseline)                                                100%       │
│                                                                         │
│   CUTLASS         ███████████████████████████████████████   270 TFLOPS │
│   (Tuned)                                                   98%        │
│                                                                         │
│   Triton          ██████████████████████████████████████    263 TFLOPS │
│   (Autotuned)                                               95%        │
│                                                                         │
│   Triton          █████████████████████████████████         248 TFLOPS │
│   (Naive)                                                   90%        │
│                                                                         │
│   Raw CUDA        █████████████████████                     165 TFLOPS │
│   (Your kernel)                                             60%        │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│   For most use cases, Triton's 90-95% is more than good enough!       │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### When to Use What

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Decision Tree                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   "I need a fast GEMM"                                                 │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────┐                                      │
│   │ Is it a standard operation? │                                      │
│   │ (GEMM, Conv, Attention)     │                                      │
│   └────────────┬────────────────┘                                      │
│                │                                                        │
│       ┌────────┴────────┐                                              │
│       │ YES             │ NO (custom op)                               │
│       ▼                 ▼                                               │
│   ┌─────────┐    ┌─────────────────────────┐                           │
│   │ cuBLAS/ │    │ Do I need 100% perf?    │                           │
│   │ cuDNN   │    └───────────┬─────────────┘                           │
│   │ (done!) │                │                                          │
│   └─────────┘       ┌────────┴────────┐                                │
│                     │ YES             │ NO                             │
│                     ▼                 ▼                                 │
│               ┌──────────┐     ┌───────────┐                           │
│               │ CUTLASS  │     │ Triton    │                           │
│               │ (pain)   │     │ (joy)     │                           │
│               └──────────┘     └───────────┘                           │
│                                                                         │
│   Special cases:                                                        │
│   • Learning GPU programming → Raw CUDA (understand the hardware!)    │
│   • Need custom epilogues   → CUTLASS or Triton                        │
│   • Research iteration      → Triton (fast to change)                  │
│   • Production at scale     → cuBLAS/CUTLASS (squeeze every TFLOP)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Real-World Examples

### FlashAttention in Triton

FlashAttention has official Triton implementations! Here's a simplified version:

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified FlashAttention in Triton."""
    pid = tl.program_id(0)
    
    # Initialize output accumulator and running stats
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    
    # Load Q block (stays in SRAM for entire inner loop!)
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N)
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Inner loop over K, V blocks
    for j in range(0, N, BLOCK_N):
        offs_n = j + tl.arange(0, BLOCK_N)
        
        # Load K, V blocks
        k_ptrs = K + offs_n[None, :] * stride_km + offs_d[:, None] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N)
        
        # Compute attention scores
        s = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
        
        # Online softmax update
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(s - m_ij[:, None]), axis=1)
        
        # Update output
        p = tl.exp(s - m_new[:, None])
        o_i = alpha[:, None] * o_i + tl.dot(p.to(tl.float16), v)
        
        m_i = m_new
        l_i = l_new
    
    # Final normalization and store
    o_i = o_i / l_i[:, None]
    o_ptrs = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_i.to(tl.float16), mask=offs_m[:, None] < N)
```

### Fused LayerNorm in Triton

```python
@triton.jit
def layernorm_kernel(
    X, Y, W, B,
    stride,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    """Fused LayerNorm: Y = W * (X - mean) / sqrt(var + eps) + B"""
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    
    # Load row
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std
    
    # Apply affine transform
    w = tl.load(W + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    y = x_norm * w + b
    
    # Store
    tl.store(Y + offs, y.to(tl.float16), mask=mask)
```

---

## 6. Getting Started

### Installing Triton

```bash
pip install triton
```

### Installing CUTLASS

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80  # For A100
make -j
```

### Your First Triton Kernel

```python
import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    tl.store(Y_ptr + offs, x * x, mask=mask)

def square(x):
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    square_kernel[(N + BLOCK - 1) // BLOCK,](x, y, N, BLOCK)
    return y

# Test it!
x = torch.randn(1000000, device='cuda')
y = square(x)
assert torch.allclose(y, x * x)
print("Triton works!")
```

---

## 7. Challenge

### Challenge 1: Triton Vector Add

Take the vector add from Blog 1 and rewrite it in Triton:

1. Notice how you use `tl.program_id(0)` instead of `blockIdx.x * blockDim.x + threadIdx.x`
2. Use `tl.arange(0, BLOCK)` to create a vector of offsets
3. Benchmark against your raw CUDA kernel

```python
@triton.jit
def vector_add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    # Your code here!
    pass
```

### Challenge 2: Triton Softmax

Implement a numerically stable softmax in Triton:

```python
@triton.jit
def softmax_kernel(X, Y, stride, N, BLOCK: tl.constexpr):
    """
    Compute softmax(X) row-wise.
    
    Hints:
    1. Each program handles one row
    2. Use tl.max for the max value (numerical stability)
    3. Use tl.exp and tl.sum for the softmax
    """
    pass
```

> **💡 Pro Tip:** Softmax involves a **reduction** across each row—this is trickier than element-wise ops! Use `tl.max(x, axis=0)` and `tl.sum(x, axis=0)` to reduce across your block. Remember: Triton operates on blocks of data, so your reduction happens over the block dimension. The pattern is:
> ```python
> max_val = tl.max(x, axis=0)           # Reduce to find max
> x_stable = x - max_val                # Numerical stability
> exp_x = tl.exp(x_stable)              # Exponentiate
> sum_exp = tl.sum(exp_x, axis=0)       # Reduce to find sum
> softmax = exp_x / sum_exp             # Normalize
> ```

### Challenge 3: CUTLASS GEMM

Configure a CUTLASS GEMM for your GPU:

1. Choose appropriate tile sizes for your GPU's shared memory
2. Add a ReLU epilogue (fused bias + activation)
3. Benchmark against cuBLAS

```cpp
// Hint: Look at CUTLASS examples/00_basic_gemm
using Gemm = cutlass::gemm::device::Gemm<
    // Your configuration here!
>;
```

---

## Summary

### Key Takeaways

| Tool | Philosophy | Best For |
|------|-----------|----------|
| **Raw CUDA** | Thread-level control | Learning, maximum flexibility |
| **CUTLASS** | Template-based composition | Peak performance, custom ops |
| **Triton** | Block-level programming | Rapid iteration, "good enough" perf |

### The Evolution of GPU Programming

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The Meta-Lesson                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   2007: "Learn CUDA, write everything from scratch"                    │
│                                                                         │
│   2015: "Use cuBLAS for GEMM, CUDA for custom ops"                    │
│                                                                         │
│   2020: "Use CUTLASS for custom high-perf GEMM"                        │
│                                                                         │
│   2023: "Use Triton for custom ops, cuBLAS for standard"              │
│                                                                         │
│   2025: "torch.compile handles most fusion; Triton for the rest"       │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════      │
│   The trend: Higher abstraction, same (or better!) performance        │
│                                                                         │
│   But understanding the lower levels makes you better at the           │
│   higher levels. That's why you learned raw CUDA first!               │
│   ═══════════════════════════════════════════════════════════════      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### When to Use Each

| Scenario | Tool |
|----------|------|
| Standard GEMM/Conv | cuBLAS/cuDNN |
| Custom fused op, need speed | Triton |
| Absolute peak performance | CUTLASS |
| Learning GPU programming | Raw CUDA |
| Research prototyping | Triton |
| Production at scale | cuBLAS → CUTLASS |

---

## References

1. [Triton Documentation](https://triton-lang.org/)
2. [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
3. [Triton: An Intermediate Language for Blocked Algorithms](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
4. [CUTLASS Documentation](https://nvidia.github.io/cutlass/)
5. [OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
6. [FlashAttention in Triton](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/triton)
