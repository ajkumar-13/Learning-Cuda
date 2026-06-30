# Notation guide

The goal is that a symbol means the same thing in every post: a reader who learns $\alpha$ is the learning rate here never has to relearn it elsewhere.

Two rules govern everything below:

1. **One quantity, one symbol — per post.** Never write the same quantity two ways in one post. Re-introduce each symbol on first use in *every* post, because readers arrive mid-series.
2. **Introduce every symbol with its shape**, e.g. a tile is `tile: [TILE, TILE]`, the inputs of a matrix multiply are `A: [M, K]`, `B: [K, N]`, `C: [M, N]`.

---

## 1. Typography (when notation is typeset as math)

| Kind | Style | Example |
|---|---|---|
| Matrix | **bold upper-case** | $\mathbf{X}$, $\mathbf{W}$ |
| Vector | **bold lower-case** | $\mathbf{x}$, $\mathbf{b}$ |
| Scalar | *italic lower-case* | $i$, $t$, $\alpha$ |
| Set / space | blackboard or upright | $\mathbb{R}^{d}$ |
| Index range | italic, stated with a shape | $i \in \{0, \dots, N-1\}$ |

Most posts express notation as a mix of typeset math and **code-span
identifiers** (`blockIdx.x`, `d_A`, `TILE`). When you typeset math, follow the
table; when you name the same thing in prose or code, use the code-span
identifier in §3 so prose, math, and CUDA agree.

## 2. Canonical symbols (identical across all series)

These are the **only** accepted forms. The right-hand column is banned.

| Quantity | Use | Never use |
|---|---|---|
| Learning rate | $\alpha$ | $\eta$ |
| Loss / objective | $L$ | $\mathcal{L}$ |
| Bias vector | $\mathbf{b}$ (lower-case) | $\mathbf{B}$ |
| Small constant (numerical) | $\epsilon$ | $\varepsilon$ |
| Weight matrix | $\mathbf{W}$ | $W$ (un-bolded) for a matrix |
| Input / activations matrix | $\mathbf{X}$ | $\mathbf{A}$ for generic input |
| Element-wise product | $\odot$ | $*$ |


> **CUDA corollary — name the concrete arrays in code spans.** Because typeset $\mathbf{B}$ is reserved (banned as a bias alias), refer to the *concrete* kernel operands by their **code-span identifiers**: `A`, `B`, `C` for a vector add, `A`, `B`, `C` for `C = A·B` in matrix multiply, and `d_A`, `d_B` for their device copies. Reserve typeset bold ($\mathbf{X}$, $\mathbf{W}$) for the abstract linear-algebra framing, and write the second matmul operand as $\mathbf{X}$ if you must typeset it — never $\mathbf{B}$.

## 3. CUDA / GPU symbol table (this series)

The teaching artifact is a sequence of hand-written CUDA kernels, from a one-line vector add up to a tiled GEMM, a warp-shuffle reduction, and a FlashAttention tile loop. Keep prose, math, and code aligned to the identifiers below.

| Concept | Symbol / identifier | Shape / type |
|---|---|---|
| Problem size (1-D) | $N$ / `N` | scalar |
| Global element index | $i$ / `i`, `idx` | scalar |
| Thread index in block | `threadIdx.x` | scalar |
| Block index in grid | `blockIdx.x` | scalar |
| Threads per block | $B$ / `blockDim.x`, `blockSize` | scalar (multiple of 32) |
| Blocks per grid | `gridDim.x`, `blocksPerGrid` | scalar |
| Warp size | 32 (constant) | scalar |
| Global index formula | `blockIdx.x * blockDim.x + threadIdx.x` | scalar |
| Matrix-multiply operands | `A`, `B`, `C` for $\mathbf{C}=\mathbf{A}\mathbf{X}$ | `[M,K]`,`[K,N]`,`[M,N]` |
| Tile width | `TILE` | scalar |
| Shared-memory tile | `s_A`, `s_B` | `[TILE, TILE]` |
| Arithmetic intensity | $I$ (FLOP / byte) | scalar |
| Peak DRAM bandwidth | $\beta$ (GB/s) | scalar |
| Achieved bandwidth | fraction of $\beta$ | scalar |

Conventions:

- **The global index is always `blockIdx.x * blockDim.x + threadIdx.x`**, named `i` (1-D) or `(row, col)` (2-D). Introduce it once per post.
- **A kernel is described from one thread's point of view.** Write "thread `i` does …", not "the loop does …": on the GPU the grid *is* the loop.
- **Bandwidth-bound vs compute-bound** is decided by arithmetic intensity $I$: state $I$ in FLOP/byte and compare it to the device's FLOP-to-bandwidth ratio.

