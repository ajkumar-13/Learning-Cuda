# 03 · Matrix multiplication — from a memory-bound dot product to shared-memory tiling

> **TL;DR.** Matrix multiply is the operation under every neural network, and each output element `C[row][col]` is a **dot product** of a row of `A` and a column of `B`, so the obvious kernel gives one thread one output element. That kernel works, but it is **memory bound**: neighboring threads re-read the *same* row and column from slow global memory, moving gigabytes when the inputs are megabytes. The fix is **tiling**: each block cooperatively loads a small tile of `A` and `B` into fast **shared memory** once, then every thread reuses it, cutting global traffic by a factor of the tile size. This post builds the naive kernel, shows exactly why it stalls, then derives the tiled kernel and benchmarks both against cuBLAS.
>
> **After reading this you will be able to:**
> - Map a 2D output matrix onto a 2D grid of blocks and write the `row`/`col` index by hand.
> - Explain, in loads-per-element, why the naive kernel is memory bound.
> - Write a tiled kernel with `__shared__` memory and `__syncthreads()`, and say what each barrier protects.
> - Read a naive-vs-tiled-vs-cuBLAS benchmark and explain the remaining gap.

![Three matrices in the standard layout: a highlighted row of A and a highlighted column of B meet at one highlighted element of C, which equals their dot product.](diagrams/01-dot-product.svg)
*Every element of the output is one row of `A` dotted with one column of `B`. One thread computes one element.*

---

## 1. The motivation: the operation under deep learning

Given `A` of shape `[M, K]` and `B` of shape `[K, N]`, the product `C` has shape `[M, N]`, and each element is a dot product over the shared dimension `K`:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik}\, B_{kj}$$

Three properties make this a perfect GPU workload: every output element is **independent** (massively parallel), the access pattern is **regular**, and there is potentially **high arithmetic intensity** (many multiply-adds per byte, once you stop re-reading memory). The catch is in that "once": each output needs an entire row of `A` and column of `B`, which is a great deal of memory traffic. As the rest of the post shows, the naive version drowns in it, and the whole craft is getting the data movement down.

## 2. Two-dimensional indexing

Vectors used a 1D grid; matrices use a **2D grid of 2D blocks**, so each thread gets a `(row, col)` from the 2D form of the global-index rule:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

A block of `16 x 16 = 256` threads is the usual starting point: it divides evenly into 8 warps and covers a `16 x 16` tile of the output. The grid is sized by ceiling division over both dimensions, so as always some threads land outside the matrix and are switched off by a boundary check.

## 3. The naive kernel, and why it stalls

The direct translation gives each thread one output element and one full dot product:

```cuda
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];   // K reads each from A and B
        C[row * N + col] = sum;
    }
}
```

It is correct, and it is slow, because of the access pattern. Consider a `1024 x 1024 x 1024` multiply. Each of the `1024 x 1024` threads reads 1024 values from `A` and 1024 from `B`: about `2 x 1024^3 ~= 2.1 billion` float reads, which at 4 bytes each is ~8 GB. But `A` and `B` are only 4 MB each. **Every value is fetched from global memory roughly a thousand times**, because neighboring threads independently re-read the same row and column.

![Four threads computing one row of C all re-read the same row of A from global memory, shown as four redundant red arrows.](diagrams/02-naive-reuse.svg)
*The arithmetic is cheap; the redundant global traffic is the whole cost. Naive matmul is memory bound.*

A small CPU model counts it exactly. For a `64 x 64 x 64` problem the naive kernel issues **524,288** global loads, and confirms the result matches a reference product. It lives at [snippets/matmul_model.py](snippets/matmul_model.py):

```text
matmul model
  sizes             = A[64,64] x B[64,64] -> C[64,64], TILE=16
  naive == ref      = True
  tiled == ref      = True
  naive global loads= 524,288
  tiled global loads= 32,768
  reduction factor  = 16x  (= TILE)
```

The full kernel (both naive and tiled, with a correctness check) lives at [snippets/matmul.cu](snippets/matmul.cu). Build and run it, or run the CPU model, with:

```bash
nvcc -O3 snippets/matmul.cu -o matmul && ./matmul
python snippets/matmul_model.py
```

## 4. The fix: tiling with shared memory

The redundancy has an obvious cure: load each value from global memory **once**, into the fast on-chip **shared memory** every block has, then let all the threads in the block reuse it. The block walks the `K` dimension in tiles of width `TILE`. For each tile it cooperatively loads a `TILE x TILE` block of `A` and of `B` into shared memory, synchronizes, computes that tile's contribution to every output, synchronizes again, and moves on.

![A flow from global memory into shared memory (loaded once, coalesced) into per-thread compute, looping over the K tiles; shared memory is highlighted in green.](diagrams/03-tiled-shared.svg)
*Each value is loaded from global memory once and reused about `TILE` times from shared memory.*

```cuda
#define TILE 16
__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;                       // coalesced
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;                       // coalesced
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                          // tiles loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();                                          // done before reuse
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```

Three details carry the whole kernel. The **two `__syncthreads()`** are not optional: the first guarantees every thread has finished *loading* the tile before any thread *reads* it, and the second guarantees every thread has finished reading before the next iteration *overwrites* it. The **loads are coalesced**: adjacent threads (varying `threadIdx.x`) read adjacent addresses, so a warp's reads collapse into a few memory transactions. And the **out-of-range slots are zero-filled** rather than skipped, so the dot product stays correct for any matrix size without a special case in the inner loop. The model above confirms the payoff: `524,288` global loads fall to `32,768`, a `16x` cut equal to `TILE`.

## 5. The numbers: naive, tiled, and cuBLAS

On a GTX 1650 (14 SMs, compute capability 7.5, ~192 GB/s) the tiling pays off, and a glance at NVIDIA's own library shows how far there is still to go. To turn a time into a rate, count the work: a multiply of `[M,K]` by `[K,N]` does `2*M*N*K` floating-point operations (one multiply and one add per inner step), so `GFLOPS = 2*M*N*K / time`. For the `1024³` case that is `2 x 1024^3 ~= 2.1 GFLOP` per call. The shipped [matmul.cu](snippets/matmul.cu) checks correctness only; the naive, tiled, and cuBLAS times below were measured separately on a GTX 1650 (compute capability 7.5).

| Implementation | Time (1024³) | GFLOPS | Speedup |
|---|---|---|---|
| GPU naive | 15.90 ms | 135 | baseline |
| GPU tiled (TILE=16) | 8.31 ms | 258 | 1.9× |
| cuBLAS | 0.82 ms | 2616 | 19.4× |

![A bar chart of GFLOPS: naive 135, tiled 258 in green, cuBLAS 2616 in blue towering over both.](diagrams/04-matmul-perf.svg)
*Tiling roughly doubles the naive kernel; cuBLAS is another order of magnitude beyond.*

The tiled kernel is about **1.9× the naive** kernel here, not the full `16×` the load count promised, because shared memory cut the *memory* cost but left synchronization, instruction overhead, and modest occupancy in place. The shape holds as matrices grow:

| Size | Naive | Tiled | cuBLAS |
|---|---|---|---|
| 256³ | 0.25 ms (135) | 0.16 ms (207) | 0.06 ms (602) |
| 512³ | 2.01 ms (134) | 1.23 ms (218) | 0.27 ms (997) |
| 1024³ | 15.90 ms (135) | 8.31 ms (258) | 0.82 ms (2616) |
| 2048³ | 75.71 ms (227) | 49.81 ms (345) | 6.43 ms (2670) |

cuBLAS reaches **2.6 TFLOPS** on a laptop GPU because it stacks optimizations this post has not: **register tiling** (each thread computes several outputs, cutting shared-memory traffic too), **vectorized `float4` loads**, **double buffering** (load the next tile while computing the current one), and warp-level primitives. The lesson is not to feel bad about the gap; it is that tiling is the first rung of a long ladder, and that calling a tuned library is usually the right move in production.

## 6. Why this matters

Matrix multiply is the foundation of deep learning (every dense and attention layer), graphics, and scientific computing. The two ideas here, **one thread per output element** and **load-once-reuse-from-shared-memory**, recur in almost every kernel that follows: reductions, transpose, convolution, and the attention kernels later in the series all turn on getting data into fast memory and keeping it there.

---

## Common pitfalls

- **Forgetting `__syncthreads()`.** Drop the first barrier and threads read a tile other threads have not finished loading; drop the second and the next iteration overwrites a tile still in use. Both are races that produce wrong, non-deterministic results.
- **Calling `__syncthreads()` inside a divergent branch.** Every thread in the block must reach the same barrier. A `__syncthreads()` that only some threads hit (inside an `if (row < M)`) can hang the block; load into shared memory unconditionally (with zero-fill), then guard only the final write.
- **Non-coalesced loads.** Indexing `B` so that adjacent threads stride by `N` instead of by 1 throws away most of the bandwidth. Make `threadIdx.x` the fastest-varying index of the address.
- **Bank conflicts in shared memory.** Shared memory has 32 banks; a column-major access where a warp hits one bank serializes. Padding a tile to `[TILE][TILE+1]` is the standard fix when a pattern conflicts.
- **Assuming a bigger tile is always faster.** `TILE=32` means 1024 threads and 8 KB of shared memory per block, which can cut occupancy enough to *lose* performance despite more reuse. Tile size trades reuse against occupancy; measure it.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — Shared Memory"* (current). The reference for `__shared__`, `__syncthreads()`, and bank conflicts (technical, reference).
- Boehm, S., *"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"* (2022). A step-by-step climb from naive to near-cuBLAS, naming every optimization (technical).
- NVIDIA, *"CUTLASS: CUDA Templates for Linear Algebra Subroutines"*. The open-source library that shows how production GEMM is structured (technical, reference).
- NVIDIA, *"cuBLAS Library Documentation"* (current). The tuned baseline this post benchmarks against (reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 02, Vector addition](../02-vector-addition/index.md)**: the memory-bound first kernel whose bandwidth lesson this post inverts into a compute problem.
- **[Post 04, Reduction](../04-reduction/index.md)**: what changes when threads must *cooperate* to produce one number, with shared memory and warp shuffles.
