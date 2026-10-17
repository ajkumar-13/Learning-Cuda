# 03 - Matrix multiplication: from a memory-bound dot product to shared-memory tiling

> **TL;DR.** Each element of a matrix product `C = A * B` is the dot product of one row of `A` and one column of `B`, so the obvious kernel gives one thread one output element. That kernel issues two global-memory loads for every multiply-add, the same values requested again and again, and its arithmetic units sit mostly idle. **Tiling** copies a `TILE x TILE` tile of `A` and of `B` into **shared memory** once per block, behind a barrier, and divides the global loads by `TILE`. On the author's GTX 1650 the tiled kernel is 1.6 times faster than the naive one at $1024^3$, and cuBLAS 7 times faster again.
>
> **Prerequisites:** [post 02](../02-vector-addition/index.md).
> **Safe to skip?** Skip it if the reader can already write a 2D kernel index, count loads per element, and place both barriers of a shared-memory tiled kernel.
>
> **After reading, you will be able to:**
>
> - Map a 2D output matrix onto a 2D grid of 16 x 16 blocks and compute any thread's (row, col) by hand.
> - Count the global loads per output element of the naive and the tiled kernel, and explain why the naive kernel is memory bound.
> - Write a tiled kernel with shared-memory tiles and two barriers, and say what each barrier protects.
> - Read a naive, tiled, and cuBLAS benchmark in ms and GFLOP/s, and name what a tuned library adds beyond one level of tiling.

![Matrices A, B and C in the multiply layout. Row 2 of A and column 4 of B, shaded green, meet at cell 2, 4 of C, shaded orange: the one thread that computes it, with its loop line beside.](diagrams/01-dot-product.svg)

*One thread per output element: the grid of threads covers `C`, and each thread runs only the loop over `k`.*

---

## 1. The question: why is the obvious matrix multiply slow?

Given `A: [M, K]` and `B: [K, N]`, the product `C: [M, N]` has one element per row of `A` and column of `B`, a dot product over the shared dimension `K`:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$$

Every output element is independent, so a graphics processing unit (GPU) can compute them all at once; the figure at the top draws thread `(row, col)` computing `C[row][col]`. Each element costs `K` multiply-adds, two floating-point operations (FLOP) each, so one product is $2MNK$ FLOP, about 2.1 billion for $M = N = K = 1024$.

The arithmetic is not the problem: each multiply-add needs one value of `A` and one of `B`, and the obvious kernel fetches both from global memory every time. This post counts those fetches, removes most of them with shared memory, and measures the result against NVIDIA's library.

## 2. Two-dimensional indexing

A matrix uses a **2D grid of 2D blocks**: each thread takes its `(row, col)` from the 2D form of the global index `blockIdx.x * blockDim.x + threadIdx.x`, `row` from the `.y` components, in this fragment:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

A block of `16 x 16 = 256` threads, 8 warps, is the usual start. The grid is sized with a ceiling division in each dimension, and the threads past the edge of `C` are switched off by the guard `row < M && col < N`. The central processing unit (CPU) model `snippets/matmul_model.py` derives the launch for a `1000 x 1000` output and one thread's index:

```text
1. launch: 16 x 16 blocks over a 1000 x 1000 output
  gridDim     = (63, 63) blocks of (16, 16) threads
  threads     = 1008 x 1008 = 1,016,064
  surplus     = 1,016,064 - 1,000,000 = 16,064  (fail row < M && col < N)
  thread blockIdx = (2, 1), threadIdx = (5, 3): row = 1 * 16 + 3 = 19, col = 2 * 16 + 5 = 37
```

The figure finds that thread.

![The corner of a 63 by 63 grid of blocks over a 1,000 by 1,000 output, block 2, 1 opened into 16 by 16 threads, and thread 5, 3 worked to row 19, col 37; 16,064 threads fail the guard.](diagrams/02-grid-2d.svg)

*The last row and column of blocks hang past the edge of `C`.*

## 3. The naive kernel, and where its time goes

The direct translation gives thread `(row, col)` one full dot product, from `snippets/matmul.cu`:

```cuda
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];     // K loads from each of A and B
        C[row * N + col] = sum;
    }
}
```

It is correct and slow. Every multiply-add needs two loads of 4 bytes, 8 bytes for every 2 FLOP, and the requests repeat: element `A[r][k]` is read by every thread of row `r` of `C`, `N` of them, and `B[k][c]` by every thread of column `c`, `M` of them. At $1024^3$ the kernel issues $2 \times 1024^3$ loads, 8 GiB of requests, for two inputs of 4 MiB each.

The model runs both kernels on the CPU, thread by thread and block by block, and counts every global load. For a `64 x 64 x 64` product it prints:

```text
2. global loads, A[64,64] x B[64,64] -> C[64,64], TILE = 16
  naive == reference      : True
  tiled == reference      : True
  naive global loads      : 524,288
  tiled global loads      : 32,768
  reduction factor        : 16  (= TILE)
3. per output element (2K FLOP each)
  naive : 128 loads = 512 bytes, 0.25 FLOP per byte loaded
  tiled : 8 loads = 32 bytes, 4.00 FLOP per byte loaded
```

If every request went to dynamic random-access memory (DRAM) at the GTX 1650's theoretical 160 GB/s, 0.25 FLOP per byte would cap the naive kernel at $0.25 \times 160 = 40$ GFLOP/s; section 5 measures about 200, so most requests never reach DRAM: a warp's 32 threads need 2 distinct values of `A` and 16 of `B` per step, and the L1 and L2 caches serve repeats across warps (inferred from the rates). They cannot remove the load instructions: every multiply-add still waits on two, and the card's arithmetic peak of about 3,100 GFLOP/s stays far out of reach. In that sense the naive kernel is **memory bound**: its time goes to fetching operands, not to arithmetic.

## 4. The fix: tiling with shared memory

The cure is to fetch each value once per block and reuse it from **shared memory**, the on-chip scratchpad of a block. The block walks `K` in steps of `TILE`. At each step every thread copies one element of `A` into `s_A: [TILE, TILE]` and one of `B` into `s_B: [TILE, TILE]`, the block waits until both tiles are complete, every thread adds its `TILE` products from the tiles, and the block waits again before the next step overwrites them. The figure sets the two kernels side by side.

![Left, four naive threads each read the same row of A from global memory, 128 loads per output, 524,288 in all. Right, tiles of A and B loaded once into s_A and s_B, used between two purple barriers, then the next tile: 8 loads per output, 32,768 in all.](diagrams/03-naive-vs-tiled.svg)

*Each value crosses from global memory once per block that needs it and is reused `TILE` times from shared memory.*

The tiled kernel, from `snippets/matmul.cu`, launched with `TILE x TILE` threads per block:

```cuda
__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[TILE][TILE];                   // launched with blockDim = (TILE, TILE)
    __shared__ float s_B[TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;              // coalesced load of A
        s_A[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;              // coalesced load of B
        s_B[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                // tiles are loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();                                // done before the next load
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```

Each `__syncthreads()` protects one thing. The first protects this step's reads from its writes: a thread reads 32 slots of the tiles, 30 of them written by other threads, and none may be read before it is written. The second protects the next step's writes from this step's reads: no thread may overwrite a slot while a slower thread still multiplies with it. Every thread of the block must reach a barrier, so the loads are unconditional, out-of-range slots get zero (which also keeps the sum right when a size is not a multiple of `TILE`), and only the final write is guarded. The index uses `blockDim`, so the kernel is correct only for a `TILE x TILE` block.

Each thread now issues 2 global loads per step instead of `2 * TILE`, so the model's count falls by a factor of `TILE` and 0.25 FLOP per byte loaded becomes 4.

## 5. The numbers: naive, tiled, and cuBLAS

`matmul.cu` times both kernels and the cuBLAS routine `cublasSgemm` from $256^3$ to $2048^3$ with CUDA events (2 warm-up launches, then the median of 5 runs of 10 launches), converts each time to GFLOP/s with $2MNK$ FLOP, and checks every result against a double-precision CPU reference on 4,096 sampled elements. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
Device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5)
C = A * B with M = N = K = size; ms per launch, median of 5 runs of 10 launches
  size    naive ms  GFLOP/s    tiled ms  GFLOP/s   cuBLAS ms  GFLOP/s   max rel err
   256       0.157      214       0.102      328       0.037      919   6.8e-07
   512       1.184      227       0.814      330       0.179     1502   9.8e-07
  1024      10.939      196       7.032      305       0.981     2189   1.4e-06
  2048     108.001      159      77.244      222       9.946     1727   2.5e-06
every result matches the CPU reference (relative error below 1e-4): PASS
```

These are measured numbers from one quiet run; the reader's card will differ, and a hot laptop card drifts by tens of percent between runs, so the shape of the table is the lesson. At $1024^3$ the tiled kernel takes $10.939 / 7.032 = 1.6$ times less time than the naive one, and cuBLAS $7.032 / 0.981 = 7.2$ times less than the tiled one, 11 times less than the naive one.

![Four bar charts of GFLOP/s, naive, tiled and cuBLAS, one per size: 214, 328, 919 at 256 cubed; 227, 330, 1,502 at 512; 196, 305, 2,189 at 1,024; 159, 222, 1,727 at 2,048.](diagrams/04-matmul-perf.svg)

*Tiling lifts the naive kernel by about half; the library is three to eight times beyond the tiled kernel.*

The tiled kernel gains 1.6 times, not 16, for two reasons. Most of the naive kernel's repeats never reached DRAM (section 3), so tiling removed little DRAM traffic. And the tiled inner loop still issues two loads per multiply-add, now from shared memory, plus a barrier per tile; at 305 GFLOP/s it uses about a tenth of the card's arithmetic peak of $14 \times 64 \times 2 \times 1.740 \approx 3{,}118$ GFLOP/s (14 streaming multiprocessors, SMs, of 64 lanes, 2 FLOP per fused multiply-add, 1,740 MHz boost clock). cuBLAS reaches 2,189 GFLOP/s, about 70% of that peak. Both assume the boost clock; under load nvidia-smi read mostly 990 to 1,500 MHz, so the true shares are larger.

The library stacks what this post stops short of. **Register tiling** gives each thread a small block of outputs held in registers, so each value read from shared memory feeds several multiply-adds; **vectorized `float4` loads** move four floats per instruction; **double buffering** loads the next tile while the current one is in use. Boehm's worklog (December 2022) climbs the first two on an RTX A6000 from 1.3% of cuBLAS for an uncoalesced naive kernel to 93.7% (a published figure, from another card), and CUTLASS's "Efficient GEMM in CUDA" (general matrix multiply) describes the same register tiling and double buffering. Double buffering is the subject of [post 15](../15-async-copy-pipelining/index.md) and CUTLASS of [post 19](../19-cutlass-triton/index.md).

## 6. Make it run

The CPU model needs no GPU. From the series root:

```bash
uv run python posts/03-matrix-multiplication/snippets/matmul_model.py
```

It prints the launch of section 2 and the load counts of section 3, also at $1024^3$.

The kernel program needs an NVIDIA GPU and the CUDA Toolkit, and links cuBLAS. From the series root, with the card's `-arch`:

```bash
nvcc -O3 -arch=sm_75 posts/03-matrix-multiplication/snippets/matmul.cu -o matmul -lcublas && ./matmul
```

On Windows the program is `matmul.exe`, and `nvcc` may need `-ccbin` to find the host compiler; [RUNNING.md](../../RUNNING.md) covers both. It runs in about 15 seconds and exits with status 1 if a check fails.

## 7. What can go wrong?

Each mistake below was reproduced by editing `matmul.cu` on the author's GTX 1650; only the correctness check caught it.

- **A missing first barrier.** Threads read slots their neighbors have not written yet; the results look plausible and are wrong by a few percent (relative error about 4e-02 at $256^3$).
- **A missing second barrier.** A fast thread overwrites the tile while a slow one still reads it; the error is smaller and varies with timing (8.5e-03 in one run).
- **A block that does not match `TILE`.** An `8 x 8` block with a `TILE` of 16 leaves three quarters of each tile unloaded and of `C` unwritten. It ran two to three times faster, with relative error 1.0: a fast wrong kernel looks like an optimization.
- **cuBLAS called with row-major arguments.** cuBLAS is column-major, so passing `A`, `B` and the row-major sizes straight through computes a different product (relative error 0.26). The row-major `C = A * B` is the column-major $C^T = B^T A^T$: `matmul.cu` passes `B` first and swaps `M` and `N`.
- **A barrier inside a branch.** A `__syncthreads()` inside `if (row < M && col < N)` is reached by only some threads of an edge block, which is undefined behavior: it can hang, and at $1000^3$ it returned relative error 0.51.

---

## Common pitfalls

1. **Expecting the load count's speedup.** Tiling divides global loads by `TILE`, but most never reached DRAM and the inner loop still loads from shared memory; measure.
2. **Making the tile as large as possible.** `TILE = 32` means 1,024 threads per block, one block per SM on the GTX 1650, which can cost more occupancy than it gains in reuse ([post 10](../10-roofline/index.md)).
3. **Uncoalesced tile loads.** Consecutive `threadIdx.x` must read consecutive addresses; a stride of `N` multiplies a warp's transactions.
4. **Reading a tile down its columns.** This kernel's shared-memory reads are free of bank conflicts, but a column-wise read of a tile hits one of the 32 banks repeatedly and needs the padded `[TILE][TILE + 1]` layout of [post 06](../06-matrix-transpose/index.md).
5. **Timing without a warm-up and a synchronize.** A launch returns before the kernel finishes and the first pays one-time costs; time with CUDA events after a warm-up.

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Shared Memory", consulted October 2026. The shared-memory tiled matrix multiply this post follows.
- Boehm, S., "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" (December 2022). From a naive kernel to 93.7% of cuBLAS, one optimization at a time.
- NVIDIA, *CUTLASS: CUDA Templates for Linear Algebra Subroutines*, consulted October 2026, and its "Efficient GEMM in CUDA" document. How a production GEMM tiles and double-buffers.
- NVIDIA, *cuBLAS Library Documentation*, consulted October 2026. The `cublasSgemm` baseline.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 04 - Reduction: summing millions of numbers without a serial bottleneck](../04-reduction/index.md):** threads that must cooperate on one result, with the same shared memory and barrier rules.
- **[Post 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy](../10-roofline/index.md):** reads this post's table against the card's bandwidth and compute roofs.
