# 04 · Reduction — summing millions of numbers without a serial bottleneck

> **TL;DR.** Adding up an array is trivial on a CPU and surprisingly hard on a GPU: point a thousand threads at one accumulator and they either race (wrong) or serialize through an atomic (slow, worse than the CPU). The answer is a **tree reduction**, which sums pairs in parallel and folds the array in half each step, finishing in `log N` steps instead of `N`. Getting it fast then turns on two hardware facts: lay the additions out with **sequential addressing** so whole 32-lane warps stay active instead of half-idle, and reduce *within* a warp using **`__shfl_down_sync`**, which reads other lanes' registers directly with no shared memory. This post builds the reduction from a broken atomic up to a warp-shuffle kernel that is memory bound, limited only by how fast it reads the input.
>
> **After reading this you will be able to:**
> - Distinguish a race (wrong) from an atomic sum (correct but serial, even slower than the CPU).
> - Implement a shared-memory tree reduction and choose sequential over interleaved addressing.
> - Reduce a warp with `__shfl_down_sync` and compose warp, block, and grid levels.
> - Read a bandwidth benchmark and recognize when a kernel has become memory bound.

![A binary reduction tree: eight values sum pairwise into four, then two, then one final value, highlighted in green.](diagrams/01-tree-reduction.svg)
*Sum pairs in parallel and fold in half each step: `log N` parallel steps instead of `N` sequential adds.*

---

## 1. The motivation: a million threads, one answer

Matrix multiply *expanded* data; reduction *compresses* it: millions of numbers into one. It is everywhere, in a loss value, a softmax denominator, a mean or variance. On a CPU it is a one-line loop. On a GPU the obvious translation is a trap:

```cuda
*result += array[idx];   // a thousand threads racing over one address: garbage
```

Multiple threads reading and writing one location interleave unpredictably, a **race condition**. The textbook fix, `atomicAdd`, restores correctness by forcing threads to take turns, but turns the parallel algorithm back into a serial one: as the benchmark below shows, a pure-atomic sum is *slower than the CPU*. We need a structure that is correct **and** parallel.

## 2. The algorithm: tree reduction

Rather than every thread writing one place, arrange the additions as a **binary tree**. Sum adjacent pairs, then pairs of those, halving the count each step. The total work is still `N-1` additions, but the *depth* is `log N`, and each level is fully parallel. For 16 million elements that is about 24 steps instead of 16 million, the entire reason reduction is viable on a GPU.

## 3. Shared memory, and why addressing matters

The classic kernel loads a block's elements into shared memory, then folds the array in half `log(blockDim)` times. The subtle part is *which* threads stay active. The natural-looking version uses **interleaved** addressing, `if (tid % (2*stride) == 0)`, which activates threads 0, 2, 4, ... at the first step. Those active threads are *scattered*, so every 32-lane warp is half-idle, and idle lanes still occupy the warp: divergence at every level.

**Sequential** addressing, `if (tid < stride)` summing `tid` and `tid + stride`, keeps the active threads *contiguous*. A whole warp is then either fully active or fully idle, and divergence appears only in the last few sub-warp steps.

![Two panels: interleaved addressing with scattered active lanes (every warp half-idle, 47 divergent warp-steps) versus sequential addressing with a contiguous active half (5 divergent warp-steps).](diagrams/02-addressing.svg)
*Same adds, same result; sequential addressing just keeps each warp all-on or all-off.*

A CPU model counts the difference exactly over a 256-thread block, and confirms the tree result equals a plain sum. It lives at [snippets/reduction_model.py](snippets/reduction_model.py); run it with `python snippets/reduction_model.py`:

```text
reduction model (block of 256, warp 32)
  tree reduce == sum    : True
  parallel steps        : 8   (log2 of 256, vs 256 sequential adds)
  interleaved divergence: 47 divergent warp-steps
  sequential divergence : 5 divergent warp-steps
  -> sequential addressing keeps whole warps active; divergence only at the end
```

## 4. Warp shuffle: skip shared memory entirely

Shared memory is fast, but registers are faster, and since Kepler the GPU has let threads in a warp read each other's registers directly. **`__shfl_down_sync(mask, val, offset)`** returns lane `(laneId + offset)`'s copy of `val`. Halving the offset from 16 down to 1 reduces a full warp in five instructions, with no shared memory and no `__syncthreads()` inside the warp (the `_sync` in `__shfl_down_sync` orders the participating lanes itself).

![A warp's lanes reduced by reading registers offset 4, then 2, then 1; lane 0 ends with the sum, in green, no shared memory.](diagrams/03-warp-shuffle.svg)
*Five register-to-register shuffles collapse a 32-lane warp; lane 0 holds the total.*

```cuda
__device__ __forceinline__ float warpReduce(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);   // lane reads lane+off
    return v;                                         // lane 0 has the sum
}
```

The `0xffffffff` mask says all 32 lanes participate. On Volta and later you **must** pass the correct mask; the old maskless `__shfl_down` worked by accident on older hardware and silently breaks on modern GPUs.

## 5. Composing the levels: warp, block, grid

Real inputs have millions of elements, so the kernel works in three tiers. A **grid-stride loop** has each thread sum many elements into a register; **`warpReduce`** collapses each warp; the warp totals go through shared memory so **warp 0** can reduce them to a block total; and finally thread 0 of each block does one `atomicAdd` into the global result. Atomics are fine *here* because there are only as many as there are blocks (about a thousand), not one per element (sixteen million).

```cuda
__global__ void reduceWarpShuffle(const float* in, float* out, int N) {
    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) sum += in[i];   // grid-stride: any N, coalesced
    sum = blockReduce(sum);                    // warp shuffle + shared + warp 0
    if (threadIdx.x == 0) atomicAdd(out, sum); // one atomic per block
}
```

All three kernels live in [snippets/reduction.cu](snippets/reduction.cu), which benchmarks each one and then verifies the warp-shuffle total against a host sum. Build and run it (the GTX 1650 is compute capability 7.5, so `-arch=sm_75`):

```bash
nvcc -O3 -arch=sm_75 snippets/reduction.cu -o reduction && ./reduction
```

## 6. The numbers

On a GTX 1650 (compute capability 7.5, theoretical peak ~192 GB/s for its 128-bit GDDR6) summing 16M floats, the three approaches span more than three orders of magnitude. The shipped `reduction.cu` runs the three kernels and prints each time and bandwidth (`bytes / ms / 1e6` over the 64 MB input); the values below were measured separately on the GTX 1650 and the bandwidth in each row is just `64 MB / time`:

| Implementation | Time | Bandwidth | % of peak |
|---|---|---|---|
| naive atomic | 847.3 ms | 0.08 GB/s | 0.04% |
| shared memory | 0.62 ms | 108 GB/s | 56% |
| warp shuffle | 0.51 ms | 132 GB/s | 69% |

![A bandwidth bar chart: naive atomic 0.08 GB/s in red, shared memory 108, warp shuffle 132 in green, against the 192 GB/s peak line.](diagrams/04-reduction-perf.svg)
*The warp-shuffle kernel reaches ~69% of peak bandwidth: the algorithm is no longer the bottleneck, only the memory system is.*

The naive atomic is **slower than the CPU**, a vivid lesson in what serialization costs. The warp-shuffle kernel reaches about **69% of peak bandwidth**, which is the real goal of a reduction: make the arithmetic free and become **memory bound**, limited only by how fast you can read the input once.

---

## Common pitfalls

- **Reducing with a bare `+=` into global memory.** That is a race, not a reduction; the result is non-deterministic garbage. Use a tree, and reserve atomics for the few per-block partials.
- **Using `atomicAdd` per element.** Correct but catastrophic: a million threads queue on one address. Atomics are acceptable only once the count is down to roughly the number of blocks.
- **Interleaved addressing.** `tid % (2*stride) == 0` scatters active lanes and makes every warp diverge. Use `tid < stride` so active lanes are contiguous.
- **Omitting the shuffle mask on Volta+.** `__shfl_down(val, off)` without `_sync` and a mask silently returns wrong values under independent thread scheduling. Always `__shfl_down_sync(0xffffffff, val, off)`.
- **Forgetting to zero the output.** The result accumulates with `atomicAdd`, so a missing `cudaMemset(out, 0, ...)` makes every run wrong by the previous run's total.

---

## Further reading

- Harris, M., *"Optimizing Parallel Reduction in CUDA"* (NVIDIA, 2007). The classic slide deck that walks from interleaved to sequential addressing and beyond (technical, foundational).
- Luitjens, J., *"Faster Parallel Reductions on Kepler"* (NVIDIA, 2014). Where warp-shuffle reduction is introduced and benchmarked (technical).
- NVIDIA, *"CUDA C++ Programming Guide — Warp Shuffle Functions"* (current). The reference for `__shfl_down_sync` and participation masks (reference).
- NVIDIA, *"Using CUDA Warp-Level Primitives"* (Developer Blog, 2018). Why the `_sync` variants are mandatory after Volta (technical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 03, Matrix multiplication](../03-matrix-multiplication/index.md)**: the shared-memory tiling whose `__syncthreads()` discipline this post reuses.
- **[Post 05, Histogram and atomics](../05-histogram/index.md)**: what to do when threads must accumulate into *many* bins at once, where privatization replaces the single accumulator.
