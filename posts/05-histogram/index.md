# 05 · Histogram — beating atomic contention with privatization

> **TL;DR.** A histogram counts how often each value appears, so many threads increment the same counters at once: a plain `bins[val]++` is a **race** (lost updates), and `atomicAdd` fixes correctness but serializes the threads that hit the *same* bin into a single-lane queue. On "boring" data like a flat image where every pixel is one value, a global-atomic histogram collapses to **slower than a CPU**. The fix is **privatization**: give each block its own histogram in shared memory, count there with fast local atomics, then merge each block's copy into the global bins once. This keeps the heavy contention on-chip and cuts the global hot-bin contention from millions to a few hundred, staying fast no matter how skewed the data is.
>
> **After reading this you will be able to:**
> - Explain the lost-update race and why `atomicAdd` is correct but contention-bound.
> - Predict when a global-atomic histogram collapses, from the data distribution.
> - Write a privatized histogram with a per-block shared array and a three-phase init/count/merge.
> - Place the two `__syncthreads()` barriers correctly and say what each protects.

![Eight threads all reading value 128 issue atomic increments to the same global bin, serializing into a queue, shown in red.](diagrams/01-contention.svg)
*Atomics make the count correct; contention on one bin makes it serial.*

---

## 1. The motivation: counting is a contention problem

Map worked independently, reduce cooperated without conflict. A histogram is the first pattern where threads genuinely *contend*: to count pixel intensities, thousands of threads increment a shared set of bins. The naive increment is a classic **race**: two threads read `bins[v]` as 5, both compute 6, both write 6, and one increment is lost. `atomicAdd(&bins[v], 1)` makes the read-modify-write indivisible and restores correctness. But correctness is not the whole story, because an atomic on a contended address is *serial*.

## 2. The naive kernel, and how it collapses

The direct version points every thread at one global bins array:

```cuda
__global__ void histGlobal(const int* in, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&bins[in[i]], 1);     // global atomic, ~400 cycles
}
```

On random data the increments spread across 256 bins, contention is low, and it manages about 45 GB/s (all bandwidth numbers here are measured on a GTX 1650). But on a flat image where every pixel is 128, *every* thread targets `bins[128]`. The atomics queue up on one global address, each paying ~400 cycles, and throughput falls to **0.8 GB/s**, far slower than a CPU. A CPU model makes the worst case concrete, counting 16,777,216 increments serializing on one address. Running `snippets/histogram_model.py` prints:

```text
histogram model (solid-color worst case)
  N                       = 16,777,216 elements, all value 128
  privatized == reference : True
  global atomics on bin128 : 16,777,216  (all serialize on one address)
  shared atomics / block   : 65,536  (fast, ~20 cycles, stays local)
  global merge atomics     : 65,536  (256 blocks x 256 bins)
  global contention on bin128 drops from 16,777,216 to 256
```

The shipped `snippets/histogram.cu` runs the privatized `histShared` kernel (it checks the count is correct); reproducing the global-atomic collapse above means swapping in `histGlobal`. Build and run both pieces with:

```bash
nvcc -O3 -arch=sm_75 snippets/histogram.cu -o histogram && ./histogram
python snippets/histogram_model.py
```

## 3. The fix: privatization

The trick is to stop every thread fighting over global memory. Give **each block its own private histogram in shared memory**, count there, and merge once at the end. The kernel has three phases, and the two barriers between them are not optional.

![Three blocks each with a private green shared histogram; threads count locally, then each block merges into the single global bins array with one atomic per bin.](diagrams/02-privatization.svg)
*The contention moves into fast shared memory; global memory sees only a handful of atomics per block.*

```cuda
__global__ void histShared(const int* in, int* bins, int n) {
    __shared__ int s[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) s[b] = 0;  // 1. zero
    __syncthreads();                          // all bins clean before counting

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&s[in[i]], 1);              // 2. count: shared atomic, ~20 cycles
    __syncthreads();                          // all counting done before merge

    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        atomicAdd(&bins[b], s[b]);            // 3. merge: one global atomic per bin
}
```

Two things make this fast. Shared-memory atomics cost about **20 cycles** versus ~400 for global, and the streaming multiprocessor (SM, the GPU core that runs a block) handles them locally. And the contention scope shrinks: even in the worst case the hottest *global* bin is touched only once per block (`gridDim` times), because each block contributes a single merged increment. The model above shows that drop from 16,777,216 to 256.

## 4. The numbers

The payoff is not just average speed but **robustness**: privatization barely changes between easy and worst-case data, while global atomics fall apart.

| Implementation | Random data | Solid color |
|---|---|---|
| global atomics | 45 GB/s | 0.8 GB/s |
| shared (privatized) | 180 GB/s | 120 GB/s |

Measured on a GTX 1650 (compute capability 7.5). The shipped `snippets/histogram.cu` only checks the count is correct; these bandwidths were timed separately on that GPU.

![A grouped bar chart: global atomics 45 and shared 180 on random data, then global collapsing to 0.8 while shared holds 120 on solid color.](diagrams/03-histogram-perf.svg)
*Global atomics fall 56× from random to solid color; privatization changes by under 2×.*

Privatization is about **4× faster** on random data and roughly **150× faster** in the worst case. The lesson generalizes past histograms: when threads must accumulate into shared state, push the contention into the smallest, fastest scope you can, and touch global memory as few times as possible.

## 5. Beyond 256 bins

Privatization works because 256 `int` bins fit in 1 KB of shared memory. At 4096 bins (16 KB) it starts to cut occupancy, and 65536 bins (256 KB) will not fit at all. Two escape hatches: process the range in **multiple passes** (bins 0–255, then 256–511, ...), or use **warp aggregation**, where lanes (the threads within a warp) that share a value combine with `__match_any_sync` (a warp intrinsic that returns a mask of which lanes hold the same value) and elect one leader to do a single atomic of the group's count. Both keep the contention bounded when the bin array itself is too large to privatize.

---

## Common pitfalls

- **Not zeroing shared memory.** `__shared__` arrays start as garbage; counting into them without an init loop gives garbage out. Zero collaboratively, then synchronize.
- **Missing the first `__syncthreads()`.** Without a barrier after zeroing, a thread can start counting before another finishes clearing its bins, corrupting the count.
- **Missing the second `__syncthreads()`.** Merging before every thread has finished counting writes partial results into global memory.
- **Forgetting to zero the global bins on the host.** `atomicAdd` accumulates, so a missing `cudaMemset` adds this run's counts on top of the last run's.
- **Integer overflow.** A single block counting more than ~2 billion items overflows a 32-bit bin; use `unsigned long long` counters (64-bit integer `atomicAdd` is available on all modern GPUs) for very large inputs.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — Atomic Functions"* (current). The reference for `atomicAdd` and shared-memory atomics (reference).
- NVIDIA, *"CUDA C++ Best Practices Guide — Shared Memory"* (current). Where the privatization pattern and occupancy trade-offs are discussed (technical).
- Gómez-Luna, J. et al., *"An Optimized Approach to Histogram Computation on GPU"* (Machine Vision and Applications, 2013). A study of privatization and contention (technical).
- Adinetz, A., *"CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics"* (NVIDIA, 2014). The warp-aggregation technique for large bin counts (technical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 04, Reduction](../04-reduction/index.md)**: the other side of atomics, where a tree plus a few atomics beats one-atomic-per-thread.
- **[Post 06, Matrix transpose](../06-matrix-transpose/index.md)**: where shared memory solves a *coalescing* problem instead of a contention one.
