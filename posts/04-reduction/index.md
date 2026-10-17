# 04 - Reduction: summing millions of numbers without a serial bottleneck

> **TL;DR.** Threads that all add into one address either race and lose updates or serialize through `atomicAdd`, which on the GTX 1650 is no faster than one central processing unit (CPU) thread. A **tree reduction** adds pairs in parallel and halves the count at every step, finishing in $\log_2 N$ steps instead of $N$. **Sequential addressing** keeps whole warps busy, a **grid-stride loop** gives each thread many elements, and **`__shfl_down_sync`** sums a warp in its registers. Together they read the 64 MiB input at about 97% of the card's theoretical peak bandwidth: the memory, not the arithmetic, sets the speed.
>
> **Prerequisites:** [post 03](../03-matrix-multiplication/index.md).
> **Safe to skip?** Skip it if the reader can already write a shared-memory tree reduction with sequential addressing, reduce a warp with `__shfl_down_sync` and say what its mask requires, and judge a reduction by its bandwidth as a share of the card's peak.
>
> **After reading, you will be able to:**
>
> - Distinguish a race from an atomic sum, and explain why one `atomicAdd` per element is correct but serial.
> - Implement a shared-memory tree reduction and count the divergent warp-steps of interleaved against sequential addressing.
> - Reduce a warp with `__shfl_down_sync` and state what its mask requires of the lanes that call it.
> - Compose warp, block, and grid levels with a grid-stride loop and one `atomicAdd` per block.
> - Read a bandwidth benchmark as a share of the card's theoretical peak and tell when a reduction has become memory bound.

![Eight values, 3 1 7 0 4 1 6 3, folded by sequential addressing: four threads give 7 2 13 3, two give 20 5, one gives the sum 25, in three steps.](diagrams/01-tree-reduction.svg)

*Each step halves the number of partial sums: eight values need three parallel steps, and 16,777,216 need 24.*

---

## 1. The question: a thousand threads, one answer

A **reduction** combines $N$ values into one: a sum, a maximum, a mean, as in a loss or a softmax denominator. On a CPU it is a loop with one running total. On a graphics processing unit (GPU) the obvious translation is this fragment, in which thread `i`, with `i = blockIdx.x * blockDim.x + threadIdx.x`, adds its element to one total:

```cuda
*result += array[i];   // thousands of threads read, add, and write one address: a race
```

Two threads that read the same old total both write it back, and one update is lost: a **race condition**, whose result depends on timing. The repair `atomicAdd(result, array[i])` makes each read-add-write indivisible, so nothing is lost, but the 16,777,216 additions of this post's input then happen one at a time on one address. Section 6 measures that kernel: it is no faster than one CPU thread running the plain loop. The question is how a thousand threads produce one sum that is both correct and parallel.

## 2. The tree: $\log_2 N$ steps instead of $N$

Arrange the additions as a **binary tree** instead: add pairs, then pairs of the results, halving the number of partial sums at every step. The work is still $N - 1$ additions, but the **depth**, the number of steps that must run one after another, falls to $\log_2 N$, and the additions inside one step are independent. The CPU model of this post, `snippets/reduction_model.py`, applies the rule the kernels use (at stride `s`, thread `t < s` adds element `t + s` into element `t`) to the eight values of the figure at the top:

```text
1. tree reduction of 8 values, sequential addressing
  start             : 3 1 7 0 4 1 6 3
  stride 4, 4 adds  : 7 2 13 3
  stride 2, 2 adds  : 20 5
  stride 1, 1 add   : 25
  sum 25 == plain sum 25: True, in 3 steps (log2 8)
2. depth of the tree against the additions
  N = 16777216: 24 steps, 16777215 additions
  one block of 256 threads: 8 steps, 255 additions
```

For 16,777,216 floats that is 24 steps instead of 16,777,215. Threads of different blocks cannot wait for each other inside a kernel, so each block reduces its own part in shared memory, and this post adds the block totals with one `atomicAdd` per block.

## 3. Shared memory, and why addressing matters

A block of 256 threads loads one element per thread into a shared buffer `s: [256]` and folds it in half eight times, with a `__syncthreads()` after every step so that no thread reads a partial sum before it is written. What matters is which threads do the adding. The natural first version uses **interleaved** addressing, whose step is this line of `reduceInterleaved`:

```cuda
if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];   // threads 0, 2, 4, ... first
```

Its active threads are scattered: at the first step every warp of 32 has 16 lanes working and 16 idle, and at stride 16 only lane 0 of each warp works. A warp with some lanes active and some idle is **divergent**: it issues the step for all 32 lanes and wastes the idle ones, as post 01 counted for a branch on `i % 2`. **Sequential** addressing keeps the active threads contiguous, as `reduceShared` in `snippets/reduction.cu` does:

```cuda
__global__ void reduceShared(const float* in, float* out, int N) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];   // the first half stays active
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}
```

With 128, 64 and 32 threads active, whole warps are fully busy or fully idle; a split warp appears only in the last five steps, when fewer than 32 threads remain. The model counts the split warps at each step of one block:

```text
3. one block of 256 threads (8 warps of 32): divergent warps per step
  step  interleaved: active  divergent   sequential: active  divergent
     1                  128          8                  128          0
     2                   64          8                   64          0
     3                   32          8                   32          0
     4                   16          8                   16          1
     5                    8          8                    8          1
     6                    4          4                    4          1
     7                    2          2                    2          1
     8                    1          1                    1          1
  total divergent warp-steps: interleaved 47, sequential 5
  tree of 256 random floats == plain sum: True
```

Both schemes do the same 255 additions and give the same sum up to rounding; the figure draws where their active lanes fall.

![Two grids of 8 steps by 8 warps for one 256-thread block. Interleaved addressing splits every warp at steps 1 to 5, 47 divergent warp-steps in all; sequential addressing splits only warp 0 at steps 4 to 8, 5 in all.](diagrams/02-addressing.svg)

*Same additions, same sum: sequential addressing keeps each warp all on or all off until fewer than 32 threads remain.*

The idea is Harris's ("Optimizing Parallel Reduction in CUDA", NVIDIA, 2007). His deck walks seven kernels on a G80: the second removes the divergence of the first (whose `%` he also calls slow) but brings shared-memory bank conflicts, and the third, sequential addressing, removes those; post 06 explains bank conflicts.

## 4. Warp shuffle: reduce a warp in its registers

Registers are faster than shared memory, and since the Kepler generation the lanes of a warp can read each other's registers directly (Luitjens, "Faster Parallel Reductions on Kepler", 2014). `__shfl_down_sync(mask, v, off)` returns the `v` of the lane `off` places above the caller; a lane whose source would be past lane 31 is handed its own value. Adding the returned value at offsets 16, 8, 4, 2 and 1 reduces a warp in five steps, with no shared memory and no barrier. Every lane adds at every step, but only lane 0 ends with the sum, and nothing reads the others:

```cuda
__device__ __forceinline__ float warpReduce(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);   // lane reads lane + off
    return v;                                         // lane 0 has the sum
}
```

The figure draws the same steps on eight lanes, and the model checks both widths:

```text
4. warp shuffle: lane l adds lane l + off
  8 lanes, off 4 2 1  : lane 0 = 25 (sum 25)
  32 lanes, off 16..1 : lane 0 = 496 (sum 496), 5 steps
```

![Eight lanes holding 3 1 7 0 4 1 6 3 in registers add the register of the lane 4, then 2, then 1 above; lane 0 ends with 25, the sum, and the other lanes' values are never read.](diagrams/03-warp-shuffle.svg)

*The hero's tree, run in registers: five shuffle-and-add steps reduce a full warp of 32 lanes.*

The first argument is a 32-bit **mask** naming the lanes that take part; `0xffffffff` names all 32. Lin and Grover ("Using CUDA Warp-Level Primitives", NVIDIA, 2018) state the rule: all the lanes in the mask must take part, and a full mask in code that some lanes never reach can hang the program. It holds here because every lane of every warp calls `warpReduce` outside any `if`. The older `__shfl_down`, without a mask, was deprecated in CUDA 9.0, and the CUDA 12.3 headers remove it for compute capability 7.0 and later, so it no longer compiles for the GTX 1650 (7.5).

## 5. Composing the levels: warp, block, grid

A real input has millions of elements, so the fast kernel works in three levels. A **grid-stride loop** (Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", 2013) has thread `i` add elements `i`, `i + stride`, `i + 2 * stride` and so on into a register, where `stride` is the number of threads in the grid; adjacent threads read adjacent addresses on every pass, so the loads stay coalesced. Then `blockReduce` runs `warpReduce` in every warp, lane 0 of each warp writes its total to `warpSums: [8]` in shared memory, and warp 0 reduces those eight values:

```cuda
__device__ __forceinline__ float blockReduce(float v) {
    __shared__ float warpSums[BLOCK / WARP];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warpReduce(v);
    if (lane == 0) warpSums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < BLOCK / WARP) ? warpSums[lane] : 0.0f;
        v = warpReduce(v);
    }
    return v;
}
```

The `__syncthreads()` makes every warp's total visible before warp 0 reads them, and thread 0 of each block adds the block's total to the result:

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

The program launches this kernel with 1,024 blocks of 256 threads, so one launch makes 1,024 atomic additions instead of 16,777,216, and each thread adds 64 elements before any tree runs: the tree's fixed cost, its barriers and its atomic, is paid once per 16,384 elements instead of once per 256. Harris's seventh kernel, "multiple elements per thread", applies the same idea. To measure that step alone, `reduction.cu` also runs `reduceSharedStride`, the tree of section 3 fed by the same loop.

## 6. The numbers

`reduction.cu` sums the same 16,777,216 floats (64 MiB, 67,108,864 bytes) with each kernel, takes the median of 21 timed launches after an untimed warm-up, zeroes the result before every launch outside the timed region, and checks every sum against a double-precision host sum. Bandwidth is the input bytes over the time, in decimal GB/s; the share of peak divides it by the 160 GB/s theoretical peak that post 01 derives from the 128-bit memory bus and the memory clock. The CPU row times the host loop, one thread, on the author's Intel Core i7-9750H. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
NVIDIA GeForce GTX 1650 with Max-Q Design, compute capability 7.5, theoretical peak 160 GB/s
N = 16777216 floats (67108864 bytes), 256 threads per block, median of 21 launches
method         blocks  time (ms)    GB/s  % peak   sum check
CPU, 1 thread       -     24.993     2.7    1.7%   reference 8388155.1
naive atomic    65536     30.234     2.2    1.4%   PASS, rel err 3e-05
interleaved     65536      2.818    23.8   14.9%   PASS, rel err 1e-07
sequential      65536      1.790    37.5   23.4%   PASS, rel err 7e-08
seq. + stride    1024      0.527   127.4   79.6%   PASS, rel err 2e-07
warp shuffle     1024      0.434   154.6   96.6%   PASS, rel err 2e-07
```

These numbers are measured; the reader's card will give other digits, and the order of the rows is the lesson. The figure draws the bandwidth column.

![Bandwidth in GB/s on the GTX 1650: CPU 2.7, naive atomic 2.2, interleaved 23.8, sequential 37.5, sequential with grid stride 127.4, warp shuffle 154.6, against the 160 GB/s peak.](diagrams/04-reduction-perf.svg)

*One fix per row: the tree, sequential addressing, many elements per thread, the warp shuffle; the last leaves only the memory to wait on.*

The per-element atomic is no faster than one CPU thread, and the least accurate: each of its 16,777,216 additions rounds a float total that ends near 8.4 million, where adjacent floats are 0.5 apart, while the trees round that total once per block. The interleaved tree is 10.7 times faster than the atomic, and sequential addressing 1.6 times faster again, the measured cost on this card of the divergent warp-steps and the interleaved kernel's `%` together. Of the fixes to the tree, the largest, 3.4 times, is the grid-stride loop. The warp shuffle cuts the remaining time by 18% and reaches 97% of the theoretical peak: the kernel spends its time waiting for bytes, which is what **memory bound** means, and no change to its arithmetic can speed it up. With one addition per 4-byte element, that is where a good reduction ends.

In a run logged with `nvidia-smi`, the streaming-multiprocessor (SM) clock of this laptop card moved between 1,140 and 1,740 MHz under a reported thermal slowdown, while the memory clock held at 5,000 MHz. Across today's quiet runs the tree rows varied by up to 1.7 times and the warp-shuffle row by under 2% (the CPU row, sharing the laptop with other programs, between 21.5 and 31.2 ms): a kernel limited by its instructions follows the SM clock, a memory-bound one the memory.

## 7. Make it run

The CPU model needs no GPU. From the series root:

```bash
uv run python posts/04-reduction/snippets/reduction_model.py
```

It prints the blocks quoted in sections 2 to 4 and the atomics per launch of each kernel. The kernel needs an NVIDIA GPU and the CUDA Toolkit, and builds from the series root with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 posts/04-reduction/snippets/reduction.cu -o reduction && ./reduction
```

On Windows the program is `reduction.exe`, and `nvcc` may need `-ccbin` to find `cl.exe`; [RUNNING.md](../../RUNNING.md) covers both. It prints section 6's table for the reader's card and exits non-zero if any sum fails its check; a busy or hot GPU gives slower tree rows, so run it twice.

## 8. What can go wrong?

- **The result is not zeroed before a launch.** Every kernel adds into `out`, so a second launch starts from the first one's total and the sum comes out about twice the reference. `reduction.cu` calls `cudaMemset(d_out, 0, sizeof(float))` before every launch.
- **A block size that is not a power of two.** The loop `stride = blockDim.x / 2; stride >>= 1` assumes one. With 192 threads the strides are 96, 48, 24, 12, 6, 3, 1; the step at stride 3 leaves three values and the step at stride 1 adds only two of them. The kernel runs and the sum is silently short.
- **A full mask where not every lane arrives.** Calling `warpReduce` inside the guard `if (i < N)` leaves the lanes past the end of the array out of a call whose mask names them; the result is undefined and the program may hang. Here every thread calls it, and those past the end contribute 0.
- **Legacy shuffles in old code.** A reduction written before CUDA 9.0 with `__shfl_down(v, off)` fails to build for `sm_75` with `identifier "__shfl_down" is undefined`; add the mask and the `_sync` suffix.
- **Harris's unrolled last warp.** His fifth kernel finishes the last 32 elements in `volatile` shared memory with no barrier, assuming a warp runs in lockstep. Lin and Grover call this implicit warp-synchronous programming unsafe: from compute capability 7.0 the lanes of a warp may run apart, and the sum can be wrong with no error.
- **Comparing a GPU sum with `==`.** A tree adds in a different order from the CPU loop, and the block totals arrive in a new order every run, so the last digits differ; `reduction.cu` checks a relative error under $10^{-4}$, and post 16 explains why.

---

## Common pitfalls

1. **Treating `+=` on one global address as a reduction.** It is a race: updates are lost and the result changes from run to run.
2. **One `atomicAdd` per element.** Correct but serial, and no faster than one CPU thread on the GTX 1650; atomics are cheap only when there are about as many as there are blocks.
3. **Interleaved addressing.** `tid % (2 * stride) == 0` splits every warp at the first five steps; `tid < stride` keeps the active threads contiguous.
4. **One element per thread.** A kernel with one thread per element pays a whole tree, barriers and atomic included, for every 256 elements; adding many elements in a register first was the largest gain after the tree itself in section 6.
5. **Judging a reduction by its time alone.** The measure is its bandwidth against the card's peak; near the peak, only reading fewer bytes helps.

---

## Further reading

- Harris, M., "Optimizing Parallel Reduction in CUDA" (NVIDIA, 2007). Seven kernels, from interleaved addressing to multiple elements per thread, measured on a G80.
- Luitjens, J., "Faster Parallel Reductions on Kepler" (NVIDIA Technical Blog, 13 February 2014). Warp shuffle, with the warp, block and atomic structure of section 5.
- Lin, Y. and Grover, V., "Using CUDA Warp-Level Primitives" (NVIDIA Technical Blog, 15 January 2018). The `_sync` primitives, their masks, and why implicit warp-synchronous code is unsafe.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Warp Shuffle Functions", consulted October 2026. The reference for `__shfl_down_sync`.
- Harris, M., "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops" (NVIDIA Technical Blog, 22 April 2013). The loop of section 5.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 05 - Histogram: beating atomic contention with privatization](../05-histogram/index.md):** what to do when threads add into many bins at once, where a private copy per block replaces the single total.
- **[Post 08 - Parallel scan: breaking the dependency chain of a prefix sum](../08-parallel-scan/index.md):** runs a reduction tree, in interleaved order and in place, as the up-sweep of a scan, which keeps every running total instead of only the last.
