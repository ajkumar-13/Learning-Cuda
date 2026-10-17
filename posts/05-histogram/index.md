# 05 - Histogram: beating atomic contention with privatization

> **TL;DR.** A histogram makes thousands of threads increment the same few counters, so a plain `bins[v]++` loses updates, and `atomicAdd` restores the count by making threads that hit one address take turns. On the author's GTX 1650 a histogram of 16,777,216 values built with global atomics runs at 13.5 GB/s on random data and 6.6 GB/s on a solid color, under a tenth of the card's 160 GB/s theoretical peak. **Privatization** gives each block its own histogram in shared memory and merges it once per bin, which makes the same count 8.7 times faster on random data and 6.5 times faster on the solid color.
>
> **Prerequisites:** [post 04](../04-reduction/index.md).
> **Safe to skip?** Skip it if the reader can already write the zero, count, merge histogram kernel, place its two barriers, and say why it beats global atomics.
>
> **After reading, you will be able to:**
>
> - Explain the lost-update race of a plain increment, and why `atomicAdd` is correct but makes threads that hit one address take turns.
> - Predict from the data distribution how many increments queue on the hottest address under global atomics and under privatization.
> - Write a privatized histogram with a per-block shared array in three phases: zero, count, merge.
> - Place the two `__syncthreads()` barriers of the privatized kernel and say what each one protects.
> - Say when a private histogram no longer fits in shared memory, and name the two ways around it: multiple passes and warp aggregation.

![Eight threads each issue one atomicAdd on the same global bin and take turns over eight illustrative time slots while the count steps from 1 to 8; a table gives the model's counts for a solid color: 16,777,216 atomics on the hottest global bin, against 65,536 per block's shared bin and 256 global merges with privatization.](diagrams/01-contention.svg)

*Atomics make the count right; contention on one address makes the threads take turns.*

---

## 1. The question: counting is a contention problem

In [post 02](../02-vector-addition/index.md) every thread wrote its own element. A histogram is the first pattern where threads genuinely *contend*: to count how often each value from 0 to 255 appears in an image, thousands of threads increment one array of 256 bins, and any two threads that read the same value increment the same bin.

The naive increment `bins[v]++` is three steps: read the count, add 1, write it back. When two threads interleave them, both read the same old count and one increment is lost. The central processing unit (CPU) model of this post, `snippets/histogram_model.py`, replays the two orders:

```text
1. the race: two threads run bins[v]++ on a count of 5
  one after the other        : 7
  both read, then both write : 6  (one increment lost)
```

On the graphics processing unit (GPU) the loss is massive. `snippets/histogram.cu` carries a kernel, `histRacy`, that does exactly this over 16,777,216 values:

```cuda
__global__ void histRacy(const int* in, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        bins[in[i]]++;                        // read, add 1, write back: not indivisible
}
```

Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, it kept 617,749 of the 16,777,216 increments on random data and 6,451 on a solid color (section 4 shows the run). More than 96% vanished.

The fix for correctness is an **atomic** operation: `atomicAdd(&bins[v], 1)` performs the read, the add and the write as one indivisible step. Indivisible has a price: two atomics on one address cannot overlap, so the threads that hit it are served one after another, as the figure at the top draws for eight threads. The count ends right, at 8, after eight turns.

## 2. The naive kernel, and the hottest address

The direct version points every thread at one array of bins in global memory:

```cuda
__global__ void histGlobal(const int* in, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&bins[in[i]], 1);           // one global atomic per element
}
```

Thread `i` starts at the global index `blockIdx.x * blockDim.x + threadIdx.x` and strides by the size of the grid, `blockDim.x * gridDim.x`; the **grid-stride loop** of [post 04](../04-reduction/index.md) lets a fixed launch cover any `N`. The launch throughout is 256 blocks of 256 threads, 65,536 threads, so for `N = 16,777,216` each thread handles 256 elements. The input is `in: [N]`, 4-byte `int` values from 0 to 255, and the output `bins: [256]`.

How long the queue on one address grows depends on the data. In **random** data every value is equally likely, so each bin receives about $N / 256 = 65{,}536$ increments. In a **solid color**, the flat image of a white wall, every value is 128, so all 16,777,216 increments target `bins[128]`. The model counts the increments on the hottest address for both inputs and both kernels:

```text
2. N = 16,777,216 values into 256 bins, <<<256, 256>>>, 256 elements per thread
                                               random       solid
  privatized count equals the reference          True        True
  histGlobal: atomics on the hottest bin       66,595  16,777,216
  histShared: hottest shared bin, 1 block         323      65,536
  histShared: merges on any global bin            256         256
  histShared: merge atomics in all             65,536      65,536
```

These counts are derived by the model (its random column uses NumPy's generator with seed 0); the `histShared` rows belong to the next section. They say which queue is longest, not what a turn costs, and random data is not free of contention either: the 256 bins occupy 1 KB, so all 16,777,216 atomics converge on one kilobyte.

## 3. The fix: privatization

Privatization gives **each block its own private histogram in shared memory**, counts there, and merges the private copy into the global bins once at the end. The kernel has three phases and two barriers:

```cuda
__global__ void histShared(const int* in, int* bins, int n) {
    __shared__ int s[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) s[b] = 0;   // 1. zero
    __syncthreads();                          // every bin is zero before any thread counts

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&s[in[i]], 1);              // 2. count: shared-memory atomic
    __syncthreads();                          // every thread has counted before any bin is merged

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&bins[b], s[b]);            // 3. merge: one global atomic per bin per block
}
```

`BINS` is 256, and `s: [256]` is the block's private histogram, 1 KB of the 48 KB of shared memory a block may use on the GTX 1650. The threads zero `s` together, count their elements into it with shared-memory atomics, and then each adds one bin to the global bins, so a block issues exactly 256 global atomics whatever its data. Each barrier protects one hand-over:

- **The first `__syncthreads()`** makes every bin zero before any thread counts. Without it a fast thread can increment `s[128]` before the thread responsible for that bin has cleared it, and the clear wipes the increment out.
- **The second `__syncthreads()`** makes every thread finish counting before any bin is merged. Without it a thread can add `s[b]` to global memory while other threads are still incrementing `s[b]`, and their late increments never arrive.

Both barriers hold one block only, which is enough: no block reads another's shared memory, and the merge is made safe by its atomics.

![Two blocks, block 0 and block 255, each with a private shared histogram drawn as bins 0, 128 and 255. In phase 1 the threads zero it. After a barrier, in phase 2, each block counts its 65,536 solid-color values, so its bin 128 holds 65,536. After a second barrier, in phase 3, each block adds its bins to the global bins with one atomic per bin, and global bin 128 receives 256 merges and ends at 16,777,216.](diagrams/02-privatization.svg)

*The long queue moves into each block's shared memory, where blocks count in parallel; global memory sees one merge per bin per block.*

Two things make the private count cheaper. A shared-memory atomic is resolved on the streaming multiprocessor (SM, the unit that runs a block) without a trip to global memory; Sakharnykh's 2015 NVIDIA Technical Blog post records that hardware shared-memory atomics arrived with Maxwell, while Kepler "emulates shared memory atomics in software". And the queue shrinks and splits: on the solid color a block's hottest shared bin receives 65,536 increments, a queue 256 times shorter than the global one, and the 14 SMs of the GTX 1650 work through separate queues at once. The hottest *global* bin receives 256 merges, whatever the data. The figure draws the three phases on the solid color.

## 4. The numbers

`snippets/histogram.cu` times both kernels on both inputs with CUDA events. It first runs `histGlobal` 100 times untimed, because a laptop GPU idles at a low clock and needs a few hundred milliseconds of work to reach its 1,740 MHz boost clock; without that warm-up the first cell timed came out up to 1.7 times slower. Each cell then gets one untimed warm-up launch and 20 timed launches, with the global bins zeroed before every launch outside the timed interval and every run's bins compared with a histogram counted on the host. The random input comes from a fixed linear congruential generator. Bandwidth is the $4 \times 16{,}777{,}216$ bytes of input divided by the average time. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5)
N = 16777216 ints, 256 bins, <<<256, 256>>>, average of 20 timed runs after warm-up
data    kernel       time (ms)    GB/s   bins match host
random  histGlobal       4.967    13.5   yes
random  histShared       0.573   117.2   yes
solid   histGlobal      10.193     6.6   yes
solid   histShared       1.557    43.1   yes
histRacy (no atomic), random data: counted 617749 of 16777216 increments
histRacy (no atomic), solid data: counted 6451 of 16777216 increments
```

| Kernel | Random data | Solid color |
|---|---|---|
| `histGlobal`, global atomics | 4.967 ms, 13.5 GB/s | 10.193 ms, 6.6 GB/s |
| `histShared`, privatized | 0.573 ms, 117.2 GB/s | 1.557 ms, 43.1 GB/s |

Every number in the table is measured on one card; the reader's card will give other digits, and the shape of the table is the lesson. The output is one of five runs, each started on an idle GPU; the five agreed to within 0.003 ms in every cell.

![Two bar charts on one scale of 0 to 160 GB/s, with a dotted line at the 160 GB/s theoretical peak. Random data: global atomics 13.5 GB/s, privatized 117.2 GB/s. Solid color: global atomics 6.6 GB/s, privatized 43.1 GB/s.](diagrams/03-histogram-perf.svg)

*Privatization wins on both inputs: 8.7 times on random data, 6.5 times on the solid color.*

Three readings follow from the table and the chart. First, privatization is $4.967 / 0.573 = 8.7$ times faster on random data and $10.193 / 1.557 = 6.5$ times on the solid color, and on random data it reads at 73% of the 160 GB/s theoretical peak.

Second, global atomics are slow on *both* inputs on this card, 4% to 8% of the peak, and the solid color costs `histGlobal` $10.193 / 4.967 = 2.1$ times its random-data time, not the collapse that one hot address might suggest. On one address the card completes 16,777,216 atomics in 10.193 ms, 1.65 billion a second, close to one per cycle of its 1,740 MHz boost clock; NVIDIA's Kepler GK110/210 whitepaper (2014) gives that rate, one operation per clock, for atomics to a common global address on the older architecture, and NVIDIA publishes no such figure for Turing. Random data spreads the atomics over 256 addresses, but all 256 share one kilobyte, and the card completes only about twice as many per second. The model's prediction holds: under `histGlobal` every element takes a turn in global memory, under `histShared` only 256 merges per bin do.

Third, privatization is not immune to skew: on the solid color it takes 2.7 times as long, because each block's threads queue on `s[128]`. It does not remove contention; it moves it into a cheaper scope, split 256 ways. The rule reaches past histograms: keep contended updates in the smallest, fastest scope, and touch global memory as few times as possible.

## 5. Beyond 256 bins

Privatization works here because 256 `int` bins fit easily in shared memory. The model's last table sets the private copy against the 48 KB a block may use on the GTX 1650:

```text
3. the private histogram's size in shared memory (4-byte int bins; 48 KB per block on the GTX 1650)
     256 bins =   1 KB  fits
   4,096 bins =  16 KB  fits
  65,536 bins = 256 KB  does not fit
```

At 4,096 bins the copy fits, but 16 KB per block limits how many blocks an SM holds at once ([post 10](../10-roofline/index.md) counts this as occupancy); at 65,536 bins it does not fit. Two ways around it:

- **Multiple passes.** Run the privatized kernel once per range of bins (0 to 4,095, then 4,096 to 8,191, and so on), each pass counting only the values in its range. Every pass reads the whole input again.
- **Warp aggregation.** Combine a warp's increments before they reach memory, so that one lane issues one atomic for the group. Adinets's 2014 NVIDIA Technical Blog post shows it for one counter: an elected leader adds the count of the warp's active lanes. Its 2017 update notes that the compiler now does this by itself in many cases; for `histGlobal`, whose address differs from lane to lane, the compiled code (`cuobjdump -sass`) still issues one atomic per element. For many bins the lanes must first find which others hold the same value, which `__match_any_sync` returns as a bit mask on compute capability 7.0 and later.

Gómez-Luna, González-Linares, Benavides and Guil (2013) take privatization further, replicating the histogram several times inside each block so that fewer threads share each copy, and padding the copies against bank conflicts, the shared-memory hazard of [post 06](../06-matrix-transpose/index.md).

## 6. Make it run: the model and the kernel

The CPU model needs NumPy and no GPU. From the series root:

```bash
uv run python posts/05-histogram/snippets/histogram_model.py
```

It prints the tables of sections 1, 2 and 5. The kernel needs an NVIDIA GPU and the CUDA Toolkit, and builds with one `nvcc` line from the series root, `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 posts/05-histogram/snippets/histogram.cu -o histogram && ./histogram
```

On Windows the program is `histogram.exe`, and if `nvcc` cannot find the host compiler it needs a `-ccbin` argument; [RUNNING.md](../../RUNNING.md) covers both and the `-arch` value for other cards. It prints the output of section 4 in under two seconds on the GTX 1650 and exits with status 1 if any kernel's bins disagree with the host count.

## 7. What can go wrong?

Each mistake below was reproduced on the author's GTX 1650 with a one-line change to `histogram.cu`, on its random input unless stated.

- **Forgetting to zero the shared array.** `__shared__` memory starts with whatever the last block on that SM left there. With phase 1 deleted, after a kernel that had filled shared memory, the bins summed to 48,413,184 instead of 16,777,216.
- **Dropping the second barrier.** Three runs summed to 16,712,899, 16,712,918 and 16,709,226: tens of thousands of late increments lost, a different number each time.
- **Dropping the first barrier.** Three runs still gave the right total, since each thread clears one bin and counts at once; the race remains, and a passing test does not prove a barrier unnecessary.
- **Not zeroing the global bins between runs.** `atomicAdd` adds to whatever is there: two launches without a `cudaMemset` between them counted 33,554,432, exactly twice the input.
- **Overflowing a 32-bit bin.** The bins are signed `int`, whose largest value is $2^{31} - 1 = 2{,}147{,}483{,}647$. Accumulating 128 solid-color launches, a true count of 2,147,483,648, left `bins[128]` at -2,147,483,648 without an error. An `unsigned int` bin holds up to $2^{32} - 1 = 4{,}294{,}967{,}295$; beyond that, `unsigned long long int` bins with the 64-bit `atomicAdd` are the fix.

## 8. Summary

A plain increment loses updates; `atomicAdd` makes the count correct by making threads that hit one address take turns, and global atomics run every element through that queue, 6.6 to 13.5 GB/s on the GTX 1650. Privatization counts in a per-block shared array, zero, count, merge, with a barrier after each of the first two phases, and turns the global traffic into 256 merges per bin: 8.7 times faster on random data and 6.5 times on a solid color, measured.

---

## Common pitfalls

1. **Reading "correct" as "fast".** `atomicAdd` guarantees that no update is lost and nothing about speed: the same correct count took 0.573 ms or 4.967 ms in section 4.
2. **Benchmarking on random data only.** Real images have large flat regions; time the kernel on the skewed input it will meet, as the solid-color column does.
3. **Launching one thread per element.** Every block merges 256 bins, so the 65,536 blocks of such a launch would merge $65{,}536 \times 256 = 16{,}777{,}216$ global atomics, as many as the naive kernel; a fixed grid of 256 blocks with a grid-stride loop merges 65,536 for any `N`.
4. **Privatizing too many bins.** A private copy that fills most of the 48 KB of shared memory per block limits how many blocks run at once; split the bins into passes instead.
5. **Expecting an atomic to order other memory.** An atomic makes one read-modify-write indivisible and orders nothing else, the subject of [post 09](../09-synchronization/index.md).

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, sections "Atomic Functions" and "Warp Match Functions", consulted October 2026. The atomics and their types, including the 64-bit `atomicAdd`, and the warp match functions.
- NVIDIA, *CUDA C++ Best Practices Guide*, section "Shared Memory", consulted October 2026. Shared memory as a block's scratchpad, and its cost in occupancy.
- Sakharnykh, N., "GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell" (NVIDIA Technical Blog, 17 March 2015). Global against shared atomics for histograms of random and real images.
- NVIDIA, *NVIDIA's Next Generation CUDA Compute Architecture: Kepler GK110/210* (whitepaper V1.1, 2014), section "Atomic Operations". Atomic throughput to a common global address, one operation per clock on Kepler.
- Adinets, A., "CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics" (NVIDIA Technical Blog, 1 October 2014). Warp aggregation for a counter shared by a warp.
- Gómez-Luna, J., González-Linares, J. M., Benavides, J. I. and Guil, N., "An Optimized Approach to Histogram Computation on GPU" (*Machine Vision and Applications* 24(5), 2013). Replicated per-block histograms and padding against bank conflicts.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 06 - Matrix transpose: coalescing, shared memory, and bank conflicts](../06-matrix-transpose/index.md):** the other use of shared memory in Part II, where it fixes a coalescing problem instead of a contention one.
- **[Post 09 - When threads share data: barriers, atomics, and the memory model](../09-synchronization/index.md):** the exact guarantees of `__syncthreads()` and of atomics that this post relied on, and what neither of them guarantees.
