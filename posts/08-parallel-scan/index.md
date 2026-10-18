# 08 - Parallel scan: breaking the dependency chain of a prefix sum

> **TL;DR.** A scan (prefix sum) turns `[3, 1, 7, 0]` into its running totals `[3, 4, 11, 11]`, and every total seems to wait for the one before it. **Hillis-Steele** and **Blelloch** break that chain with more additions than the loop but only $\log_2 N$ or $2 \log_2 N$ parallel steps, and a **three-phase** scheme stretches either over an array larger than a block. On the author's GTX 1650 the hand-written scans of 16,777,216 floats take 2.3 to 4.8 ms, while NVIDIA's CUB library takes 0.96 ms, as fast as a plain copy, because it reads and writes each element once. Scan's payoff is **stream compaction**: an exclusive scan of a keep-or-drop flag gives every kept element its output position.
>
> **Prerequisites:** [post 04](../04-reduction/index.md).
> **Safe to skip?** Skip it if the reader can already walk the Blelloch up-sweep and down-sweep by hand, remove its bank conflicts, scan an array larger than a block, and build stream compaction from a scan.
>
> **After reading, you will be able to:**
>
> - Explain why a prefix sum looks serial and how trading extra additions for fewer parallel steps breaks the chain.
> - Count the additions and parallel steps of Hillis-Steele and Blelloch, and pick one for a warp, a block, or a whole array.
> - Walk the Blelloch up-sweep and down-sweep by hand on eight values and say what each phase computes.
> - Count the bank conflicts of the Blelloch tree indices and remove them with the offset `index + (index >> 5)`.
> - Scan an array larger than one block in three phases, and explain why that moves twice the bytes of a single-pass library scan.
> - Implement stream compaction as predicate, exclusive scan, and scatter.

![Left: the serial scan of 3 1 7 0 4 1 6 3 as a red chain of 7 dependent additions giving 3 4 11 11 15 16 22 25. Right: Hillis-Steele reaching the same totals in 3 parallel steps of stride 1, 2 and 4.](diagrams/01-scan-dependency.svg)

*The serial loop is a chain of seven additions; Hillis-Steele reaches the same totals in three steps by letting every element work at once.*

---

## 1. The question: a running total that waits on itself

After the map of [post 02](../02-vector-addition/index.md) and the reduction of [post 04](../04-reduction/index.md), the **scan** is the third basic shape of data-parallel work: $N$ inputs give $N$ outputs, each the total of everything up to it. An **inclusive** scan counts the element itself; an **exclusive** scan stops just before it, so its first output is 0, the identity of addition. On a central processing unit (CPU) the scan is one loop, `out[i] = out[i - 1] + in[i]`, and every addition reads the result of the one before it: a chain of $N - 1$ steps. The CPU model of this post, `snippets/scan_model.py`, runs the loop on the eight values that `scan.cu` also uses:

```text
1. the serial scan: out[i] = out[i - 1] + in[i], a chain of 7 additions
  input           :  3  1  7  0  4  1  6  3
  inclusive       :  3  4 11 11 15 16 22 25
  exclusive       :  0  3  4 11 11 15 16 22
```

How can a thousand threads produce all the totals when total 999 seems to need total 998 first? By trading work for depth: as many additions as the loop or more, arranged so that the additions inside one step are independent. The figure at the top draws the loop's chain beside the three steps of the first parallel algorithm.

## 2. Hillis-Steele: more additions, fewer steps

The **Hillis-Steele** scan (Hillis and Steele, "Data Parallel Algorithms", 1986) is the direct one. At each step, with a stride doubling from 1, every element adds the element `stride` places before it; after $\log_2 N$ steps each element holds the sum of all the inputs up to it. The model prints the three steps for eight values:

```text
2. Hillis-Steele: at stride s, element t adds element t - s
  stride 1, 7 adds :  3  4  8  7  4  5  7  9
  stride 2, 6 adds :  3  4 11 11 12 12 11 14
  stride 4, 4 adds :  3  4 11 11 15 16 22 25
  equals the inclusive scan: True, in 3 steps
```

On the graphics processing unit (GPU) thread `t` of a block owns element `t`, and every thread must read the old values of a step before any thread overwrites them. `hillisSteele` in `snippets/scan.cu` keeps two copies of the block's array in shared memory and alternates, reading one half and writing the other:

```cuda
__global__ void hillisSteele(const float* in, float* out, float* sums, int n) {
    extern __shared__ float buf[];   // 2 * blockDim.x floats: two ping-pong halves
    int m = blockDim.x, t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pin = 1, pout = 0;           // the first swap makes half 0, which holds the input, the one read
    buf[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < m; stride *= 2) {
        pout = 1 - pout; pin = 1 - pin;
        buf[pout * m + t] = buf[pin * m + t] + (t >= stride ? buf[pin * m + t - stride] : 0.0f);
        __syncthreads();
    }
    if (i < n) out[i] = buf[pout * m + t];                   // inclusive scan
    if (sums && t == m - 1) sums[blockIdx.x] = buf[pout * m + t];   // the block's total
}
```

Thread `i`, with `i = blockIdx.x * blockDim.x + threadIdx.x`, loads its element into half 0, which the first swap makes the half read first; the `__syncthreads()` after each step publishes its writes before the next step reads them. The last line saves the block's total for section 5.

The price is work. Every thread adds at every step, so $N$ elements cost $N \log_2 N$ additions, of which $N \log_2 N - (N - 1)$ are useful (threads with `t < stride` add 0). The model counts them against the loop and against the second algorithm:

```text
4. work (additions) and depth (parallel steps)
  N                    serial adds    Hillis-Steele adds   useful of those   Blelloch adds   depth S / HS / B
   1,048,576 (2^20)      1,048,575            20,971,520        19,922,945       2,097,150   1,048,575 / 20 / 40
  16,777,216 (2^24)     16,777,215           402,653,184       385,875,969      33,554,430   16,777,215 / 24 / 48
  at N = 2^20 Hillis-Steele adds 10.0 times as often as Blelloch (9.5 times counting only the useful adds)
```

Where the extra additions cost depends on where the scan runs. Inside a **warp** they cost almost nothing, since its 32 lanes run each step as one instruction anyway, and `__shfl_up_sync` lets lane `l` read the register of lane `l - off` with no shared memory and no barrier:

```cuda
__device__ __forceinline__ float warpInclusiveScan(float v) {
    int lane = threadIdx.x % WARP;
    for (int off = 1; off < WARP; off *= 2) {
        float up = __shfl_up_sync(0xffffffff, v, off);   // the value of lane - off
        if (lane >= off) v += up;
    }
    return v;
}
```

A lane whose source lies below lane 0 gets its own value back, hence the guard `lane >= off`; the full mask requires every lane to call, the `_sync` rule of post 04. The kernel `warpScan` scans each warp of a 1,024-thread block this way, has warp 0 scan the 32 warp totals, and adds to each thread the totals of the warps before it. Across a whole array the extra additions are not free: a Hillis-Steele scan of 16,777,216 floats in global memory, one launch per step, would read and write the array 24 times.

## 3. Blelloch: an up-sweep and a down-sweep

The **Blelloch** scan (Blelloch, "Prefix Sums and Their Applications", 1990) does $2(N - 1)$ additions in two passes over a tree. The **up-sweep** is the reduction tree of post 04 run in place: at offsets 1, 2, 4, the right element of each pair adds the left one, and the last element ends with the total. The root is then cleared to 0. The **down-sweep** walks back down: at each node the left child takes the parent's value and the right child the parent's value plus the left child's old value, so every element ends with the sum of everything before it, an exclusive scan:

```text
3. Blelloch: up-sweep, clear the root, down-sweep
  after up-sweep  :  3  4  7 11  4  5  6 25   (the last element is the total, 25)
  root cleared    :  3  4  7 11  4  5  6  0
  after down-sweep:  0  3  4 11 11 15 16 22
  equals the exclusive scan: True, in 6 steps
```

The figure draws every level of both sweeps.

![Blelloch on 3 1 7 0 4 1 6 3: three up-sweep levels reach 3 4 7 11 4 5 6 25, the root is cleared to 0 in purple, and three down-sweep levels end at the exclusive scan 0 3 4 11 11 15 16 22.](diagrams/02-blelloch.svg)

*The up-sweep leaves partial sums at the right of every pair; the down-sweep turns them into the exclusive scan.*

`blelloch` in `scan.cu` scans a tile of two elements per thread, 2,048 floats for 1,024 threads. The function `at<PAD>` gives each element's shared-memory slot; with `PAD` false it is the index itself, and section 4 turns it on:

```cuda
template <bool PAD>
__device__ __forceinline__ int at(int k) { return PAD ? k + (k >> 5) : k; }

// Blelloch exclusive scan of one block's 2 * blockDim.x elements, two per thread: the up-sweep builds
// partial sums (a reduction tree), the root is cleared, and the down-sweep distributes them.
template <bool PAD>
__global__ void blelloch(const float* in, float* out, float* sums, int n) {
    extern __shared__ float s[];
    int t = threadIdx.x, m = 2 * blockDim.x, offset = 1;
    int base = blockIdx.x * m;
    int a = t, b = t + m / 2;                         // one element from each half of the tile
    s[at<PAD>(a)] = (base + a < n) ? in[base + a] : 0.0f;
    s[at<PAD>(b)] = (base + b < n) ? in[base + b] : 0.0f;

    for (int d = m >> 1; d > 0; d >>= 1) {            // up-sweep
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            s[at<PAD>(bi)] += s[at<PAD>(ai)];
        }
        offset *= 2;
    }
    if (t == 0) {
        if (sums) sums[blockIdx.x] = s[at<PAD>(m - 1)];   // the block's total
        s[at<PAD>(m - 1)] = 0.0f;                     // clear the root for an exclusive scan
    }
    for (int d = 1; d < m; d *= 2) {                  // down-sweep
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            float tmp = s[at<PAD>(ai)];               // save the left child
            s[at<PAD>(ai)] = s[at<PAD>(bi)];          // left <- parent
            s[at<PAD>(bi)] += tmp;                    // right <- parent + saved left
        }
    }
    __syncthreads();
    if (base + a < n) out[base + a] = s[at<PAD>(a)];
    if (base + b < n) out[base + b] = s[at<PAD>(b)];
}
```

Thread `t` loads elements `t` and `t + 1024`, one from each half, as Harris, Sengupta and Owens do ("Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3, chapter 39, 2007), so a warp's load reads 32 adjacent floats. The barrier at the top of each level publishes the previous level's writes, and the last one lets the down-sweep finish before results are copied out. Thread 0 saves the block's total before the clear destroys it. The indices `offset * (2 * t + 1) - 1` and `offset * (2 * t + 2) - 1` are node `t`'s left and right child, the `- 1` making them 0-based. A 2,048-element tile has 11 levels per sweep, so a block passes 23 barriers.

## 4. Bank conflicts in the tree, and the offset that removes them

Shared memory on compute capability 7.5 has 32 banks of 4-byte words: a warp's access takes one pass when its addresses fall in distinct banks, and $k$ passes when $k$ distinct addresses share a bank, a $k$-way **bank conflict** ([post 06](../06-matrix-transpose/index.md) works the model through). The tree indices are multiples of the offset, so they crowd into fewer banks as it doubles: at offset 16 a warp's 32 right children, `32 * t + 31`, all lie in bank 31. The model counts the passes in one 2,048-float tile with the plain index and with one slot skipped after every 32, `index + (index >> 5)`:

```text
5. bank conflicts in one Blelloch tile of 2,048 floats (1,024 threads, 32 banks)
  up-sweep offset  active threads  worst degree, plain  worst degree, index + (index >> 5)
                1           1,024                    2                                   1
                2             512                    4                                   1
                4             256                    8                                   1
                8             128                   16                                   1
               16              64                   32                                   1
               32              32                   32                                   2
               64              16                   16                                   2
              128               8                    8                                   2
              256               4                    4                                   2
              512               2                    2                                   2
            1,024               1                    1                                   1
  shared passes of both sweeps: plain 2,681, padded 511, conflict-free ideal 476
  padded tile: the highest index, 2,047, lands at 2,110; scan.cu reserves 2,048 + 64 = 2,112 floats
```

These counts are derived by the model: plain indices take 2,681 passes where a conflict-free access would take 476, 5.6 times as many, and the offset brings them to 511. It shifts each run of 32 indices one bank further, for 64 extra floats per tile, 256 bytes of the 48 KB a block may use on the GTX 1650. GPU Gems 3 wrote the offset for the 16 banks of the GeForce 8800; on 32 banks it is `index >> 5`. The conflict-free kernel is `blelloch<true>`.

## 5. Arrays larger than a block: three phases

One block scans at most 1,024 elements with `hillisSteele` or `warpScan` and 2,048 with `blelloch`, and blocks cannot wait for each other inside a kernel. `scanDevice` in `scan.cu` therefore scans a large array in the three phases of GPU Gems 3: (1) every block scans its part and saves its total; (2) the totals are scanned by the same method, recursively while they exceed one block; (3) every element of block $b$ adds the scanned total of the blocks before $b$, which is entry $b - 1$ of an inclusive scan of the totals and entry $b$ of an exclusive one. The buffers of every level are allocated once, before any timing. The model counts the levels and the bytes:

```text
6. the three-phase scan of N = 16,777,216 floats in scan.cu
  Hillis-Steele and warp shuffle, 1,024 per block: blocks per level 16,384 -> 16 -> 1
  Blelloch, 2,048 per block: blocks per level 8,192 -> 4 -> 1
  bytes: phase 1 reads and writes 134,217,728, phase 3 again 134,217,728: 268,435,456 in all, twice the 134,217,728 of a single pass
```

Phases 1 and 3 each read and write the whole array, so a three-phase scan moves twice the bytes of one that touches each element once. NVIDIA's CUB library does the latter: `cub::DeviceScan` uses the decoupled look-back of Merrill and Garland ("Single-pass Parallel Prefix Scan with Decoupled Look-back", NVIDIA, 2016), in which each block reads the running total of earlier blocks as they publish it.

## 6. The numbers

`scan.cu` scans $N = 16{,}777{,}216$ floats, each 0 or 1 from a fixed generator, so every total is an integer below $2^{24}$ and exact in a float. It first runs 100 untimed scans, because the laptop GPU idles at a low clock; each row then gets one untimed and 20 timed scans between two CUDA events, averaged, and is compared element by element with a host prefix sum. A copy kernel gives the ceiling. Bandwidth is the $8N$ bytes a scan must move at least, $N$ floats read and $N$ written, over the time. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), CUB 2.2.0
one block, 8 values:   3 1 7 0 4 1 6 3
  inclusive (Hillis-Steele): 3 4 11 11 15 16 22 25
  exclusive (Blelloch)     : 0 3 4 11 11 15 16 22
N = 16777216 floats of 0 or 1, 1024 threads per block, average of 20 timed scans after 100 warm-up scans
method                    result     per block  levels  time (ms)    GB/s  check
copy (the ceiling)        copy               -       -      0.957   140.2  ok
Hillis-Steele, shared     inclusive       1024       3      3.965    33.8  ok
Blelloch, plain index     exclusive       2048       3      4.808    27.9  ok
Blelloch, conflict-free   exclusive       2048       3      3.390    39.6  ok
warp shuffle              inclusive       1024       3      2.337    57.4  ok
CUB DeviceScan            exclusive          -       -      0.961   139.6  ok
```

These numbers are measured; the reader's card will give other digits, and the order of the rows is the lesson. The card does not hold its 1,740 MHz boost clock under a sustained kernel: it sits at its software power cap (about 35 W), and `nvidia-smi`, sampling every 100 ms, read its streaming-multiprocessor clock at 1,650 to 1,725 MHz while this run ran, which started at 79 degrees C. Three such runs agreed within 0.05 ms per row; one that shared the GPU with another program was up to 5.4 times slower. An earlier draft quoted an RTX 3080 table; this one is the shipped `scan.cu`'s output.

![Scan bandwidth on the GTX 1650: Hillis-Steele 33.8 GB/s, Blelloch plain 27.9, conflict-free 39.6, warp shuffle 57.4, CUB 139.6, against the copy's 140.2 and the three-phase limit of 70.1.](diagrams/04-scan-perf.svg)

*Only CUB reaches the copy; every three-phase scan stays under half of it, the bound its second pass over memory sets.*

Four readings follow. First, work-efficiency alone does not win: Blelloch with plain indices takes $4.808 / 3.965 = 1.21$ times as long as Hillis-Steele, with a tenth of the additions, because inside a block both wait on barriers and shared-memory passes, and Blelloch has 23 barriers to 11 and the surplus passes of section 4. Second, the conflict-free offset makes it $4.808 / 3.390 = 1.42$ times faster, 1.17 times ahead of Hillis-Steele. Third, the warp shuffle is $3.965 / 2.337 = 1.70$ times faster than the shared-memory Hillis-Steele and $3.390 / 2.337 = 1.45$ times faster than the conflict-free Blelloch: inside a warp a step is one instruction with no barrier, so the extra additions cost nothing and the work-inefficient algorithm wins. Fourth, every three-phase scan pays its second pass: twice the copy is $2 \times 0.957 = 1.914$ ms, and the best takes 2.337 ms. CUB, in one pass, takes 0.961 ms, within 0.5% of the copy, 87% of the 160 GB/s theoretical peak and 2.43 times faster than the warp shuffle.

## 7. The payoff: stream compaction

A scan's real product is positions. **Stream compaction** removes the elements a filter rejects and closes the gaps: compute a **predicate**, 1 to keep and 0 to drop; take its **exclusive scan**, which gives each kept element the number of kept elements before it, its index in the output; and **scatter**, each thread writing its kept element to that index. The model keeps the positive values of nine numbers:

```text
7. stream compaction: keep the positive values
  input           :  3 -1  7  0 -2  4  1 -5  6
  predicate       :  1  0  1  0  0  1  1  0  1
  exclusive scan  :  0  1  1  2  2  2  3  4  4
  scatter         : [3, 7, 4, 1, 6]  (kept 5 of 9; the scan's total is 5)
```

![Compaction of 3 -1 7 0 -2 4 1 -5 6: predicate 1 0 1 0 0 1 1 0 1, its exclusive scan 0 1 1 2 2 2 3 4 4, and arrows scattering the kept values to 3 7 4 1 6.](diagrams/03-stream-compaction.svg)

*The exclusive scan of the predicate is each kept element's output index, and the predicate's total is the output's length.*

Only the scan couples the elements, and the survivors keep their order. Blelloch's report lists the scan's uses, radix sort and quicksort among them, and GPU Gems 3 builds stream compaction, summed-area tables and radix sort on it.

## 8. Make it run

The CPU model needs only the standard library and no GPU. From the series root:

```bash
uv run python posts/08-parallel-scan/snippets/scan_model.py
```

It prints the blocks quoted in sections 1 to 7. The kernel needs an NVIDIA GPU and the CUDA Toolkit, whose CUB headers it includes, and builds from the series root with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 posts/08-parallel-scan/snippets/scan.cu -o scan && ./scan
```

On Windows the program is `scan.exe`, and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both. The build takes about 12 seconds; the program prints section 6's output in under two seconds and exits with status 1 if any check fails.

## 9. What can go wrong?

Each mistake below was reproduced on the author's GTX 1650 with a one-line change to `scan.cu`.

- **Starting the ping-pong from the wrong half.** With `pin = 0, pout = 1`, the first step reads the unwritten half 1 and overwrites the input. An earlier `scan.cu` did this and printed `0 0 0 0 0 0 0 0`; uninitialized shared memory need not be zero, so another card may print other garbage.
- **Dropping the up-sweep barrier.** The eight values, one warp, still came out right; both full-size Blelloch rows failed. A small test does not prove a barrier unnecessary.
- **Dropping the `- 1` from the tree indices.** The eight values came out `0 3 1 9 0 12 1 8`, and the full-size run stopped with "an illegal memory access was encountered": the last right child, index 2,048, lies past the tile.
- **Saving the block's total after clearing the root.** Every saved total is 0, so phase 3 adds nothing; both Blelloch rows failed while the one-block demonstration passed.
- **Mixing inclusive and exclusive offsets.** Adding entry $b$ of an inclusive scan of the totals counts block $b$ twice; the Hillis-Steele and warp-shuffle rows failed.

---

## Common pitfalls

1. **Expecting the work-efficient algorithm to win everywhere.** Built on plain-index Blelloch blocks, the scan of section 6 was 1.21 times slower than on Hillis-Steele blocks; additions matter when each step is a pass over memory, barriers and banks when it is not.
2. **Confusing inclusive and exclusive.** Blelloch gives an exclusive scan, `inclusive[i] = exclusive[i] + in[i]`, and compaction needs the exclusive one.
3. **Assuming a power-of-two length.** Pad the tile with the identity (0 for a sum) when loading, as `blelloch` does past `n`, instead of resizing the array.
4. **Allocating inside the recursion.** A `cudaMalloc` at every level is a driver call that can synchronize the device; `scan.cu` allocates every level's buffers once.
5. **Writing a device-wide scan by hand.** A three-phase scan moves twice the bytes of CUB's single pass; the best hand-written one here took 2.43 times as long.

---

## Further reading

- Harris, M., Sengupta, S. and Owens, J. D., "Parallel Prefix Sum (Scan) with CUDA" (*GPU Gems 3*, chapter 39, NVIDIA, 2007). The Blelloch kernel, conflict-free addressing, large arrays, compaction.
- Blelloch, G. E., "Prefix Sums and Their Applications" (Carnegie Mellon University, CMU-CS-90-190, November 1990). The up-sweep and down-sweep, and the scan as a building block.
- Hillis, W. D. and Steele, G. L., "Data Parallel Algorithms" (*Communications of the ACM* 29(12), December 1986). The doubling-stride scan.
- Merrill, D. and Garland, M., "Single-pass Parallel Prefix Scan with Decoupled Look-back" (NVIDIA Technical Report NVR-2016-002, March 2016). The one-pass scan behind CUB's `DeviceScan`.
- NVIDIA, *CUB* documentation, `cub::DeviceScan`, consulted October 2026. The library scan of section 6.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 09 - When threads share data: barriers, atomics, and the memory model](../09-synchronization/index.md):** the exact guarantees of the barriers placed between sweep levels here, and why only a new launch, the boundary between the three phases, orders every block.
- **[Post 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy](../10-roofline/index.md):** why a scan with one addition per element is bound by memory, and how the copy's 140 GB/s becomes the roof that CUB reaches.
