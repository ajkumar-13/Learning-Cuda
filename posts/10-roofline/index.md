# 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy

> **TL;DR.** Arithmetic intensity $I$, the floating-point operations (FLOP) a kernel performs per byte of dynamic random-access memory (DRAM) traffic, decides which roof caps it: on the author's GTX 1650, $I \times 160$ GFLOP/s below the ridge point of 19.5 FLOP per byte and 3,118 GFLOP/s above it. The series' measured kernels sit under those roofs, except the naive matrix multiplication, which runs at 4.9 times its roof because the caches serve reads the count sends to DRAM. Occupancy, the share of a streaming multiprocessor's (SM's) warp slots that resident warps fill, is capped by block size, registers or shared memory, and it hides latency only until enough bytes are in flight, which independent loads per thread can supply at lower occupancy.
>
> **Prerequisites:** [post 03](../03-matrix-multiplication/index.md), [post 06](../06-matrix-transpose/index.md).
> **Safe to skip?** Skip it if the reader can already count a kernel's FLOP per byte, compute a card's ridge point, and derive blocks per SM from a kernel's register and shared-memory counts.
>
> **After reading, you will be able to:**
>
> - Count a kernel's arithmetic intensity in FLOP per byte and place it on the roofline of the GTX 1650.
> - Say from the ridge point whether a kernel is memory bound or compute bound, and name the lever that moves it.
> - Read a measured kernel against its roof, and explain why a kernel can run above the roof its count predicts.
> - Derive theoretical occupancy from block size, registers per thread and shared memory per block, and name the binding limit.
> - Explain why more occupancy stops helping once enough bytes are in flight, and why a low-occupancy kernel can still be fast.

![A log-log roofline: a 160 GB/s slope meets a flat 3,118 GFLOP/s peak at 19.5 FLOP per byte. Measured dots sit under it, except the naive matrix multiply at 196 GFLOP/s, above its roof of 40, with an arrow to an intensity of at least 1.225.](diagrams/01-roofline.svg)

*The roof is counted, the dots are measured: every dot sits under its roof except the naive matrix multiply, whose real DRAM traffic is smaller than its count.*

---

## 1. The question: what is the fastest a kernel can run?

Every post so far judged a kernel against a ceiling stated in passing: [post 02](../02-vector-addition/index.md) against the card's 160 GB/s, [post 03](../03-matrix-multiplication/index.md) against the arithmetic peak. This post joins the two into one model, the roofline of Williams, Waterman and Patterson (Communications of the ACM, 2009), and asks of any kernel: what is the fastest it could run on this card, and which resource stops it? The answer says which optimization can help at all.

The roofs, intensities and occupancy limits are derived by [`snippets/roofline_model.py`](snippets/roofline_model.py), a standard-library script that needs no graphics processing unit (GPU). The kernels placed under the roofs are measured: the harnesses of posts 02, 03, 04 and 06 timed them on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, and the script carries them as constants with the post named. Nothing is timed again here.

## 2. Arithmetic intensity: FLOP per byte

A kernel moves bytes to and from global memory and performs floating-point operations on them. The ratio is its **arithmetic intensity**:

$$I = \frac{\text{floating-point operations}}{\text{bytes moved to and from DRAM}} \quad \text{(FLOP per byte)}$$

For vector addition `C = A + B`, thread `i` does one add and moves 12 bytes (it reads `A[i]` and `B[i]` and writes `C[i]`, four bytes each), so $I = 1/12 \approx 0.083$. A reduction does one add per four-byte element read, $I = 0.25$. Each byte is needed once, so these counts are exact.

Matrix multiplication looks compute heavy. With `A: [M, K]`, `B: [K, N]` and `C: [M, N]` at $M = N = K = 1{,}024$, the naive kernel's thread `(row, col)` reads $2K$ floats for $2K$ FLOP: $2N^3$ FLOP against $4(2N^3 + N^2)$ bytes, $I = 0.250$. Post 03's tiling with `TILE` = 16 lets each loaded value serve 16 threads through shared memory and raises $I$ to 3.969, 15.9 times (post 03 counted loads only, 4.00; the 4 MiB write of `C` is the difference). A `TILE` of 32 is hypothetical here. Moving each element once gives the most any algorithm can reach: $N/6 = 170.7$.

```text
2. arithmetic intensity, every read of the algorithm as written going to DRAM (derived)
  kernel                    I FLOP/byte  bound     ceiling GFLOP/s  % of pi
  vector add  C = A + B         0.083    memory            13.3     0.4%
  saxpy  y = a * x + y          0.167    memory            26.7     0.9%
  reduction  sum                0.250    memory            40.0     1.3%
  matmul naive                  0.250    memory            40.0     1.3%
  matmul tiled, TILE 16         3.969    memory           635.0    20.4%
  matmul tiled, TILE 32         7.877    memory          1260.3    40.4%
  matmul, each once           170.667    compute         3118.1   100.0%
  tiling with TILE 16: 0.250 -> 3.969 FLOP per byte, 15.9 times (0.25 -> 4.00 counting loads only)
```

These numbers are derived (section 3 explains the ceiling column). The count assumes every read of the algorithm as written goes to DRAM; a cache can only remove traffic, so for a kernel that re-reads data the count is a lower bound on the real intensity.

## 3. The two roofs and the ridge point

A kernel is limited by whichever of two resources it exhausts first. The compute peak $\pi$ is every lane doing a fused multiply-add, 2 FLOP, every cycle: the GTX 1650 has 14 SMs of 64 lanes for 32-bit floating point (FP32), 896 lanes, and a maximum SM clock of 1,740 MHz (`nvidia-smi -q -d CLOCK`), so $\pi = 896 \times 1.740 \times 10^9 \times 2 = 3{,}118$ GFLOP/s. The bandwidth $\beta$ is the theoretical 160 GB/s of post 01. A kernel of intensity $I$ receives at most $\beta$ bytes per second, enough for $I \times \beta$ FLOP/s, and never exceeds $\pi$:

$$P(I) = \min(\pi,\; I \times \beta)$$

On log-log axes $P(I)$ is a roof: a slope where bandwidth binds, a flat ceiling where arithmetic binds, meeting at the **ridge point** $\pi / \beta = 19.5$ FLOP per byte. Left of it a kernel is **memory bound**, and the lever is fewer bytes or bytes moved closer to $\beta$; right of it, **compute bound**, and the lever is fewer or cheaper instructions. A ridge belongs to one card.

Every kernel of section 2 but the each-once bound sits left of 19.5. The tiled matrix multiplication's ceiling is 635 GFLOP/s, a fifth of $\pi$; passing the ridge takes more reuse, such as each thread computing several outputs from registers, or the Tensor Cores of [post 17](../17-tensor-cores/index.md).

The peak needs the right clock: the 1,155 MHz that `cudaGetDeviceProperties` reports as `clockRate` gives 2,070 GFLOP/s, below the 2,189 that post 03 measured for cuBLAS, which no peak can be. Under a sustained kernel the laptop card also sits at its software power cap of about 35 W ([post 08](../08-parallel-scan/index.md)), below its boost clock, so a measured share of $\pi$ understates how close a kernel came.

## 4. The series on the roofline

The hero figure places the measured kernels under these roofs. A memory-bound kernel's GFLOP/s is its bandwidth times its intensity: post 02's vector addition moved 145.0 GB/s, $145.0 / 12 = 12.1$ GFLOP/s.

```text
3. measured on the author's GTX 1650 by earlier posts, against the roof (measured / derived)
  kernel                  I FLOP/byte  measured GFLOP/s  roof GFLOP/s  of roof  source
  vector add                  0.083              12.1          13.3      91%  post 02, 10,000,000 elements, 145.0 GB/s
  reduction, shuffle          0.250              38.6          40.0      97%  post 04, warp shuffle, 154.6 GB/s
  matmul naive                0.250             196.0          40.0     490%  post 03, 1024 cubed
  matmul tiled, TILE 16       3.969             305.0         635.0      48%  post 03, 1024 cubed
  cuBLAS                    170.667            2189.0        3118.1      70%  post 03, 1024 cubed; I at most
  naive matmul runs 4.9 times its roof: its DRAM intensity is at least 1.225 FLOP per byte,
    because the caches serve repeats the count sends to DRAM
  cuBLAS needs at least 13.7 FLOP per byte to reach 2,189 GFLOP/s at 160 GB/s
  transpose, padded: I = 0, 124.4 GB/s = 78% of beta
```

The measured column is measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, by the posts named; the rest is derived. The reader's card gives other digits; the positions against the roofs are the lesson.

The two kernels whose count is exact hug the slope, at 91% and 97%; their only lever left is fewer bytes. Post 06's padded transpose has $I = 0$, which a log axis cannot show, and moves 78% of $\beta$. The tiled matrix multiplication reaches 48% of its roof: post 03 traced the gap to the shared-memory loads and barriers of its inner loop, a limit inside the SM that neither roof models. cuBLAS reaches 70% of $\pi$, at an intensity between 13.7 (below which 160 GB/s could not feed it) and 170.7, where the figure places it.

The naive matrix multiplication is the instructive point. Its count gives a roof of 40 GFLOP/s, and it ran at 196, 4.9 times higher. No kernel outruns the roof of the bytes it really moves, so it moved fewer than the count: post 03 found that a warp's 32 threads need only 2 distinct values of `A` and 16 of `B` per step, and inferred that the L1 and L2 caches serve the repeats. At 196 GFLOP/s and at most 160 GB/s, its true DRAM intensity is at least $196 / 160 = 1.225$ FLOP per byte. A dot above its roof means the byte count is wrong, never the card: the roofline takes the intensity of the traffic that reaches DRAM, which counting bounds and a profiler measures ([post 11](../11-profiling-debugging/index.md)).

## 5. Occupancy: filling the warp slots

Reaching the roof takes keeping memory busy, which the GPU does by switching to a ready warp while one waits ([post 01](../01-introduction-to-cuda/index.md)). The CUDA C++ Best Practices Guide (section "Occupancy") defines **occupancy** as "the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps". Theoretical occupancy is what the launch allows; achieved occupancy is a counter reading, which post 11 describes.

The CUDA C++ Programming Guide's table for compute capability 7.5 gives the limits per SM: 32 resident warps (1,024 threads), 16 resident blocks, 65,536 32-bit registers, and 64 KB of shared memory (48 KB per block unless the kernel opts in to 64 KB). A launch spends three resources, and each caps the blocks that fit:

1. **Block size.** A 256-thread block is 8 warps, so at most $32 / 8 = 4$ fit; a 32-thread block is one warp, but the 16-block limit stops it at 16 warps.
2. **Registers per thread.** The Best Practices Guide (section "Calculating Occupancy"): "register allocations are rounded up to the nearest 256 registers per warp". The model also splits the register file into four quarters of 16,384, one per warp scheduler, each holding whole warps, as the toolkit's own calculator, `cuda_occupancy.h`, does with its four sub-partitions. With both rules the model matches the runtime's `cudaOccupancyMaxActiveBlocksPerMultiprocessor` in all 24,840 cases that [`snippets/occupancy_check.cu`](snippets/occupancy_check.cu) tries on this card, 185 of which the quarters decide. At 96 registers a warp needs $96 \times 32 = 3{,}072$; a quarter holds 5, the SM 20, and 20 warps make 2 blocks of 8.
3. **Shared memory per block,** rounded up to 256 bytes. A block asking for 49,152 bytes allows $65{,}536 / 49{,}152 = 1$ block.

The smallest binds. The compiler reports a kernel's registers and static shared memory with `-Xptxas=-v`: for post 03's tiled kernel, today's compile printed `Used 39 registers, 2048 bytes smem`, the 2,048 bytes being `s_A` and `s_B`, two `[16, 16]` tiles. The model applies the limits to the series' kernels at the counts their compiles report, and to four hypothetical launches:

```text
  kernel                    regs   smem  block  byBlock byReg bySmem  blk/SM  binds on    occ
  vector add (02)             12      0    256        4    16     16       4  block size  100%
  matmul naive (03)           49      0    256        4     4     16       4  block size  100%
  matmul tiled 16 (03)        39   2048    256        4     6     16       4  block size  100%
  reduction, shuffle (04)     15     32    256        4    16     16       4  block size  100%
  transpose, padded (06)      20   4224    256        4    10     15       4  block size  100%
  register-heavy              96      0    256        4     2     16       2  registers    50%
  over-tiled (smem)           32  49152    256        4     8      1       1  shared mem   25%
  TILE 32, 1,024 threads      39   8192   1024        1     1      8       1  block size  100%
  tiny blocks                 12      0     32       16    16     16      16  block size   50%
```

These are derived. Every kernel the series has written reaches 100% theoretical occupancy, bound by block size, so occupancy is not what held the tiled matrix multiplication at 48% of its roof. A `TILE` of 32 also fills every slot, but as one block per SM, so each `__syncthreads()` stalls all of the SM's warps, and above 64 registers per thread ($65{,}536 / 1{,}024$) the block does not fit at all.

![Left: blocks per SM each limit allows for 256-thread blocks at 96 registers per thread: block size 4, registers 2 in red, shared memory 16. Right: 32 warp slots in four rows of eight, one row per quarter of the register file, 16 filled and 16 empty, occupancy 50%.](diagrams/02-occupancy-limiters.svg)

*The scarcest resource sets the blocks per SM: at 96 registers per thread two blocks fit, and half of the warp slots stay empty.*

## 6. Occupancy is not performance

Occupancy buys latency hiding, which has a target. Little's law, as Volkov puts it ("Better Performance at Lower Occupancy", GPU Technology Conference, 22 September 2010), is "Needed parallelism = Latency x Throughput": to stream $\beta$ bytes per second through loads that take $L$ seconds, $\beta \times L$ bytes must be in flight. Volkov gives "400+ cycles for memory"; with an illustrative 400 cycles at 1,740 MHz the model prints:

```text
5. latency hiding (illustrative latency of 400 cycles at 1.740 GHz)
  bytes in flight to sustain beta = 160 GB/s x 229.9 ns = 36,782 bytes, 2,627 per SM
  1 independent 128-byte load(s) per warp: 21 warps per SM, occupancy  66%
  2 independent 128-byte load(s) per warp: 11 warps per SM, occupancy  34%
  4 independent 128-byte load(s) per warp:  6 warps per SM, occupancy  19%
```

A warp loading one float per lane has 128 bytes in flight until the load returns. With one load per warp, an SM needs 21 warps, 66% occupancy, to keep 2,627 bytes in flight; past that, more warps only queue. With two or four independent loads per thread before any is used, **instruction-level parallelism** (ILP), 11 or 6 warps suffice. The latency is illustrative and the knees move with it; that ILP moves them left does not.

![Attainable bandwidth against resident warps per SM, 0 to 32, for an illustrative 400-cycle latency: with one load in flight per warp the line reaches 160 GB/s at 21 warps, with two at 11, with four at 6, and is flat after.](diagrams/03-latency-hiding.svg)

*More warps help only until the bytes in flight cover the latency; independent loads per thread get there with fewer warps.*

Volkov measured this on a GTX 480: a copy kernel moving 14 `float4` values per thread reached "84% of peak at 4% occupancy", and his matrix multiplication rose from 242 GFLOP/s at 67% occupancy to 485 at 33% by computing eight outputs per thread. The Best Practices Guide: "Higher occupancy does not always equate to higher performance". Capping registers to add warps (`__launch_bounds__`, `-maxrregcount`) can make the compiler spill to local memory, which lives in device memory, and the spills can cost more than the warps save. For triage: cover the latency with warps or independent loads, then look elsewhere.

## 7. What the roofline does not see

- **Launch overhead.** Post 02's vector addition took 3.6 µs over 1,000 elements, 3.4 GB/s, about 2% of $\beta$: the time was the launch, not the bytes, the regime of [post 13](../13-cuda-graphs/index.md).
- **Contention.** The model counts bytes, not queues. Post 05's histogram ($I = 0$) read its 64 MiB with global atomics at 13.5 GB/s on random data and 6.6 GB/s when every value hit one bin, 4% to 8% of $\beta$; privatization lifted the random case to 117.2 GB/s ([post 05](../05-histogram/index.md)).
- **The access pattern.** The slope assumes coalesced access. Post 06's naive transpose moves the same bytes as its padded one at 41.5 GB/s against 124.4.

## 8. Make it run

The model needs only the standard library. From the series root:

```bash
uv run python posts/10-roofline/snippets/roofline_model.py
```

It prints the five parts quoted above in under a second. To place the reader's own kernel, add its FLOP and bytes to `KERNELS`, and its registers and static shared memory from `-Xptxas=-v` to `OCC`:

```bash
nvcc -O3 -arch=sm_75 -Xptxas=-v -c posts/03-matrix-multiplication/snippets/matmul.cu -o matmul.obj
```

Dynamic shared memory (`extern __shared__`) is not in that report. To check the occupancy rule against the runtime on the reader's own card:

```bash
nvcc -O3 -arch=sm_75 posts/10-roofline/snippets/occupancy_check.cu -o occupancy_check
./occupancy_check
```

On the author's GTX 1650 its last two lines read:

```text
24840 cases, 0 mismatches between the rule and cudaOccupancyMaxActiveBlocksPerMultiprocessor
in 185 of the cases, dividing the whole register file would give another answer
```

On a card of another compute capability the rule's constants do not apply, and the program says so. On Windows `nvcc` may need `-ccbin` ([RUNNING.md](../../RUNNING.md)). For another card, `SMS`, `CLOCK_HZ` and `PEAK_BW` change, and so does the ridge.

## 9. What can go wrong?

- **Leaving out dynamic shared memory.** A kernel with 4,224 static bytes that also takes 16,384 at the launch uses 20,608 per block: 3 blocks per SM, not the 15 the report suggests.
- **Dividing the whole register file.** $65{,}536$ over the registers per block ignores the quarters: at 104 registers and 64-thread blocks it gives 9 blocks where the card fits 8.
- **Judging a kernel with no arithmetic in FLOP/s.** A transpose or a histogram has $I = 0$ and is judged by its bandwidth as a fraction of $\beta$.

---

## Common pitfalls

1. **Optimizing for the wrong roof.** Tuning arithmetic in a memory-bound kernel moves nothing; compare $I$ with the ridge point, then pick the lever.
2. **Reading many FLOP as compute bound.** The naive matrix multiplication does $2N^3$ FLOP and still counts at $I = 0.25$, far left of the ridge of 19.5.
3. **Comparing with another card's ridge.** The ridge is $\pi / \beta$ of one card; an intensity is judged against the card it runs on.
4. **Chasing 100% occupancy.** Occupancy pays only until the bytes in flight cover the latency; capping registers to add warps can spill and lose more than it gains.
5. **Reading a dot above its roof as a broken model.** It means the caches cut the kernel's DRAM traffic below the count.

---

## Further reading

- Williams, S., Waterman, A. and Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (Communications of the ACM 52(4), 2009). The model of this post.
- NVIDIA, *CUDA C++ Best Practices Guide*, sections "Occupancy" and "Calculating Occupancy", consulted October 2026.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Occupancy Calculator" and the compute-capability table, consulted October 2026 (CUDA 12.3 edition).
- Volkov, V., "Better Performance at Lower Occupancy" (GPU Technology Conference, 22 September 2010). Little's law on a GPU, and instruction-level parallelism in place of occupancy.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy](../11-profiling-debugging/index.md):** measures how close a kernel gets to the ceilings this post counted, and the occupancy it achieves.
- **[Post 14 - Kernel fusion: the fastest kernel is the one never launched](../14-kernel-fusion/index.md):** raises arithmetic intensity by removing whole round trips to DRAM between kernels.
