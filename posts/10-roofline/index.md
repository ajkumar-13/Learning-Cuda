# 10 · How fast can a kernel be? — roofline, arithmetic intensity, and occupancy

> **TL;DR.** Before optimizing a kernel it is worth knowing its *ceiling*: the fastest it could possibly run on your card, and which resource sets that limit, and two numbers decide it. **Arithmetic intensity** `I` (floating-point operations per byte moved) says whether a kernel is **memory bound** (starved for bytes) or **compute bound** (starved for FLOP/s), and the **roofline** turns `I` into a hard performance ceiling. **Occupancy** (the fraction of an SM's warp slots you fill) says whether enough warps are in flight to hide memory latency, and it is capped by whichever of three resources runs out first: block size, registers, or shared memory. This post derives all of it from first principles with a runnable model, so that when [Post 11](../11-profiling-debugging/index.md) opens a profiler, every number it reports already means something.
>
> **After reading this you will be able to:**
> - Count a kernel's arithmetic intensity and place it on the roofline.
> - Say whether a kernel is memory bound or compute bound, and which lever (bandwidth or FLOP/s) can move it.
> - Derive theoretical occupancy from registers, shared memory, and block size.
> - Explain why higher occupancy stops helping, and why a low-occupancy kernel can still be fast.

![A log-log roofline: a sloped line rising with bandwidth in the memory-bound region bends at a ridge point into a flat compute-bound ceiling; vector-add sits far left on the slope, tiled matmul partway up toward the ridge.](diagrams/01-roofline.svg)
*Arithmetic intensity fixes which roof bounds a kernel. Optimization is moving the dot up to its roof, or rightward to a higher one.*

---

## 1. The ceiling question

Every optimization in the series so far aimed at a ceiling nobody had drawn. Coalescing in the transpose, tiling in matrix multiply, privatization in the histogram: each made a kernel faster, but faster *toward what*? Without the ceiling you cannot tell a kernel that is 40% of the way to its limit (worth more work) from one already at 95% (leave it alone). Worse, you cannot tell *which* limit you are near, so you cannot tell which optimization could even help.

So before touching a kernel, ask two questions. What is the fastest it could run on this card? And what is stopping it from running faster? The answers come from counting, not from a profiler, and the counting is the subject of this post. The profiler in [Post 11](../11-profiling-debugging/index.md) then tells you how close to the ceiling you actually got.

Every concrete number below is derived by [`snippets/roofline_model.py`](snippets/roofline_model.py), a standard-library script that counts operations and bytes and applies the published limits of the reference **GTX 1650** (Turing, compute capability 7.5). It needs no GPU, because none of these are measurements; they are the ceiling a measurement will be judged against.

```bash
python snippets/roofline_model.py
```

## 2. Arithmetic intensity: FLOPs per byte

A kernel does two things with the data it touches: it moves bytes to and from global memory, and it performs floating-point operations on them. The ratio of the two is its **arithmetic intensity**:

$$I = \frac{\text{floating-point operations}}{\text{bytes moved to and from DRAM}} \quad (\text{FLOP/byte})$$

Intensity is a property of the *algorithm*, not the hardware, and it is what decides the kernel's fate. Count it for vector addition `C = A + B`: each element does one add (1 FLOP) and moves twelve bytes (read `A`, read `B`, write `C`, four bytes each). So `I = 1/12 ≈ 0.083`. That is a tiny amount of arithmetic for a lot of traffic.

The counterintuitive case is matrix multiply, which *looks* compute-heavy. A naive `N×N` product does `2N³` FLOPs, a genuinely large number, but it re-reads a full row of `A` and column of `B` from DRAM for every output element, moving about `8N³` bytes. So `I = 2N³ / 8N³ = 0.25` FLOP/byte: the same intensity as a plain reduction, and only three times that of vector-add. All that arithmetic is throttled by all that traffic. **Tiling** (Post 03) is the fix precisely because it changes this number: staging a `32×32` tile in shared memory and reusing it cuts the DRAM reads by the tile width, raising `I` by 32×.

The model counts intensity for the series' kernels and prints the ceiling each implies:

```text
kernel                  AI FLOP/byte   bound     ceiling GFLOP/s   % of FP32 peak
--------------------------------------------------------------------------------
vector-add  C=A+B         0.083      memory            13.3          0.6%
saxpy  y=a*x+y            0.167      memory            26.7          1.3%
reduction  sum            0.250      memory            40.0          1.9%
matmul naive              0.250      memory            40.0          1.9%
matmul tiled 32x32        7.877      memory          1260.3         60.9%
```

Read the two matmul rows together: tiling raised intensity from `0.25` to `7.9` FLOP/byte and lifted the ceiling from 40 to 1260 GFLOP/s, a 32× jump in how fast the kernel is *allowed* to go, without changing a single arithmetic operation.

## 3. The two roofs and the ridge point

Where does the "ceiling" column come from? A kernel is limited by whichever resource it exhausts first, and there are only two candidates: memory bandwidth and arithmetic throughput. Call the card's peak bandwidth `β` (160 GB/s on the GTX 1650) and its peak compute `π` (about 2.07 TFLOP/s FP32). A kernel of intensity `I` can move at most `β` bytes per second, which supplies at most `I × β` FLOP/s of work; and it can never exceed `π`. So the attainable performance is:

$$P(I) = \min(\pi,\; I \times \beta)$$

Plot that on log-log axes and it is a **roof**: a sloped line `I × β` on the left, where bandwidth binds, and a flat ceiling `π` on the right, where compute binds. The corner where they meet is the **ridge point**, at `I = π/β = 12.9` FLOP/byte for this card. Its meaning is sharp: a kernel with intensity below 12.9 is **memory bound**, and no amount of faster arithmetic will help it; above 12.9 it is **compute bound**, and no amount of bandwidth will. This is the single most useful number for triage, and it is one division.

Now the whole series lands on one picture. Vector-add, saxpy, reduction, and even naive matmul all sit far left on the slope, memory bound, their ceilings a rounding error against peak FLOP/s. Tiled matmul sits at `I = 7.9`, most of the way up the slope toward the ridge, its ceiling now three-fifths of peak compute. It is *still* left of the ridge, so on this small card it is still memory bound, just barely; pushing it past the ridge needs even more reuse (register tiling) or different hardware (the Tensor Cores of [Post 17](../17-tensor-cores/index.md)). The transpose of [Post 06](../06-matrix-transpose/index.md) has `I` near zero, pure data movement, which is exactly why its entire story was coalescing and bandwidth and never arithmetic.

## 4. Occupancy: filling the warp slots

The roofline says what a kernel *could* reach. Occupancy is about whether you have enough parallel work in flight to *approach* it. Recall from [Post 01](../01-introduction-to-cuda/index.md) that the GPU hides the latency of a slow memory load by switching to another ready warp while the first waits. **Occupancy** is the fraction of the SM's warp slots that are filled:

$$\text{occupancy} = \frac{\text{resident warps per SM}}{\text{maximum warps per SM}}$$

More resident warps means more cover for stalls. How many stay resident is set by whichever of three per-SM resources runs out first. The published `sm_75` limits are 32 warp slots, 65,536 registers, 65,536 bytes of shared memory, and at most 16 blocks. A launch spends three of these, and each caps how many blocks fit:

1. **Block size / warp slots.** A 256-thread block is 8 warps, so at most `32 / 8 = 4` fit in the warp slots.
2. **Registers per thread.** Registers are allocated per warp. A thread using 96 registers needs `96 × 32 = 3072` per warp, `× 8` warps `= 24,576` per block, allowing `65,536 / 24,576 = 2` blocks.
3. **Shared memory per block.** A block asking for 49,152 bytes allows `65,536 / 49,152 = 1` block.

The **binding limit** is the smallest of the three. The model prints the full derivation for a handful of kernels:

```text
kernel               regs  smem   blk  byWarp byReg bySmem  blk/SM  binds on    occ%
------------------------------------------------------------------------------------
vector-add lean        32      0   256       4     8     16      4   block size  100%
register-heavy         96      0   256       4     2     16      2   registers    50%
tiled matmul 32x32     40   8192   256       4     6      8      4   block size  100%
over-tiled (smem)      32  49152   256       4     8      1      1   shared mem   25%
tiny blocks            32      0    64      16    32     16     16   block size  100%
```

![Three vertical bars showing how many blocks fit per SM under each limit; the registers bar at two blocks is shortest and highlighted as the binding limit, with a dashed cap line drawn across at that height.](diagrams/02-occupancy-limiters.svg)
*A block fits as many copies on the SM as its scarcest resource allows. The shortest bar binds.*

Same 256-thread block, three different ceilings: the lean kernel fills every warp slot (100%), the register-heavy one is capped at 50% by registers, the over-tiled one at 25% by shared memory. To see your own kernel's costs, ask the compiler with `nvcc -Xptxas=-v`, which prints registers and shared memory per kernel; feed those into the model and you have the occupancy ceiling before you launch. [Post 11](../11-profiling-debugging/index.md) then measures the *achieved* occupancy and compares.

## 5. Occupancy is not performance

It is tempting to treat 100% occupancy as the goal. It is not. Occupancy buys latency hiding, and latency hiding has sharply diminishing returns: once enough warps are resident to keep the memory pipeline busy, usually somewhere around 50–75%, more warps cover nothing extra. Past that point the only way to add warps is to use fewer registers per thread, and forcing that makes the compiler **spill** live values to local memory (off-chip, as slow as global memory). The spills can cost more than the extra warps save.

The other half of latency hiding does not involve occupancy at all. A single warp can keep many memory loads in flight at once if its code issues independent loads before consuming any of them; this is **instruction-level parallelism**, and it is why a famously fast kernel can run at 25% occupancy by giving each thread more independent work. Occupancy and ILP are two ways to buy the same thing, latency hiding, and you trade between them. The lesson for triage: reach a reasonable occupancy, then stop looking at it and look at whether the kernel is actually stalled on memory.

## 6. What the roofline does not see

The model is powerful because it is simple, and its blind spots are exactly where its assumptions break. Three matter for this series.

First, it assumes the kernel is large enough that steady-state throughput dominates. A tiny kernel, or a stream of many tiny kernels, is instead dominated by the fixed **launch overhead** the CPU pays per launch, a cost the roofline ignores entirely. That overhead is the whole subject of [Post 13](../13-cuda-graphs/index.md).

Second, it counts DRAM traffic but not *contention* for a single address. The histogram of [Post 05](../05-histogram/index.md) can move very few bytes yet crawl, because thousands of threads serialize on the same atomic counter. No intensity number captures that; the bound is the atomic unit, not the bandwidth.

Third, it assumes bytes move at peak bandwidth, which only happens when accesses are coalesced. An uncoalesced kernel has the same intensity but achieves a fraction of `β`, so it sits below its own roof. The roofline tells you the ceiling; the profiler tells you the gap to it, and reading that gap is [Post 11](../11-profiling-debugging/index.md).

---

## Common pitfalls

- **Optimizing the wrong roof.** Hand-tuning arithmetic in a memory-bound kernel (or memory layout in a compute-bound one) does nothing. Compute the intensity, find the roof, then pick the lever that roof responds to.
- **Reading "lots of FLOPs" as compute bound.** Naive matmul does `2N³` FLOPs and is still memory bound, because it moves `8N³` bytes. Intensity, not raw FLOP count, decides the roof.
- **Chasing 100% occupancy.** Occupancy hides latency with diminishing returns past roughly 50–75%. Forcing more warps by cutting registers causes spills that can cost more than they save.
- **Forgetting the coalescing assumption.** The roofline's bandwidth ceiling assumes coalesced access. An uncoalesced kernel has the same intensity but lands well below its roof.
- **Applying the roofline to tiny or many-launch workloads.** For small kernels, per-launch overhead dominates and the steady-state roofline does not apply; that regime is CUDA Graphs, Post 13.

---

## Further reading

- Williams, S., Waterman, A., & Patterson, D., *"Roofline: An Insightful Visual Performance Model for Multicore Architectures"* (2009). The origin of the memory-bound versus compute-bound picture (technical, foundational).
- NVIDIA, *"CUDA C++ Best Practices Guide"* (current). The reference for arithmetic intensity, occupancy, and the resource limits behind resident warps (technical, reference).
- NVIDIA, *"CUDA Occupancy Calculator"* and the `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API (current). The runtime tools that automate Section 4's arithmetic (reference).
- Volkov, V., *"Better Performance at Lower Occupancy"* (GTC, 2010). The classic argument that instruction-level parallelism can substitute for occupancy (technical, historical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 11, Profiling and debugging](../11-profiling-debugging/index.md)**: open Nsight and compute-sanitizer to *measure* how close a real kernel gets to the ceiling this post drew, and read the stall reasons that explain the gap.
- **[Post 14, Kernel fusion](../14-kernel-fusion/index.md)**: the most direct way to raise arithmetic intensity, by removing whole round-trips to DRAM instead of reducing them.
- **[Post 03, Matrix multiplication](../03-matrix-multiplication/index.md)**: re-read the tiling result as the 32× intensity jump this post quantified, sliding the dot up its roof.
