# 01 - Why GPUs exist: the CUDA execution model from the ground up

> **TL;DR.** A central processing unit (CPU) is built to finish one instruction stream as fast as possible, while a graphics processing unit (GPU) spends its transistors on arithmetic and hides the wait for memory by keeping other groups of threads ready to run. A CUDA program writes a **kernel** from the point of view of one thread and launches a **grid** of **blocks** of threads, and each thread finds its element with `blockIdx.x * blockDim.x + threadIdx.x`. The hardware hands each block to a **streaming multiprocessor (SM)** and runs its threads in **warps** of 32 that share one instruction, so a branch that splits a warp runs both sides one after the other. This post builds that model and the memory tiers that feed it.
>
> **Prerequisites:** none. This is the first post of the series.
> **Safe to skip?** Skip it if the reader can already size a launch, compute any thread's global index and say what the surplus threads do, count the passes of a divergent warp, and name the memory tiers with their scopes.
>
> **After reading, you will be able to:**
>
> - Explain why a GPU beats a CPU on data-parallel work, in terms of chip area and latency hiding.
> - Map thread, block, and grid onto the hardware that runs them: streaming multiprocessors, warps of 32, and their lanes.
> - Compute any thread's global index by hand, size a launch with a ceiling division, and say what the surplus threads do.
> - Predict how many passes a warp needs when its lanes branch, and when warp divergence halves throughput.
> - Name the memory tiers from registers to global memory with the scope of each, and read a card's limits from a device query.

![On the left, a schematic CPU of four large cores above a large cache block and a control block. On the right, the GTX 1650 as 14 streaming multiprocessors of 64 arithmetic lanes each, 896 in all, above one 1,024 KB L2 cache.](diagrams/01-cpu-vs-gpu.svg)

*The GPU is not a faster CPU. It makes a different trade: less control per thread, far more arithmetic, and enough threads to keep it busy.*

---

## 1. The question: two different bets on where the time goes

A CPU core is a *latency* machine. It assumes the program is one stream of instructions in which each step often depends on the one before it and the path can change at any branch (any `if`). It spends its transistors on making *that* stream finish fast: large caches so it rarely waits on slow memory, branch prediction so it can guess which way an `if` will go and keep working, and out-of-order execution so one slow instruction does not stall everything queued behind it. A typical CPU has only a handful of these powerful cores.

A GPU makes the opposite bet. It assumes the work is *data-parallel*: the same operation applied to many independent elements, like `C[i] = A[i] + B[i]` for every `i`. NVIDIA's CUDA C++ Programming Guide puts the design rule in one line: in a GPU "more transistors are devoted to data processing rather than data caching and flow control". The figure at the top draws the trade. The CPU side is a schematic, not a floor plan; the GPU side is the author's GTX 1650, whose 14 SMs (reported by the device query of section 7) each finish 64 single-precision (32-bit floating-point) additions, multiplications, or fused multiply-adds per clock, the Programming Guide's figure for compute capability 7.5. That is 896 arithmetic lanes.

> **What is a kernel?** A **kernel** is a function that runs on the GPU, written from the point of view of *one* element. It contains no loop over the data: it says what a single thread does, and the launch starts thousands of copies of it. The launch syntax `kernel<<<blocks, threads>>>(...)`, like the `__global__` marker on the kernel, is a CUDA extension, not ordinary C++.

The rule of thumb falls straight out of the two bets: serial or heavily branching work belongs on the CPU; large, regular, data-parallel work (vector and matrix math, the heart of deep learning) belongs on the GPU.

## 2. Latency hiding: a GPU does not wait, it switches

Arithmetic lanes are only useful if they are fed, and a read from the GPU's main memory takes hundreds of clock cycles (an illustrative order of magnitude; section 6 places it among the other tiers). A CPU fights that wait with caches and prediction; a GPU tolerates it. The SM keeps many groups of threads resident at once, and each clock its scheduler issues an instruction from any group that is ready. When one group stalls on a load, the next one runs; the Programming Guide's phrase is that the GPU "can hide memory access latencies with computation". Each such group is a **warp** of 32 threads, the unit section 5 takes apart.

The CPU model of this post, `snippets/thread_model.py`, makes the arithmetic concrete with illustrative timings: a warp issues 4 instructions and then waits 12 cycles for its load to return. It prints what share of the cycles the scheduler has something to issue:

```text
4. latency hiding (illustrative: 4 instructions, then a 12-cycle wait)
  warps  scheduler busy
      1            25%
      2            50%
      4           100%
      8           100%
```

One warp keeps the scheduler busy for 4 cycles out of every 16. Four warps fill the 12-cycle wait exactly, since while one waits the other three issue; more only queue up. The figure draws both cases cycle by cycle.

![Two timelines of 32 cycles on one scheduler. One warp issues for 4 cycles and waits 12, so the scheduler is busy 8 of 32 cycles, 25 percent. Four staggered warps take turns, one issuing while three wait, and the scheduler is busy on all 32 cycles.](diagrams/02-latency-hiding.svg)

*A stall costs nothing when another warp is ready; latency hiding is a supply of ready warps.*

With a real wait of hundreds of cycles it takes tens of warps to cover it, which is why an SM of the GTX 1650 holds up to 1,024 resident threads, 32 warps, at once; [post 10](../10-roofline/index.md) counts them as **occupancy**. The bet pays only when the program supplies far more threads than the card has lanes.

## 3. The programming model: host and device

CUDA is **heterogeneous**: the CPU (the *host*) and the GPU (the *device*) work together, each with its own memory. A program runs on the host, copies its data into device memory, launches a kernel on the device, and copies the results back. The full allocate, copy, launch, copy, free cycle is the subject of [post 02](../02-vector-addition/index.md); here only the shape matters: the host orchestrates, the device computes, and the two memories are separate.

> **Reference card and the three kinds of number.** The early posts of this series measure on one **NVIDIA GeForce GTX 1650 with Max-Q Design** (Turing architecture, **compute capability 7.5**), the author's card; later posts that measure on an RTX 3080 or an A100 name that card where they use it. Every number in the series is one of three kinds, and the sentence that uses it says which: **derived** by a post's committed CPU model, which prints it; **measured** on a named card; or a **published figure** from a cited source. The reader's own card will differ; `snippets/device_query.cu` prints its limits.

## 4. The hierarchy: thread, block, grid

> **Three words, defined once.** A **thread** runs one copy of the kernel for one element. A **block** is a group of threads (up to 1,024 on this card) that run on the same SM, can share fast on-chip memory, and can synchronize with `__syncthreads()`. A **grid** is all the blocks of one launch. Blocks must be able to run *independently and in any order*, which lets one binary scale from a small laptop GPU to a datacenter card.

Each thread flattens its two-level position into one global index, `i = blockIdx.x * blockDim.x + threadIdx.x`. The kernel reads its position from four built-in, read-only variables: `threadIdx` (this thread's position within its block), `blockIdx` (which block it is in), `blockDim` (threads per block), and `gridDim` (blocks in the grid, which the index does not need). Each has `.x`, `.y`, and `.z` components; this fragment is the 2D index [post 03](../03-matrix-multiplication/index.md) uses for matrices:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Take `N = 1000` elements with 256 threads per block. Blocks come whole, so the launch needs the ceiling of $1000 / 256$, which is 4 blocks, and those start $4 \times 256 = 1{,}024$ threads: 24 more than there are elements. The CPU model derives the launch and the index of a few threads, with the warp and the lane (the thread's position inside its warp) each one lands in:

```text
2. i = blockIdx.x * blockDim.x + threadIdx.x
  blockIdx.x  threadIdx.x     i  warp  lane  i < N
           0            0     0     0     0   True
           1            5   261     0     5   True
           3            5   773     0     5   True
           3          231   999     7     7   True
           3          232  1000     7     8  False
           3          255  1023     7    31  False
```

Thread 5 of block 3 owns element $3 \times 256 + 5 = 773$. The last 24 threads of block 3 compute an index of 1,000 or more and own nothing. They still launch, so the kernel guards them with `if (i < N)` and they return without touching memory. The figure draws this launch.

![A grid of four blocks of 256 threads, each sent by an arrow to one of the 14 streaming multiprocessors while ten stay idle. Below, block 3 opened on its SM into eight warps of 32 lanes: lane 5 of warp 0 is emphasized and its index worked as 3 times 256 plus 5 equals 773, and the last 24 lanes of warp 7 are empty surplus threads.](diagrams/03-thread-hierarchy.svg)

*A thread's index is its block's offset plus its place in the block; the guard turns the launch's rounding into 24 idle threads.*

Those pieces make a real kernel, from `snippets/add_one.cu`:

```cuda
__global__ void addOne(float *data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        data[i] += 1.0f;                             // one thread, one element
}
```

There is no loop over the array: thread `i` touches one element. The program around it allocates exactly 1,000 floats, launches 4 blocks of 256 threads, and checks the result; measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, it prints:

```text
addOne<<<4, 256>>> over N = 1000: 1024 threads, 24 surplus
elements incremented exactly once: 1000 of 1000
```

In integer arithmetic the ceiling division is `(N + threads - 1) / threads`, the pattern post 02 reuses.

## 5. The hardware: streaming multiprocessors and warps

The GPU is a set of streaming multiprocessors. A launch hands whole blocks to SMs that have room; several blocks can share an SM if its resources allow. The four blocks above can occupy at most four of the GTX 1650's 14 SMs, which is why a launch of a few blocks leaves most of the card idle. Inside an SM the threads of a block run in **warps** of 32 consecutive threads: the hardware issues one instruction and all 32 lanes carry it out together, each on its own data. This is the **SIMT** model, single instruction, multiple threads. Software says grid, block, thread; the hardware runs it as SM, warp, lane, and a lane is not a core that one thread owns for life: the SM issues from many resident warps over time.

That shared instruction is free when every thread of a warp takes the same path, and it is not free when they disagree. In the Programming Guide's words, "full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path." This fragment splits every warp:

```cuda
if (i % 2 == 0) doA();   // even lanes
else            doB();   // odd lanes
```

Both paths are taken inside one warp, so the warp runs them **one after the other**: the even lanes run `doA()` while the odd lanes sit idle, then the reverse for `doB()`. This is **warp divergence**. When the two sides do comparable work, a warp that splits evenly takes about twice the time; a lopsided split, or more than two paths, changes the factor. Divergence happens only inside a warp: different warps run independently, whatever paths they take. On compute capability 7.0 and later each thread also keeps its own program counter, so the two paths of a divergent warp may interleave, but the warp still issues one path at a time and the passes still add up.

The CPU model counts passes, one per distinct path that a warp's 32 lanes take:

```python
def passes(branch, warp_starts):
    """Passes a set of warps needs: one per distinct path its 32 lanes take."""
    total = 0
    for start in warp_starts:
        paths = {branch(start + lane) for lane in range(WARP)}
        total += len(paths)          # 1 when the whole warp agrees, 2 when it splits
    return total
```

Over one block of 256 threads, eight warps, it prints:

```text
3. divergence in one block of 256 threads (8 warps of 32), passes:
  no branch                  = 8
  branch on i % 2            = 16  (every warp splits)
  branch on (i / 32) % 2     = 8  (whole warps agree)
```

It counts passes, not time, and the figure draws them for one warp. The last line is the fix. A branch on `(i / 32) % 2` still sends half the threads each way, but whole warps agree, so no warp splits.

![A warp of 32 lanes in three rows. With no branch all 32 lanes run in one pass. Branching on i mod 2 takes two passes, the even lanes running doA while the odd lanes idle, then the reverse. Below, the passes for a block of 256 threads: 8 with no branch, 16 on i mod 2, 8 on a warp-uniform branch.](diagrams/04-warp-divergence.svg)

*Divergence is not a crash, it is waste: the idle lanes still hold the warp's issue slot while the other half runs.*

The lesson: branch on conditions that are uniform across a warp, not on a thread's lane.

## 6. The memory hierarchy

A GPU has several kinds of memory, and the gap between the fastest and the slowest is large: a register answers in about a cycle and global memory, the off-chip dynamic random-access memory (DRAM), in hundreds. Those latencies are illustrative orders of magnitude that vary by architecture; the ratio between tiers is the point. The figure draws the tiers as a ladder, with each tier's scope and its size on the GTX 1650.

![A ladder of four memory tiers widening downward. Registers, one thread, 65,536 per SM, about 1 cycle. Shared memory, emphasized, one block, 48 KB per block. L2 cache, whole GPU, 1,024 KB. Global memory, whole GPU, 4.00 GiB, hundreds of cycles. The latencies are marked illustrative.](diagrams/05-memory-hierarchy.svg)

*Inputs and outputs live in global memory, at the bottom of the ladder; fast kernels bring data up once and reuse it.*

| Tier | Scope (who shares it) | Size on the GTX 1650 | What it holds |
|---|---|---|---|
| Registers | one thread | 65,536 32-bit registers per SM | a thread's private variables |
| Shared memory | one block | 48 KB per block | a scratchpad the kernel fills and reuses by hand |
| L2 cache | whole GPU | 1,024 KB | a hardware-managed cache over global memory |
| Global memory (DRAM) | whole GPU, and the host through copies | 4.00 GiB | the card's main memory, where inputs and outputs live |

The sizes are measured: the device query of section 7 printed them on the author's GTX 1650. On Turing, shared memory and the L1 cache are one 96 KB array per SM that the driver splits between them (the Programming Guide's figure for compute capability 7.5); the difference is that L1 is managed by the hardware and shared memory by the kernel.

The tier a kernel controls by hand is **shared memory**. A high-performance kernel loads a tile of data from global memory into shared memory once, then reuses it many times. That single idea, load once and reuse from on-chip memory, is the engine behind tiled matrix multiplication, reduction, transpose, convolution, and FlashAttention.

## 7. From code to the GPU: nvcc, PTX, and compute capability

The CUDA compiler `nvcc` can emit a kernel in two forms. **PTX** (Parallel Thread Execution) is a portable virtual assembly; the device driver compiles it to machine code when the program loads, which the Programming Guide calls just-in-time compilation and describes as "the only way for applications to run on devices that did not exist at the time the application was compiled". A **cubin** (CUDA binary) is native machine code for one architecture (`sm_75` for Turing, `sm_86` for the RTX 3080's Ampere generation). A binary can carry both, so it runs natively on the card it was built for and still runs on later hardware. A GPU's **compute capability** (7.5, 8.6, and so on) names its feature set and limits, and the `-arch=sm_XX` flag is the compute capability with the dot dropped.

`snippets/device_query.cu` calls `cudaGetDeviceProperties` and prints the limits the rest of the series reasons about. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
Device 0: NVIDIA GeForce GTX 1650 with Max-Q Design
  compute capability   : 7.5
  streaming multiproc. : 14 SMs
  warp size            : 32 threads
  max threads / block  : 1024
  max threads / SM     : 1024
  registers / SM       : 65536 (32-bit)
  shared mem / block   : 48 KB
  L2 cache             : 1024 KB
  global memory        : 4.00 GiB
  peak bandwidth       : 160 GB/s (theoretical)
```

Every line but the last is a property the driver reports. The last is computed from two other reported properties: the 128-bit memory bus moves 16 bytes per transfer, at two transfers per clock of the reported 5,001 MHz memory clock, which gives 160 GB/s. It is a theoretical peak, not a measurement; post 02 measures how much of it a real kernel reaches.

## 8. Make it run: the model, the device query, and the first kernel

The CPU model needs no GPU. From the series root:

```bash
uv run python posts/01-introduction-to-cuda/snippets/thread_model.py
```

It prints the launch, then the tables of sections 4, 5, and 2. The launch block:

```text
1. launch: N = 1000, blockDim.x = 256
  gridDim.x   = ceil(1000 / 256) = 4 blocks
  threads     = 4 x 256 = 1024
  surplus     = 1024 - 1000 = 24  (fail the guard i < N)
  every element covered exactly once: True
```

The two `.cu` programs need an NVIDIA GPU and the CUDA Toolkit. Each builds with one `nvcc` line from the series root, with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 posts/01-introduction-to-cuda/snippets/device_query.cu -o device_query && ./device_query
nvcc -O3 -arch=sm_75 posts/01-introduction-to-cuda/snippets/add_one.cu -o add_one && ./add_one
```

On Windows the programs are `device_query.exe` and `add_one.exe`, and if `nvcc` cannot find the host compiler `cl.exe` it needs a `-ccbin` argument; [RUNNING.md](../../RUNNING.md) covers both and the `-arch` value for other cards. On the GTX 1650 the two programs print the outputs of sections 7 and 4.

## 9. What can go wrong?

The mistakes a first kernel makes with this post's ideas, each reproduced with `add_one.cu` on the author's GTX 1650.

- **A floor division for the block count.** `N / threads` gives 3 blocks for 1,000 elements, 768 threads; the program runs without an error and the last 232 elements are never touched (`768 of 1000` in the check).
- **A missing guard.** With the `if (i < N)` line deleted, the 24 surplus threads read and write past the end of the array, and the program still prints `1000 of 1000` and exits normally. The damage shows only when the memory past the end belongs to another allocation; `compute-sanitizer` reports the first bad read, by thread 232 of block 3, which is `i = 1000` ([post 11](../11-profiling-debugging/index.md)).
- **A block larger than 1,024 threads.** The launch does not run at all, and the only sign is the error that `cudaGetLastError` returns, `invalid configuration argument`; a program that never checks it copies its input back unchanged and carries on.
- **A build for the wrong architecture.** A binary built only for `sm_86` has no code the GTX 1650 can run, and the launch fails with `no kernel image is available for execution on the device`.
- **A host pointer passed to a kernel.** The kernel dereferences an address that means nothing on the device, and the launch fails with `an illegal memory access was encountered`. Keeping `h_` and `d_` prefixes on every pointer, as post 02 does, makes the mix visible in the code.

## 10. Summary

A kernel describes one thread's work; each thread finds its element with `blockIdx.x * blockDim.x + threadIdx.x` and is guarded by `if (i < N)` because blocks come whole. The GPU runs blocks on SMs in warps of 32 that share an instruction and hides latency by switching warps, so fast programs expose many threads, keep warp control flow uniform, and minimize trips to global memory.

---

## Common pitfalls

1. **Thinking the GPU runs a program faster.** It does not speed up one thread; it runs many. Serial or branch-heavy code can be slower on a GPU than on a CPU.
2. **Expecting blocks to cooperate.** Blocks have no guaranteed order and no barrier between them inside a kernel; `__syncthreads()` synchronizes one block only, and coordination across blocks needs a second launch or atomics ([post 09](../09-synchronization/index.md)).
3. **Branching on the lane.** A condition like `i % 2` splits every warp and serializes both paths; a condition that whole warps agree on costs nothing extra.
4. **Picking a block size that is not a multiple of 32.** Threads are scheduled in whole warps, so a block of 100 threads occupies four warps, 128 lanes, and wastes 28. Start with 128 or 256 and measure; the best size also depends on the kernel's register and shared-memory use.
5. **Launching too few threads to hide latency.** A grid of a few blocks leaves most SMs idle and each SM with too few warps to switch between.

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, consulted October 2026. The CPU against GPU trade, SIMT and divergence, just-in-time compilation, and the compute capability 7.5 limits and throughputs quoted here.
- Harris, M., "An Even Easier Introduction to CUDA" (NVIDIA Technical Blog, 25 January 2017). A gentle first pass over kernels, launches, and the global index.
- NVIDIA, *CUDA C++ Best Practices Guide*, consulted October 2026. Where the advice on block sizes, divergence, and memory access is spelled out.
- Patterson, D. and Hennessy, J., *Computer Organization and Design: The Hardware/Software Interface*. The architectural background for latency against throughput.
- `cmp-002` (How computers work, in the compute series): how a CPU executes a program, the machine the GPU is contrasted with here.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 02 - Your first CUDA kernel: vector addition and the memory wall](../02-vector-addition/index.md):** turns this model into a complete program, with the host and device copies, the guard, and an honest benchmark.
- **[Post 09 - When threads share data: barriers, atomics, and the memory model](../09-synchronization/index.md):** the rules for threads that must cooperate, and why the block independence of section 4 rules out a barrier across the grid.
