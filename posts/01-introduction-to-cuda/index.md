# 01 · Why GPUs exist — the CUDA execution model from the ground up

> **TL;DR.** A central processing unit (CPU) is built to finish *one* instruction stream as fast as possible, so most of its chip area is cache and control logic, whereas a graphics processing unit (GPU) devotes far more of its transistors to arithmetic cores and runs the *same* instruction across thousands of data elements at once. To program it you write a **kernel** (one function, from the point of view of one element) and launch a **grid** of threads, organized into **blocks**; each thread finds its own data with `blockIdx.x * blockDim.x + threadIdx.x`. The hardware runs threads in lockstep groups of 32, called **warps**, and feeds them from a steep **memory hierarchy** (registers, shared memory, L2, global). This post builds that whole mental model, so the kernels in the rest of the series are never a mystery.
>
> **After reading this you will be able to:**
> - Explain why a GPU beats a CPU on data-parallel work, in terms of chip area and latency hiding.
> - Map the software hierarchy (thread, block, grid) onto the hardware (warp, streaming multiprocessor).
> - Compute any thread's global index by hand, and say what the surplus threads do.
> - Predict when **warp divergence** will roughly halve your throughput, and avoid it.

![A CPU spends most of its area on a few large cores plus cache and control, while a GPU is a dense grid of small compute cores; the ratio of compute to control is the whole difference.](diagrams/01-cpu-vs-gpu.svg)
*The GPU is not a faster CPU. It is a different trade: less control per thread, far more threads.*

---

## 1. The motivation: two different bets on where the time goes

A CPU core is a *latency* machine: it assumes your program is one stream of instructions where each step often depends on the result of the step before it, and where the path can change at any branch (any `if`). It spends transistors making *that* stream finish fast — large caches so it rarely waits on slow memory, branch prediction so it can guess which way an `if` will go and keep working, and out-of-order execution so one slow instruction does not stall everything queued behind it. A typical CPU has only a handful of these powerful cores.

A GPU makes the opposite bet. It assumes the work is *data-parallel*: the same operation applied to millions of independent elements, like `C[i] = A[i] + B[i]`. For that workload, per-thread cleverness is wasted; raw arithmetic throughput is everything. So the GPU spends its area on thousands of small cores, with far less control logic devoted to each one, as the diagram shows. When a thread stalls waiting on memory, the GPU does not try to predict around the stall; it simply switches to another group of threads that is ready to run (such a group is a **warp**, defined in section 4). With enough threads in flight, the memory latency is hidden behind useful work.

> **What is a kernel?** A **kernel** is a function that runs on the GPU, written from the point of view of *one* element. You do not write a loop over the data. You write what a single thread does, then launch thousands of copies of it. The launch syntax `kernel<<<blocks, threads>>>(...)` is the one piece of CUDA that is not ordinary C++.

The rule of thumb falls straight out of this: serial or heavily branching work belongs on the CPU; large, regular, data-parallel work (vector and matrix math, the heart of deep learning) belongs on the GPU.

## 2. The programming model: host and device

CUDA is **heterogeneous**: the CPU (the *host*) and the GPU (the *device*) work together, each with its own memory. A program runs on the host, copies the data it needs into device memory, launches a kernel to run on the device, and copies the results back. The full allocate, copy, launch, copy, free cycle is the subject of [Post 02](../02-vector-addition/index.md); here only the shape matters: the host orchestrates, the device computes, and the two memories are separate.

> **Reference card.** Every concrete number in this series is measured on one **NVIDIA GeForce GTX 1650** (Turing, **compute capability 7.5**), unless a post says otherwise. Compute capability names the GPU's feature set and limits, so when you see `sm_75` in a build command or a specific cycle count below, that is the card it refers to. Your own GPU's numbers will differ; [`snippets/device_query.cu`](snippets/device_query.cu) prints yours.

## 3. The hierarchy: thread, block, grid

A launch does not give you a flat pool of threads. CUDA organizes them in two levels, and the index arithmetic depends on it.

> **Three words, defined once.** A **thread** runs one copy of the kernel for one element. A **block** is a group of threads (up to 1024) that run on the same streaming multiprocessor and can share fast on-chip memory and synchronize with `__syncthreads()`. A **grid** is all the blocks from one launch. Blocks must be able to run *independently and in any order*, which is exactly what lets one binary scale from a tiny laptop GPU to a datacenter card.

Each thread finds its own data by flattening its two-level position into a single global index:

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

![A grid of four blocks; block 1 zooms into its eight threads; thread t5 computes its global index as 1 times 8 plus 5 equals 13.](diagrams/02-thread-hierarchy.svg)


For the thread the diagram highlights, that is just arithmetic:

```text
blockIdx.x  = 1   (its block)
blockDim.x  = 8   (threads per block)
threadIdx.x = 5   (its lane within the block)
i = 1 * 8 + 5 = 13
```

The kernel reads its position from four built-in, read-only variables: `threadIdx` (this thread's lane within its block), `blockIdx` (which block it is in), `blockDim` (threads per block), and `gridDim` (how many blocks are in the grid — the one the index formula above does not need). Each carries `.x`, `.y`, `.z` components, so the same idea extends to 2D and 3D launches:

```cuda
// 2D grid of 2D blocks (used for matrices in Post 03)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Because you launch whole blocks, a grid almost always has more threads than elements. Those surplus threads still launch (you cannot un-launch them), but an `if (i < N)` guard makes them return without reading or writing anything, the subject of the next post.

Those are all the pieces of a real kernel, so here is one: everything from the last two sections, in five lines.

```cuda
__global__ void addOne(float *data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        data[i] += 1.0f;                             // one thread, one element
}
```

Do not worry yet about how `data` gets onto the GPU — that is [Post 02](../02-vector-addition/index.md). For now notice the shape: there is no loop over the array. Each thread computes its own index and touches a single element. That *is* the kernel mindset from section 1, made concrete.

> **Check your understanding.** Suppose `N = 1000` with 256 threads per block. How many blocks must launch, how many threads is that in total, and how many fail the `i < N` guard? Work it out before reading on.

The launch needs `ceil(1000 / 256) = 4` blocks, which start `4 * 256 = 1024` threads, so the last `1024 - 1000 = 24` threads compute an index of 1000 or more and the guard makes them return without touching memory. That ceiling division and boundary guard are the exact pattern [Post 02](../02-vector-addition/index.md) turns into a real kernel.

## 4. The hardware: streaming multiprocessors and warps

The GPU is a collection of **streaming multiprocessors (SMs)**. When you launch a grid, the hardware hands whole blocks to available SMs; several blocks can share an SM if resources allow. Inside an SM, threads do not run one at a time, and they do not run fully independently. They run in **warps**: groups of 32 threads in *lockstep*, which means the hardware issues one instruction and all 32 lanes carry it out together, each on its own data. This is the **SIMT** model: single instruction, multiple threads. That ties the two hierarchies together: a **grid** of **blocks** (software) is spread across the GPU's **SMs** (hardware), and within an SM the **threads** of a block execute as **warps** of 32 lanes — the "small cores" from section 1.

That lockstep is free when every thread in a warp agrees on the control flow. It is not free when they disagree:

```cuda
if (i % 2 == 0) doA();   // even lanes
else             doB();  // odd lanes
```

Both branches exist in the same warp, so the hardware runs them **one after the other**: first the even lanes execute `doA()` while the odd lanes sit idle, then the odd lanes execute `doB()` while the even lanes sit idle. This is **warp divergence**: when the two sides do comparable work, a fully split warp costs roughly twice the time (a lopsided split, or more than two paths, changes the exact factor).

![A warp of lanes: when the whole warp agrees it is one pass; when it splits on i mod 2, pass one runs the even lanes with the odd lanes idle and pass two runs the odd lanes, about twice the work.](diagrams/03-warp-divergence.svg)
*Divergence is not a crash, it is waste: the idle lanes still occupy the hardware while the other half runs.*

A small CPU model makes the cost concrete. It groups threads into warps of 32 and counts execution passes (a warp that agrees is one pass; a warp that splits is two). It is a control-flow model, not a timing simulator — it isolates only the extra passes divergence adds, and says nothing about memory latency or the cost of individual instructions. It lives at [snippets/thread_model.py](snippets/thread_model.py), needs only the standard library, and runs anywhere (no GPU required):

```bash
python snippets/thread_model.py
```

```text
thread model
  launch            = 4 blocks x 32 threads = 128 threads
  N                 = 120  (surplus 8 guarded by i < N)
  covers 0..N-1 once= True
  warps             = 4  (32 threads each)
  divergence (passes over all warps, lower is better):
    no branch              = 4
    branch on (i % 2)      = 8   -> every warp splits, ~2x cost
    branch on whole warp   = 4   -> no divergence
```

The lesson: branch on conditions that are uniform across a warp, not on a thread's lane.

> **The mental model so far.** You launch a **grid**. A grid holds **blocks**, and the hardware hands each block to a **streaming multiprocessor (SM)**. A block holds **threads**, and inside the SM those threads run in **warps** of 32 lanes, in lockstep. One caution about the "small cores" picture from section 1: a lane is not something a single thread keeps for its whole life — the SM shares its lanes across many warps over time. Software says *grid → block → thread*; the hardware runs it as *SM → warp → lane*.

## 5. The memory hierarchy

A GPU has several kinds of memory, and the gap between fastest and slowest is enormous: registers answer in about one cycle, global memory (off-chip DRAM) in hundreds. The tiers form a pyramid, fast and tiny at the top, slow and huge at the bottom. The per-tier cycle counts in the diagram below are order-of-magnitude, illustrative figures (the exact latencies vary by architecture); what matters is the *ratio* between tiers, not the precise numbers.

![A pyramid of memory tiers: registers, then shared memory and L1, then L2, then global DRAM, getting bigger and slower toward the bottom.](diagrams/04-memory-hierarchy.svg)
*Most large inputs and outputs live in global memory and must climb toward the cores before use. Fast kernels minimize those trips.*

The tiers, and who can see each one:

| Tier | Scope (who shares it) | Speed | What it holds |
|---|---|---|---|
| Registers | one thread | fastest (~1 cycle) | a thread's private variables |
| Shared memory | one block | very fast, on-chip | a scratchpad you fill and reuse by hand |
| L2 cache | whole GPU | medium | a hardware-managed cache over global memory |
| Global memory (DRAM) | whole GPU | slowest (hundreds of cycles) | the card's main memory, where inputs and outputs live |

(The diagram pairs shared memory with L1 because both sit on-chip at the same speed tier; the difference is that L1 is hardware-managed and shared memory is not.)

The one tier you control by hand is **shared memory**: a small, per-block scratchpad with on the order of ten times the bandwidth of global memory (the exact ratio is hardware-dependent). A high-performance kernel loads a tile of data from global memory into shared memory once, then reuses it many times. That single idea (load once, reuse from on-chip memory) is the engine behind tiled matrix multiply, reductions, convolution, and almost every optimization later in the series.

## 6. From code to the GPU: nvcc, PTX, and compute capability

The CUDA compiler `nvcc` can emit a kernel in two forms. **PTX** (Parallel Thread Execution) is a portable virtual assembly that the driver just-in-time compiles for *future* GPUs. **cubin** (CUDA binary) is native machine code for one *specific* architecture (`sm_75` for Turing, `sm_86` for Ampere). Which of the two end up in your program depends on the build flags, and a binary can carry both — so it runs today and still runs on hardware that did not exist when it was built. A GPU's **compute capability** (7.5, 8.6, ...) names its feature set and limits.

Before writing a kernel it is worth asking the device what it is. [snippets/device_query.cu](snippets/device_query.cu) prints the numbers the rest of the series reasons about. Build and run it with:

```bash
nvcc -O3 -arch=sm_75 snippets/device_query.cu -o device_query && ./device_query
```

(On Windows the binary is `device_query.exe`; run `.\device_query.exe` and match `-arch` to your card — read its compute capability as `sm_XX` with the dot dropped, so the GTX 1650's 7.5 becomes `sm_75`. See [SETUP.md](../../SETUP.md); every `nvcc` command in this series works the same way.)

On the GTX 1650 used throughout, it reports:

```text
Device 0: NVIDIA GeForce GTX 1650
  compute capability   : 7.5
  streaming multiproc. : 14 SMs
  warp size            : 32 threads
  max threads / block  : 1024
  shared mem / block   : 48 KB
  global memory        : 4.0 GB
  peak bandwidth       : ~192 GB/s
```

Those seven numbers (and especially the warp size of 32, the 1024-thread block limit, and the bandwidth) are the constraints every later kernel is tuned against.

---

## The model in three sentences

A CUDA kernel describes the work of *one* thread. A launch creates a grid of blocks, and each thread computes which element it owns with `blockIdx.x * blockDim.x + threadIdx.x`. The GPU schedules whole blocks onto SMs and runs their threads in warps of 32, so fast CUDA programs expose many independent threads, keep warp control flow uniform, and minimize trips through global memory.

---

## Common pitfalls

- **Thinking the GPU runs your code faster.** It does not speed up a single thread; it runs many. Code that is serial or branch-heavy can be *slower* on a GPU than a CPU.
- **Forgetting blocks must be independent.** There is no guaranteed order or concurrency *between* blocks, and no global barrier inside a kernel. Cross-block coordination needs a separate launch or atomics, not `__syncthreads()` (which only synchronizes within a block).
- **Ignoring warp divergence.** A branch that splits a warp serializes both paths. Branch on warp-uniform conditions where you can, and never assume an `if` is free.
- **Treating host and device pointers as interchangeable.** A device pointer dereferenced on the host (or vice versa) is a crash or silent corruption. Keep `h_` and `d_` prefixes and never mix them.
- **Picking a block size that is not a multiple of 32.** Threads are allocated in whole warps, so 100 threads occupy 128 lanes and waste 28. Start with 128 or 256 and benchmark from there; the best size depends on the kernel's register and shared-memory use.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide"* (current). The authoritative reference for the execution model, memory spaces, and compute capabilities (technical, reference).
- Harris, M., *"An Even Easier Introduction to CUDA"* (NVIDIA, 2017). A gentle first pass over kernels and the launch model (beginner-friendly).
- NVIDIA, *"CUDA C++ Best Practices Guide"* (current). Where the occupancy, divergence, and memory-coalescing advice is spelled out (technical).
- Patterson, D. & Hennessy, J., *"Computer Organization and Design"* (latency-vs-throughput chapter). The architectural background for why the two chips diverge (historical, foundational).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 02, Vector addition](../02-vector-addition/index.md)**: turn this model into your first real kernel, with memory management, the boundary guard, and honest benchmarking.
- **[Post 03, Matrix multiplication](../03-matrix-multiplication/index.md)**: the first kernel where the memory hierarchy pays off, using shared-memory tiling to become compute bound.
