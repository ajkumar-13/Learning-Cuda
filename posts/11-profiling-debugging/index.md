# 11 · Profiling and debugging — Nsight, compute-sanitizer, and the occupancy you can measure

> **TL;DR.** The earlier posts kept making claims like "98% of peak bandwidth" and "profile your occupancy" without ever opening a tool, and this post pays that debt. The discipline is *profile, don't guess*: a loop of write, measure, read the bottleneck, fix, where the step beginners skip is measure. You will learn the three tools that close it: `compute-sanitizer` to prove the kernel is correct (it catches the missing `if (i < N)` guard and shared-memory races), Nsight Systems (`nsys`) for the timeline, and Nsight Compute (`ncu`) for per-kernel detail like achieved memory throughput and **occupancy**. Occupancy and the roofline are the *model* of [Post 10](../10-roofline/index.md); here you measure the achieved numbers with `ncu` and read the gap to the ceiling that model drew.
>
> **After reading this you will be able to:**
> - Run the *profile, don't guess* loop instead of guessing at performance.
> - Catch out-of-bounds writes and shared-memory data races with `compute-sanitizer`.
> - Choose between Nsight Systems for the timeline and Nsight Compute for per-kernel detail, and run each.
> - Measure achieved occupancy and memory throughput, and read them against the roofline model of Post 10.

![A four-stage clockwise cycle of boxes: write the kernel, then measure with a profiler highlighted in green, then read the bottleneck asking memory- or compute-bound, then fix the one limiter named, with an arrow looping from fix back to measure.](diagrams/01-profiling-loop.svg)
*You cannot tune what you cannot see. Measurement is the hinge of the loop, and every fix is re-measured before you trust it.*

---

## 1. Why measure: you cannot tune what you cannot see

Every post so far ended with a performance claim. Post 02 said vector addition runs "near peak bandwidth"; post 03 said tiling moves a kernel "toward compute-bound"; post 06 said a padded transpose "recovers about 88% of a plain copy's bandwidth." Those are not opinions. They are *measurements*, and until now the series asked you to take them on faith. This post opens the tools that produce them.

The mental model is the **roofline** of [Post 10](../10-roofline/index.md): a kernel is **memory bound** or **compute bound** according to its arithmetic intensity, and the cure differs for each. Speeding up the arithmetic in a memory-bound kernel does nothing; you were never waiting on arithmetic. That post *derived* the ceilings by counting; this one *measures* how close a real kernel gets, and reads why it falls short.

> **The trap of guessing.** The single most common waste of an afternoon is "optimizing" the wrong roof: hand-unrolling loops in a kernel that is stalled on memory, or chasing memory layout in one that is starved for warps. The tools exist so you stop guessing which roof you are under. You write the kernel, *measure*, read what the measurement says the bottleneck is, fix exactly that one thing, then measure again. That loop is the hero image above, and the rest of this post is the three tools that turn each arrow into a command.

## 2. Correctness first: `compute-sanitizer`

A fast wrong kernel is worthless, and GPU bugs are quieter than CPU bugs: an out-of-bounds write often corrupts a neighbor silently instead of crashing. So the loop starts with correctness, and the tool is `compute-sanitizer`, the CUDA 12 successor to the old `cuda-memcheck` (it ships with the CUDA 12.x toolkit). It runs your unmodified binary under instrumentation and reports the exact kernel, thread, and address of a fault.

It has three sub-tools you will reach for:

| Tool | Flag | Catches | The post it ties to |
|---|---|---|---|
| memcheck | `--tool memcheck` (default) | out-of-bounds and misaligned access, leaks | the missing `if (i < N)` from post 02 |
| racecheck | `--tool racecheck` | shared-memory data races | the missing `__syncthreads()` from posts 04 and 09 |
| synccheck | `--tool synccheck` | illegal barrier or warp-sync use | divergent `__syncthreads()` |

The picture is the bug itself. When you forget the boundary guard, the surplus threads (the grid is launched in whole blocks, so it is almost always larger than `N`) write one past the end of the array:

![A row of eight threads above a six-cell array; threads 0 to 5 write into valid cells with gray arrows, while the surplus threads 6 and 7 send red arrows into a dashed out-of-bounds box past the last cell.](diagrams/02-sanitizer-catch.svg)
*The surplus threads with no element to own scribble past the array. This is the bug `compute-sanitizer --tool memcheck` reports as an invalid global write, naming the thread and address.*

Take the post 02 kernel with its guard deleted:

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];     // BUG: no `if (i < N)`; surplus threads write past the end
}
```

Build and run it under the tool:

```bash
compute-sanitizer --tool memcheck ./vector_add
```

The output points straight at the line: `Invalid __global__ write of size 4 bytes`, with the kernel name, the offending thread index, and the address relative to the allocation. You did not have to add a single `printf`. For a shared-memory race (a tile written by one thread and read by another with no `__syncthreads()` between, the classic post 04 / post 09 bug), swap the tool: `compute-sanitizer --tool racecheck ./reduction` reports the two accesses that race and the barrier that should separate them.

> **Run the sanitizer before the profiler, every time.** Profiling a buggy kernel measures the wrong thing: an out-of-bounds access can inflate memory traffic or, worse, give a "fast" number for a kernel that is silently wrong. Correctness is step zero of the loop. The runnable demo for this post, [snippets/occupancy_demo.cu](snippets/occupancy_demo.cu), already passes clean under `compute-sanitizer ./occupancy_demo`; break its guard on purpose to watch the tool catch it.

## 3. Two profilers: a timeline and a microscope

Once the kernel is correct, two profilers answer two different questions, and reaching for the wrong one wastes time.

**Nsight Systems (`nsys`) is the timeline.** It shows the whole application across time: when each kernel runs, when each `cudaMemcpy` runs, and crucially whether copy and compute *overlap*. That overlap is the entire payoff of the CUDA streams you will meet in [post 12](../12-cuda-streams/index.md); `nsys` is how you confirm it actually happened instead of serializing. It is a wide, shallow view: which thing ran when, on which engine.

```bash
nsys profile -o timeline ./occupancy_demo
```

**Nsight Compute (`ncu`) is the microscope.** It re-runs a single kernel many times under hardware counters and reports everything about *that one kernel*: achieved DRAM throughput as a percentage of peak, achieved versus theoretical occupancy, and the warp-stall reasons (was the warp waiting on a memory load, a barrier, or an arithmetic dependency?). It is deep and narrow, and it is slow, so you point it at the one kernel the timeline flagged.

```bash
ncu --set full -o kernel ./occupancy_demo
```

> **The division of labor, in one line.** Use `nsys` to find *which* kernel or copy is the problem across the whole run; use `ncu` to find *why* that one kernel is slow. Timeline first to localize, microscope second to diagnose. Running `ncu --set full` over a whole application is a way to wait a very long time for data you did not need.

## 4. Occupancy: measuring what the model predicts

[Post 10](../10-roofline/index.md) derived **occupancy** (the fraction of an SM's warp slots you fill) and the three resources that cap it: block size, registers, and shared memory. The profiler's job is to supply that model's inputs and check its output.

Ask the compiler for a kernel's register and shared-memory cost with the verbose flag:

```bash
nvcc -O3 -arch=sm_75 -Xptxas=-v snippets/occupancy_demo.cu -o occupancy_demo
```

`-Xptxas=-v` prints a line like `Used 32 registers, 0 bytes smem` per kernel. Feed those two numbers and your block size into the Post 10 calculator and you have the *theoretical* occupancy ceiling before you launch. Then `ncu` reports the **achieved** occupancy: when it falls short of theoretical, the cause is usually tail effects or uneven block scheduling, not the resource math. To *cap* registers and trade a few spills for more warps, annotate the kernel with `__launch_bounds__(maxThreadsPerBlock)`, as the demo's `saxpy` does; the runtime also exposes `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the same arithmetic at run time.

> **More occupancy is not always the goal.** [Post 10](../10-roofline/index.md) makes the full case: the returns flatten past roughly 50% to 75%, forcing more warps causes register **spills** to slow local memory, and a famously fast kernel can run at 25% on instruction-level parallelism. Measure achieved occupancy and the stall reasons in `ncu`; do not chase 100% as a number.

## 5. Reading the roofline in the profiler

[Post 10](../10-roofline/index.md) drew the **roofline** and placed the series' kernels on it by arithmetic intensity. `ncu` is where you confirm which roof a real kernel actually hit. For a memory-bound kernel like the demo's SAXPY (a single fused multiply-add, `y = a*x + y`) it reports a high "Memory Throughput" percentage, a low "Compute (SM) Throughput," and a long-scoreboard stall as the dominant reason: the model's memory-bound verdict, read off a counter instead of predicted. For a tiled matmul those two percentages shift toward compute, exactly as raising the arithmetic intensity predicted. The profiler does not replace the model; it measures the gap between the ceiling the model drew and the performance you got, and that gap is the only number that says whether more tuning is worth it.

## 6. The build-and-measure recipe

From this post's directory, the CUDA demo is real and runnable on a machine with a GPU and CUDA 12.x. Build it with the verbose flag, check it, then profile it:

```bash
nvcc -O3 -arch=sm_75 -Xptxas=-v snippets/occupancy_demo.cu -o occupancy_demo
compute-sanitizer ./occupancy_demo     # correctness, step zero
nsys profile -o timeline ./occupancy_demo
ncu --set full -o kernel ./occupancy_demo
```

The theoretical ceilings to compare against come from the Post 10 model (`python ../10-roofline/snippets/roofline_model.py`), which needs no GPU. Any GPU timing you read from `nsys` or `ncu` is specific to your card, so treat the absolute milliseconds as representative and the *shape* (which roof, which limiter, which stall) as the lesson, exactly as post 02 did with its benchmark table. The setup for the toolkit is in [SETUP.md](../../SETUP.md), and how to run the series' snippets is in [RUNNING.md](../../RUNNING.md).

---

## Common pitfalls

- **Profiling before the sanitizer is clean.** A kernel with an out-of-bounds write can post a fast, meaningless number. Run `compute-sanitizer` first, every time; correctness is step zero of the loop.
- **Timing the first launch.** The first kernel pays one-time context and just-in-time compilation costs. Run an untimed warm-up (the demo does), then let the profiler measure the second launch, never a host wall-clock that ignores the asynchronous launch.
- **Running `ncu --set full` over a whole application.** Nsight Compute re-runs each kernel under counters and is slow. Localize with `nsys` first, then point `ncu` at the one kernel that matters.
- **Chasing 100% occupancy.** Occupancy hides latency with diminishing returns; about 50% to 75% is usually enough. Forcing more warps by cutting registers causes spills to slow local memory that cost more than they save.
- **Optimizing the wrong roof.** Hand-tuning arithmetic in a memory-bound kernel (or memory layout in a compute-bound one) does nothing. Read the roofline and the stall reasons first, then fix the limiter the measurement actually named.
- **Trusting `-Xptxas=-v` register counts across architectures.** Register usage is per target. Numbers printed for one `-arch` need not hold for another, so measure for the architecture you deploy on, here `sm_75`.

---

## Further reading

- NVIDIA, *"Nsight Compute Documentation"* (current). The per-kernel profiler: counters, occupancy, and stall-reason metrics (technical, reference).
- NVIDIA, *"Nsight Systems Documentation"* (current). The system-wide timeline for copy/compute overlap and kernel scheduling (technical, reference).
- NVIDIA, *"Compute Sanitizer Documentation"* (current). The CUDA 12 correctness tool with memcheck, racecheck, and synccheck (technical, reference).
- NVIDIA, *"CUDA C++ Best Practices Guide"* (current). The reference for the profile-driven optimization loop and memory throughput targets (technical, reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 10, Roofline and occupancy](../10-roofline/index.md)**: the model this post measures against; re-read the ceilings now that you have the tools to check how close a kernel gets.
- **[Post 12, CUDA streams](../12-cuda-streams/index.md)**: where the Nsight Systems timeline pays off, confirming that copy and compute actually overlap.
- **[Post 03, Matrix multiplication](../03-matrix-multiplication/index.md)**: re-read the tiling result as a roofline move, now that you can place both kernels under their roof.
