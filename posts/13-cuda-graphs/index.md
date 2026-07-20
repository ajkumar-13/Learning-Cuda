# 13 · Cutting launch overhead — CUDA Graphs and stream-ordered allocation

> **TL;DR.** Each kernel launch costs the host a few microseconds of driver work, so a workload of many tiny kernels is bottlenecked not by the GPU but by the CPU issuing the launches: exactly the blind spot the roofline of [Post 10](../10-roofline/index.md) warned about. **CUDA Graphs** fix this by recording a stream's work once (`cudaStreamBeginCapture` to `cudaStreamEndCapture`) into a graph of nodes with dependencies, instantiating it once, then replaying it many times with a single `cudaGraphLaunch`, so the launch cost is paid once and amortized. A second, quieter stall is **allocation**: `cudaMalloc` and `cudaFree` synchronize the whole device, freezing unrelated streams, while **stream-ordered allocation** (`cudaMallocAsync` drawing from a memory **pool**) hands back a reused block with no device sync. Combined with the streams of [Post 12](../12-cuda-streams/index.md), graphs plus pooled reuse turn a launch-bound loop into a replay-bound one.
>
> **After reading this you will be able to:**
> - Explain why a workload of many tiny kernels is bound by host launch overhead, not compute.
> - Capture a stream of work into a CUDA graph and replay it with a single call.
> - Say when graphs help (many small, repeated launches) and when they do not (a few large kernels).
> - Replace device-synchronizing `cudaMalloc`/`cudaFree` with a stream-ordered memory pool.

![A CPU timeline of many short kernel launches separated by gray host gaps, versus one packed green graph replay with the gaps closed.](diagrams/01-launch-overhead.svg)
*Direct launches leave a host gap before every kernel; a graph replays the whole sequence with one call, closing the gaps.*

---

## 1. The cost the roofline ignores

The roofline of [Post 10](../10-roofline/index.md) drew a ceiling for a kernel in steady state, and it was explicit about one thing it does not model: a stream of many tiny kernels, where the fixed cost the CPU pays to launch each one dominates the arithmetic. This post is that regime.

Picture the timeline. A `kernel<<<...>>>` is not free on the host. Recall from [Post 01](../01-introduction-to-cuda/index.md) that the CPU is the host that *launches* work and the GPU is the device that *runs* it, so every launch is a real host-to-device handoff: the driver validates the arguments, packs a launch command, and pushes it onto the stream, and only then does the GPU pick it up. That host-side work costs on the order of a few microseconds per launch. For one big kernel it is a rounding error. For ten thousand small ones it is the whole bill, and the GPU sits half-idle waiting for the CPU to feed it.

## 2. Where the microseconds go

Model one iteration as two costs: a host launch overhead `L` the CPU pays to issue the kernel, and the device work `W` the kernel actually does. Doing `N` iterations as `N` separate launches costs:

$$T_\text{direct} = N \,(L + W)$$

When `W` is large this is just `N \cdot W`, the compute you wanted. When `W` is small (a tiny kernel) the `L` term takes over, and you are paying the CPU to launch rather than the GPU to compute. A representative `L` is a few microseconds; take $L = 5$ microseconds as an illustrative figure (the true value depends on driver, kernel, and argument count). At that rate ten thousand launches burn 50 milliseconds of pure overhead before a single useful arithmetic operation.

The roofline cannot see this because `L` is not in it: intensity and bandwidth describe steady-state throughput, and `L` is a per-launch constant that vanishes as `W` grows. The escape is not to make each launch cheaper but to stop paying `L` per launch at all.

## 3. CUDA Graphs: capture once, replay many

A **CUDA graph** is a recording of the work you would otherwise launch piece by piece: a set of nodes (kernels, memory copies, memsets) with the dependency edges between them.

![A small directed graph: an H2D copy node feeding two kernel nodes that both feed a reduce node, with dependency arrows, the whole graph boxed as one replayable unit.](diagrams/02-graph-dag.svg)
*A graph is nodes (kernels, copies) plus their dependency edges, instantiated once and launched as a single unit.*

You rarely build the graph by hand. Instead you **capture** it: put a stream into capture mode, issue your normal stream code, and end capture. Every operation you would have launched is recorded as a node instead of executed, and the stream ordering becomes the graph's edges:

```cuda
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<g, b, 0, stream>>>(...);      // recorded as a node, not launched
kernelB<<<g, b, 0, stream>>>(...);      // edge A -> B, from the stream order
cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t exec;
cudaGraphInstantiate(&exec, graph, 0);   // pay the setup ONCE (CUDA 12+ signature)
for (int i = 0; i < N; ++i)
    cudaGraphLaunch(exec, stream);       // replay the whole graph with one call
```

`cudaGraphInstantiate` does the expensive work once: it validates the whole graph, resolves the dependencies, and bakes a launchable executable. After that, each `cudaGraphLaunch` submits the entire sequence with a single host call, so the launch overhead is paid once at build time instead of `N` times in the loop. Replaying `N` iterations costs:

$$T_\text{graph} = T_\text{build} + N \,(\ell + W)$$

where $T_\text{build}$ is the one-time capture-and-instantiate and $\ell$ is the tiny per-replay host cost, with $\ell \ll L$. If only the parameters change between replays (a new pointer, a new scalar) you do not recapture: `cudaGraphExecUpdate` patches the instantiated graph in place, far cheaper than rebuilding it.

## 4. When graphs help, and when they do not

The whole story is in the crossover. The `snippets/graphs_model.py` script (standard library, no GPU) sweeps the kernel size `W` and compares `N` direct launches against one graph replayed `N` times, with the illustrative costs $L = 5$ microseconds, per-replay $\ell = 0.5$ microseconds, a one-time build of 40 microseconds, over `N = 1000` iterations:

```bash
python snippets/graphs_model.py
```

```text
launch-overhead model  (illustrative: L=5.0 us/launch, replay=0.5 us, build=40 us, N=1000)

  W (us)   direct (us)   graph (us)   speedup   verdict
  ------------------------------------------------------------
     0.5       5500.0       1040.0     5.29x   graphs win big
     1.0       6000.0       1540.0     3.90x   graphs win big
     2.0       7000.0       2540.0     2.76x   graphs win big
     5.0      10000.0       5540.0     1.81x   graphs help
    20.0      25000.0      20540.0     1.22x   graphs help
   100.0     105000.0     100540.0     1.04x   negligible
   500.0     505000.0     500540.0     1.01x   negligible

  max speedup as W -> 0            : 9.26x  (tiny kernels, pure host overhead)
  W where speedup falls to 1.1x    : 44.1 us  (bigger kernels: graphs stop mattering)

allocation model  (illustrative: malloc=20 us, free=15 us, pool op=0.3 us, N=1000)

  strategy                        total (us)   per-iter (us)   note
  --------------------------------------------------------------------------
  cudaMalloc/cudaFree each iter      35000.0          35.00   blocks: device-wide sync
  cudaMallocAsync pool (reuse)         320.0           0.32   stream-ordered, amortized ~0

  pool speedup on allocation      : 109.4x
  host time returned to other work: 34.7 ms over 1000 iters
```

Read the top table down the `W` column. For a 0.5-microsecond kernel the graph is 5.29x faster, because almost all of $T_\text{direct}$ was launch overhead and the graph deletes it. By `W = 5` microseconds the kernel work rivals the launch cost and the win falls to 1.81x; by `W = 100` microseconds the launches are a rounding error and the graph buys 1.04x, essentially nothing. The closed form makes the boundary exact: the speedup is `(L + W) / (ℓ + T_build/N + W)`, which tops out at 9.26x as `W` approaches zero and drops to 1.1x once `W` passes about 44 microseconds. That is the rule of thumb: graphs pay off for many small, repeated launches (inference, iterative solvers, a fixed per-step kernel sequence) and do essentially nothing for a few large kernels.

The companion `snippets/graphs.cu` runs the same experiment on a real GPU: it times `N` direct launches of a deliberately tiny kernel, then captures that kernel into a graph and replays it `N` times, printing the measured speedup (which, being launch-overhead bound, depends on your GPU and driver). Build and run it with:

```bash
nvcc -O3 -arch=sm_75 snippets/graphs.cu -o graphs && ./graphs
```

## 5. The other stall: cudaMalloc versus a pool

Graphs remove the launch tax; a second tax hides in allocation. `cudaMalloc` and `cudaFree` are not stream-ordered: to keep memory safe they **synchronize the whole device**, waiting for every outstanding kernel on every stream to finish before they return. Call one inside a per-iteration loop and it stalls not just that loop but the overlap you built in [Post 12](../12-cuda-streams/index.md), and it cannot be captured into a graph at all.

![On the left a cudaMalloc call freezes every stream behind a red device-wide barrier; on the right a memory pool hands the same reused block straight back in green with no barrier.](diagrams/03-stream-ordered-alloc.svg)
*cudaMalloc synchronizes the whole device; a stream-ordered pool returns a reused block without stalling other streams.*

The fix is **stream-ordered allocation**. `cudaMallocAsync` and `cudaFreeAsync` take a stream and are ordered within it like any other operation, drawing memory from a **pool** the driver keeps around. The first allocation grows the pool with a real (blocking) reservation; every allocation after that reuses a freed block from the pool for almost nothing, and a `cudaFreeAsync` returns its block to the pool rather than to the OS, so the next iteration finds it waiting. The bottom table of the model output makes the gap concrete: allocating and freeing every iteration costs 35.00 microseconds per iteration of blocking device syncs, while the pool amortizes to 0.32 microseconds, a 109x cut that hands roughly 34.7 milliseconds back to useful work over 1000 iterations.

```cuda
float* d_tmp;
cudaMallocAsync(&d_tmp, bytes, stream);   // stream-ordered: no device-wide sync
kernel<<<g, b, 0, stream>>>(d_tmp, n);
cudaFreeAsync(d_tmp, stream);             // returns the block to the pool for reuse
```

Simpler still when the size is fixed: allocate once before the loop and reuse the buffer. That is the amortized-zero ideal, and the pool approximates it automatically when the sizes vary from step to step.

## 6. Putting it together

The three system-level tools compose into one pattern for a steady-state loop. Use **streams** ([Post 12](../12-cuda-streams/index.md)) to overlap copy and compute across the GPU's independent engines. Allocate the working buffers **once** (or from a stream-ordered pool) so no `cudaMalloc` stalls the pipeline mid-flight. Then **capture** one iteration of that streamed, pre-allocated work into a graph and **replay** it, so the CPU issues one `cudaGraphLaunch` per step instead of a dozen separate launches. The overlap is recorded into the graph and replayed for free, the launch overhead is paid once, and the allocator never stalls. This is how an inference server or an iterative solver runs the same kernel sequence thousands of times at close to the device's real ceiling, the ceiling the roofline drew back in [Post 10](../10-roofline/index.md).

A graph is not magic, though: it captures a *fixed* topology. If the shape of the work changes every iteration (data-dependent control flow, a different kernel each step) a graph either needs `cudaGraphExecUpdate` for the parts that changed or does not apply at all, and you are back to direct launches. Graphs reward repetition.

---

## Common pitfalls

- **Reaching for graphs when the kernels are large.** Graphs amortize launch overhead. If each kernel already runs for tens of microseconds or more, the launch cost is noise and a graph buys almost nothing; measure the kernel size first.
- **`cudaMalloc` or `cudaFree` inside the capture or the loop.** They synchronize the device, cannot be captured, and stall every stream. Allocate up front or use `cudaMallocAsync`; a single stray free is a silent pipeline bubble.
- **Recapturing every iteration.** Rebuilding and re-instantiating a graph each step pays back the very setup cost you were trying to amortize. Instantiate once, then use `cudaGraphExecUpdate` to change parameters.
- **Assuming a graph adapts to new shapes.** A captured graph has a fixed topology and fixed launch dimensions. Data-dependent control flow or changing grid sizes need an update or a fresh capture.
- **Capturing on the legacy default stream.** Capture must run on a non-default (or per-thread default) stream. Legacy default-stream work cannot be captured and will error or serialize the whole device.
- **Trusting an illustrative speedup as a measurement.** The model's numbers (and the `.cu`'s printout) depend on the assumed `L` and on your GPU and driver. The *shape* of the crossover is the lesson, not the exact multiple.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — CUDA Graphs"* (current). The reference for capture, instantiation, nodes, and `cudaGraphExecUpdate` (reference).
- NVIDIA, *"Getting Started with CUDA Graphs"* (Developer Blog, 2019). The canonical capture-and-replay walkthrough and where it pays off (technical, foundational).
- NVIDIA, *"Using the NVIDIA CUDA Stream-Ordered Memory Allocator"* (Developer Blog, 2021). `cudaMallocAsync`, pools, and why plain `cudaMalloc` stalls the device (technical).
- NVIDIA, *"CUDA C++ Programming Guide — Stream Ordered Memory Allocator"* (current). The reference for pools, `cudaMallocAsync`/`cudaFreeAsync`, and pool trimming (reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 14, Kernel fusion](../14-kernel-fusion/index.md)**: the other answer to too-many-kernels, merging their work so there is less to launch at all.
- **[Post 12, CUDA streams](../12-cuda-streams/index.md)**: the overlap this post records into a graph and replays.
- **[Post 10, Roofline and occupancy](../10-roofline/index.md)**: the launch-overhead regime the steady-state roofline explicitly does not model.
