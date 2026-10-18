# 13 - Cutting launch overhead: CUDA Graphs and stream-ordered allocation

> **TL;DR.** On the author's GTX 1650 under Windows, a stream of directly launched kernels pays about 1.5 µs per kernel beyond the kernel's own work, and 2,000 empty kernels take 1.80 µs each; captured once into a CUDA graph and replayed, the same kernels take 0.77 µs each, 2.34 times faster, a gain that falls to 1.06 times by 8 µs kernels. Allocation is the second stall: `cudaFree` held the host for 20.3 ms until a kernel on another stream had finished, while `cudaMallocAsync` and `cudaFreeAsync` on a stream-ordered memory pool returned in microseconds. The pool keeps its memory only when told to: at the default release threshold of 0, a step that synchronizes cost 2,004 µs, 3.9 times a `cudaMalloc` step, and at the maximum threshold 45.8 µs.
>
> **Prerequisites:** [post 12](../12-cuda-streams/index.md).
> **Safe to skip?** Skip it if the reader can already measure what one kernel launch costs, capture a stream into a graph and replay it, say from the kernel's length whether that pays, and run a per-step allocation from a stream-ordered pool that keeps its memory.
>
> **After reading, you will be able to:**
>
> - Explain why a stream of many short kernels is bound by the cost of each launch rather than by compute, and measure that cost per kernel.
> - Capture a stream's kernels into a CUDA graph between `cudaStreamBeginCapture` and `cudaStreamEndCapture`, instantiate it once, and replay it.
> - Use the launch-overhead model, fed with measured costs, to say when a graph helps, when it does not, and after how many kernels it repays its build.
> - Replace a per-step `cudaMalloc` and the device-synchronizing `cudaFree` with `cudaMallocAsync` and `cudaFreeAsync` on a pool that keeps its memory.

![Twenty 2 µs kernels launched directly end at 70.4 µs, each preceded by a 1.52 µs gap on the GPU while the host issues launches of 1.75 µs; replayed from a graph with one 5.6 µs host call, the gaps shrink to 0.81 µs and the last kernel ends at 56.2 µs.](diagrams/01-launch-overhead.svg)

*Every direct launch leaves a gap on the GPU before the next kernel; a replayed graph narrows each gap and issues all twenty kernels with one host call.*

---

## 1. The question: what does a launch cost?

The roofline of [post 10](../10-roofline/index.md) models a kernel in steady state and names what it leaves out: a stream of many short kernels, where the fixed cost of starting each one outweighs the work inside it. Post 10's example came from [post 02](../02-vector-addition/index.md): vector addition over 1,000 elements took 3.6 µs, about 2% of the card's bandwidth, because the time was the launch, not the bytes.

The central processing unit (CPU) is the host that launches work and the graphics processing unit (GPU) the device that runs it ([post 01](../01-introduction-to-cuda/index.md)), so every `kernel<<<grid, block, 0, stream>>>` is a handoff: the driver checks the arguments, builds a launch command and submits it to the stream's queue, and the GPU's front end fetches the command and starts the grid. The Programming Guide's section on CUDA Graphs names the cost: the host driver's operations, "necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued." For one long kernel it is noise. For an inference step or an iterative solver that runs the same few dozen short kernels thousands of times, it can be most of the time.

This post measures that cost on the author's GTX 1650, removes most of it with a CUDA graph, and removes a second stall, allocation, which blocks the host thread itself, with a stream-ordered memory pool. Both are costs of one machine's launch path, where the CPU, the driver and the operating system matter as much as the card, so they are measured here rather than quoted.

## 2. Measuring the cost of a launch

`snippets/graphs.cu` launches kernels whose device time $W$ is set rather than measured: one warp spins on `clock64()` for $W$ microseconds' worth of SM (streaming multiprocessor) cycles at the clock measured before the sweep, and $W = 0$ is an empty kernel. Each pass issues 2,000 kernels, 100 steps of 20, on one stream, and the host times two things with `std::chrono`: when the loop of launch calls returns (the issue time) and when the stream has finished (the done time), both per kernel. The same 20 kernels are then captured into a graph and replayed 100 times. Every row is the median of 7 passes, after about half a second of untimed launches. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, driver 545.84, Windows 10 with an Intel Core i7-9750H, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), CUDA driver API 12030, runtime 12030, memory pools supported
launch sweep: 2000 kernels per pass (100 steps of 20), one warp each; median of 7 passes; us per kernel
  W (us)   direct issue   direct done   graph issue   graph done   speedup
       0           1.71          1.80          0.28         0.77     2.34x
       2           1.75          3.52          0.28         2.81     1.25x
       4           1.76          5.54          0.29         4.79     1.15x
       8           1.76          9.33          0.29         8.79     1.06x
      16           1.78         17.43          0.28        16.83     1.04x
      32           1.90         33.59          0.32        32.86     1.02x
graph build (capture 20 empty kernels + instantiate): 59.9 us, once
```

These are measured numbers, host-side costs of this laptop's whole launch path, from a run whose SM clock read 1,743 MHz before the timed rows and 1,529 MHz after (section 9); the reader's machine will give other digits.

An empty kernel costs 1.80 µs per direct launch: 2,000 of them take 3.6 ms. Reruns on the same laptop gave 1.9 to 2.6 times for empty kernels and 1.02 to 1.04 times at 32 µs, so the digits move and the shape holds. The issue time stays near 1.75 µs whatever $W$ is, so for $W = 2$ the loop of calls returns after 3.5 ms while the stream needs 7.0 ms: the host runs ahead and the queue fills. Yet each kernel still costs about $W + 1.5$ µs (3.52, 5.54, 9.33, 17.43, 33.59). The overhead is therefore not only the issuing thread's time, which asynchronous launches hide behind the previous kernel: the GPU waits about 1.5 µs between consecutive kernels of a stream. That much is inferred from the two times; how the gap divides between the driver's submission and the front end starting each grid, this harness cannot say, and a trace with Nsight Systems needs administrator rights on this laptop ([post 11](../11-profiling-debugging/index.md)). The card runs under WDDM, the Windows Display Driver Model, in which the operating system schedules the GPU's work; another operating system, driver model or card gives other numbers, and the shape is what transfers.

The graph pays less on both sides. One `cudaGraphLaunch` issues 20 kernels, 0.28 µs of host time per kernel. On the GPU each kernel costs about $W + 0.8$ µs: the gap shrinks but does not vanish, because each node is still a grid the front end must start. The hero figure draws the $W = 2$ row from these averages: twenty kernels end at 70.4 µs launched directly and at 56.2 µs replayed.

A published figure shows the same shape on other hardware. Alan Gray's "Getting Started with CUDA Graphs" (NVIDIA Technical Blog, 5 September 2019) timed steps of 20 kernels of 2.9 µs on a Tesla V100 with CUDA 10.1: 3.8 µs per kernel launched directly, 3.4 µs from a graph, and 9.6 µs when the host synchronized after every kernel.

## 3. CUDA Graphs: capture once, replay many

A **CUDA graph** records work that would otherwise be launched piece by piece: nodes (kernels, copies, memsets, allocations) and the dependency edges between them. The Programming Guide splits its use into definition, instantiation and execution, and says what separating them buys: "CPU launch costs are reduced compared to streams, because much of the setup is done in advance".

The simplest definition is a **capture**: put a stream into capture mode, issue the usual stream code, end the capture. Each operation issued to the stream is recorded as a node instead of run, and the stream's order becomes the edges. The harness builds its graph so:

```cuda
// Capture one step of KERNELS launches into a graph and instantiate it: paid once, before any replay.
cudaGraphExec_t buildGraph(cudaStream_t s, long long cycles) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < KERNELS; ++k) launch(s, cycles);   // recorded as nodes, not run
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));     // CUDA 12 form: graph, flags
    CUDA_CHECK(cudaGraphDestroy(graph));                   // the executable graph keeps its own copy
    return exec;
}
```

`cudaGraphInstantiate` checks the graph, resolves its dependencies and builds an executable graph, a `cudaGraphExec_t`; since CUDA 12 it takes the graph and a flags word (CUDA 11's form took an error node, a log buffer and the buffer's size in place of the flags). The replay is one call per step, the fragment

```cuda
for (int st = 0; st < STEPS; ++st) CUDA_CHECK(cudaGraphLaunch(exec, s));
```

Capturing and instantiating 20 empty kernels took 59.9 µs, once. The stream `s` was created with `cudaStreamNonBlocking`, and that matters: the Guide's "Prohibited and Unhandled Operations" says that while a stream not created with that flag is being captured, "any attempted use of the legacy stream is invalid", and so are synchronous calls such as `cudaMemcpy`, which enqueue to the legacy stream and wait for it. A capturing stream cannot be synchronized or queried either, since its work has not been scheduled.

![Capture code from cudaStreamBeginCapture to cudaGraphInstantiate records kernels 1 to 20 as a chain of nodes, and a pool round trip as three nodes; instantiation takes 59.9 µs once, and each replay costs 0.28 µs of host time per kernel.](diagrams/02-graph-capture.svg)

*Capture turns stream order into edges; instantiation is paid once, and each replay submits the whole graph with one call.*

An instantiated graph is fixed: its nodes, its edges and each kernel's launch configuration. When only parameters change between steps (a pointer, a scalar), the Guide gives two ways to skip a new instantiation: `cudaGraphExecKernelNodeSetParams` and its siblings change one node in place, and `cudaGraphExecUpdate` copies the parameters of a newly captured graph of identical topology into the executable one, so the cheap capture is repeated and the instantiation is not. A change of topology needs a new instantiation.

## 4. When a graph helps: the model fed with measured costs

`snippets/graphs_model.py` turns the sweep into a rule. A kernel of device time $W$ costs $L + W$ launched directly and $\ell + W$ inside a replayed graph, so $N$ kernels cost

$$T_\text{direct} = N (L + W), \qquad T_\text{graph} = T_\text{build} + N (\ell + W).$$

The script holds the run of section 2 as constants, with the date and machine in a comment, fits $L$ and $\ell$ as the mean of the measured time minus $W$ over the six rows, and checks the fit before it derives anything:

```bash
uv run python posts/13-cuda-graphs/snippets/graphs_model.py
```

```text
inputs: graphs.cu on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, driver 545.84,
        Intel Core i7-9750H, Windows 10, 2026-10-18; graph of 20 kernels; build 59.9 us

fit: per-kernel cost above W, direct L = 1.54 us, in a graph l = 0.81 us

  W (us)   measured direct   model   measured graph   model   model speedup
       0              1.80    1.54             0.77    0.81           1.90x
       2              3.52    3.54             2.81    2.81           1.26x
       4              5.54    5.54             4.79    4.81           1.15x
       8              9.33    9.54             8.79    8.81           1.08x
      16             17.43   17.54            16.83   16.81           1.04x
      32             33.59   33.54            32.86   32.81           1.02x

derived: 10000 kernels, direct against one graph built once and replayed
  W (us)   direct (ms)   graph (ms)   speedup
       0         15.35         8.14     1.89x
       2         35.35        28.14     1.26x
       5         65.35        58.14     1.12x
      10        115.35       108.14     1.07x
     100       1015.35      1008.14     1.01x
  speedup as W -> 0                   : 1.89x
  W where the speedup falls to 1.1x   : 6.4 us
  kernels until the graph repays its build: 82 (4.1 replays of 20)
  host issue time per kernel (median) : 1.76 us direct, 0.29 us from a graph (6.1x less)

allocation: 1000 steps of allocate, kernel, free, synchronize (4 MiB)
  strategy                               per step (us)   total (ms)
  cudaMalloc + cudaFree                         519.32        519.3
  pool, release threshold 0 (default)          2004.06       2004.1
  pool, release threshold maximum                45.81         45.8
  kept pool against cudaMalloc        : 11.3x less per step, 473.5 ms saved
  default pool against cudaMalloc     : 3.9x more per step
  cudaFree beside a 20.6 ms kernel     : blocked 20.3 ms, 99% of the kernel
  pool pair beside the same kernel    : blocked 9.2 us
```

With $L = 1.54$ µs and $\ell = 0.81$ µs the model matches every spinning kernel within 0.21 µs. It misses the empty kernel, which costs 0.26 µs more than the fit when launched directly, so the model's limit of 1.89 times as $W \to 0$ is below the 2.34 times measured for empty kernels. Everything under "derived" is computed from the fitted inputs, not measured.

The speedup is $(L + W) / (\ell + T_\text{build}/N + W)$: the graph saves a fixed $L - \ell = 0.73$ µs per kernel, which matters only while $W$ is of that size. The speedup falls to 1.1 times at $W = (L - 1.1(\ell + T_\text{build}/N)) / 0.1 = 6.4$ µs, and by 100 µs it is 1.01 times: for a few long kernels a graph buys nothing measurable. The break-even against the build is short, $T_\text{build} / (L - \ell) = 59.9 / 0.73 \approx 82$ kernels, about four replays of the 20-kernel graph. A graph built once for a loop of thousands of steps repays itself in the first few; a graph rebuilt every step never does.

![Speedup of graph replay against kernel length from 0 to 32 µs: measured 2.34, 1.25, 1.15, 1.06, 1.04 and 1.02 at 0, 2, 4, 8, 16 and 32 µs, with the model's curve, which crosses 1.1 at 6.4 µs.](diagrams/03-launch-sweep.svg)

*A graph saves a fixed 0.73 µs per kernel, so its speedup is large only for kernels a few microseconds long.*

The host gains separately: a replay costs the CPU 0.29 µs per kernel instead of 1.76 µs, six times less, so even when the GPU is the bottleneck the thread that feeds it is free sooner, for the next step's host work or another stream.

## 5. The other stall: `cudaMalloc` and `cudaFree`

Allocation is the second cost, and it can make the host wait for work it has nothing to do with. The Guide's section "Implicit Synchronization" lists a device memory allocation among the operations that, issued between two commands of different streams, keep them from running concurrently, and Kini and Hemstad's "Using the NVIDIA CUDA Stream-Ordered Memory Allocator, Part 2" (NVIDIA Technical Blog, 27 July 2021) states the contrast: "Unlike cudaFree, cudaFreeAsync does not implicitly synchronize the device."

The harness measures it. It times a pair of each kind with the GPU idle, then starts a 20 ms spin kernel on a second stream, `other`, and times one call on the host, once per kind. The same run as section 2:

```text
allocation, 4 MiB, GPU idle; median of 7 passes of 200 pairs; us per pair
  cudaMalloc + cudaFree           :   408.38
  cudaMallocAsync + cudaFreeAsync :     9.30
host time a call blocks while a 20.6 ms kernel runs on another stream; median of 7; us
  cudaMalloc                     :    145.2
  cudaFree                       :  20343.4
  poolRoundTrip, pool trimmed    :     15.4
  poolRoundTrip, pool kept       :      9.2
```

With the GPU idle a `cudaMalloc` and `cudaFree` of 4 MiB cost 408.38 µs per pair, 44 times a pair from the pool. Beside a kernel on another stream, `cudaFree` held the host for 20.3 ms, 99% of the 20.6 ms kernel: it waited for the whole device. `cudaMalloc` returned in 145.2 µs without waiting, on this driver, but by the Guide's rule it still keeps work issued around it on different streams from overlapping. A `cudaFree` inside a step loop is thus a device-wide barrier nobody asked for: every stream of [post 12](../12-cuda-streams/index.md) drains before the host goes on.

![Beside a 20.6 ms kernel on another stream, cudaMalloc returns after 0.15 ms, cudaFree after 20.3 ms and the pool round trip after 9.2 µs; one allocate, kernel, free and synchronize step costs 519 µs with cudaMalloc, 2,004 µs from a pool at threshold 0 and 45.8 µs at the maximum threshold.](diagrams/04-allocation-stall.svg)

*`cudaFree` waits for every stream; the pool waits for none, and a pool that keeps its memory makes each step cheap.*

## 6. Stream-ordered allocation and the pool's release threshold

`cudaMallocAsync` and `cudaFreeAsync` take a stream and are ordered in it like a kernel, drawing memory from a **memory pool** the driver keeps per device. The Guide's section "API Fundamentals (cudaMallocAsync and cudaFreeAsync)" says the pointer "is determined synchronously and is available for constructing future work", while the memory becomes usable, and stops being so, in stream order. The harness's round trip:

```cuda
// One stream-ordered allocation, used and freed on stream s: no device-wide synchronization.
void poolRoundTrip(cudaStream_t s, size_t bytes) {
    void* d = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d, bytes, s));   // from the pool, ordered in s
    empty<<<1, 32, 0, s>>>();                     // the work that uses d
    CUDA_CHECK(cudaFreeAsync(d, s));             // back to the pool once s reaches this point
}
```

Beside the 20.6 ms kernel it returned in 9.2 to 15.4 µs. Whether the pool still holds memory at the next step depends on its **release threshold**. The Guide: "When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or device synchronize"; Kini and Hemstad's Part 1: "By default, the release threshold of a pool is zero". A loop that synchronizes every step therefore hands the pool's memory back each step and maps it again at the next allocation. The harness times one step of allocate, empty kernel, free and `cudaStreamSynchronize` three ways, with the time inside each call:

```text
step = allocate, kernel, free, cudaStreamSynchronize; median of 7 passes of 200 steps; us per step
                                          step   allocate       free   synchronize
  cudaMalloc + cudaFree                  519.32     112.33     399.42          4.35
  pool, release threshold 0 (default)  2004.06     158.90       2.76       1818.46
  pool, release threshold maximum        45.81       1.66       1.14         39.19
capture: cudaMalloc inside a global-mode capture returns "operation not permitted when stream is capturing"
capture: poolRoundTrip becomes a graph of 3 nodes (allocate, kernel, free), replayed ok
SM clock: 1743 MHz before the timed rows, 1529 MHz after
```

At the default threshold the pool was the slowest of the three, 2,004 µs per step against 519 µs for `cudaMalloc`, 3.9 times as much: 1,818 µs inside the synchronize, where it returned its 4 MiB to the operating system, and 159 µs in the next allocation, which took it back. At the maximum threshold the pool kept the block, allocation and free cost 1.66 and 1.14 µs, and the step fell to 45.81 µs, 11.3 times cheaper than `cudaMalloc`; the 39 µs left are the synchronize itself. The setting is two lines, a fragment of the harness, where `pool` comes from `cudaDeviceGetDefaultMemPool`:

```cuda
uint64_t keepNone = 0, keepAll = UINT64_MAX;
CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepAll));
```

Kini and Hemstad's Part 2 advises the same for a pool one process uses: "Use the maximum release threshold." Such a pool never shrinks by itself; `cudaMemPoolTrimTo(pool, minBytesToKeep)` returns memory at a moment the program chooses, between phases that need different amounts.

The capture lines of the run are why pools and graphs belong together. Inside a capture `cudaMalloc` is refused and the capture invalidated; the pool's calls are stream operations, so the round trip is recorded as three nodes and replays. When a buffer's size never changes, simpler still is to allocate it once before the loop, which a kept pool approximates when sizes vary.

## 7. Putting it together

The tools of this part compose for a loop that repeats. Streams ([post 12](../12-cuda-streams/index.md)) overlap copies and kernels. Buffers come from allocations made once, or from a pool that keeps its memory, so no `cudaFree` drains the device mid-step. One step is captured into a graph, whose replay costs the host one call per step and the GPU about 0.8 µs per kernel instead of 1.5. The gain is largest for kernels a few microseconds long; with 100 µs kernels the derived table gives 1.01 times.

A graph rewards repetition. Data-dependent control flow, different kernels each step, or a launch configuration that changes with the input means a new capture and instantiation, 59.9 µs here, repaid only after about 82 kernels. When short kernels can become fewer, merging them removes the launches and the trips through memory at once: [post 14](../14-kernel-fusion/index.md).

## 8. Make it run

The model needs only Python; the harness needs an NVIDIA GPU with CUDA 12 or later. From the series root:

```bash
uv run python posts/13-cuda-graphs/snippets/graphs_model.py
nvcc -O3 -arch=sm_75 posts/13-cuda-graphs/snippets/graphs.cu -o graphs && ./graphs
```

On Windows the program is `graphs.exe`, and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both and the `-arch` flag for other cards. The model runs in well under a second; the harness builds in about 5 seconds and runs in about 10. A reader who runs `graphs.cu` elsewhere can paste its sweep, build cost and step costs into the constants at the top of `graphs_model.py` to derive that machine's break-even.

## 9. What can go wrong?

Each of these happened on the author's laptop while this post was measured.

- **Timing while something else uses the GPU or the power source changes.** A run of an earlier version of the harness, started while another program kept the GPU at 100%, printed an SM clock of 225 MHz and 21.75 µs per empty kernel instead of 1.80. The laptop's charger connection flapped all afternoon, which drops the GPU clock; the printed clock lines are how such a run is caught and discarded.
- **Reading the issue time as the cost.** The loop of direct launches returned after 1.75 µs per kernel, but the stream needed 3.52 µs per 2 µs kernel; only the time until the stream finishes counts.
- **A kernel set in cycles that runs longer.** The 20 ms busy kernel is set in cycles at the measured clock; in a run of an earlier version the clock fell during it and it took 72.0 ms, which is why the harness prints its real duration.
- **Trusting the pool's default.** At release threshold 0 with a synchronize per step, the pool cost 3.9 times as much per step as `cudaMalloc`.

---

## Common pitfalls

1. **Using a graph to speed up long kernels.** It saves about 0.73 µs per kernel here, 1.02 times at 32 µs kernels; measure the kernels' length first.
2. **A `cudaFree` inside the loop.** It waits for every stream, 20.3 ms here behind another stream's kernel, and `cudaMalloc` cannot be captured; allocate once or use the pool.
3. **Leaving the release threshold at 0 in a loop that synchronizes.** Each synchronize returns the pool's memory: 2,004 µs per step against 45.8 µs at the maximum.
4. **Re-instantiating every step.** The build cost 59.9 µs, the saving of about 82 kernels; change parameters with `cudaGraphExecKernelNodeSetParams` or `cudaGraphExecUpdate`.
5. **Capturing a stream that touches the legacy default stream.** During a capture of a blocking stream, legacy-stream work and synchronous calls such as `cudaMemcpy` are invalid; capture on a stream created with `cudaStreamNonBlocking`.

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "CUDA Graphs", consulted October 2026 (CUDA 12.3 edition). Capture, instantiation, prohibited operations, and graph update.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Streams", consulted October 2026 (CUDA 12.3 edition). Implicit synchronization.
- Gray, A. "Getting Started with CUDA Graphs." NVIDIA Technical Blog, 5 September 2019. Capture and replay of 20 short kernels on a V100.
- Kini, V. and Hemstad, J. "Using the NVIDIA CUDA Stream-Ordered Memory Allocator, Part 1" and "Part 2." NVIDIA Technical Blog, 27 July 2021. Pools, the release threshold, and why `cudaFree` stalls the device.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Stream Ordered Memory Allocator", consulted October 2026 (CUDA 12.3 edition). Pools, the release threshold, and trimming.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 14 - Kernel fusion: the fastest kernel is the one never launched](../14-kernel-fusion/index.md):** the other answer to many short kernels, fewer of them, which saves the launches and the memory traffic between them.
- **[Post 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy](../10-roofline/index.md):** the steady-state model whose blind spot, the fixed cost of a launch, this post measures.
