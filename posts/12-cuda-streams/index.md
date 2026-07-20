# 12 · CUDA streams — overlapping copy and compute for free throughput

> **TL;DR.** Done serially on the default stream, only one of the GPU's engines works at a time, so a copy leaves the cores idle and a kernel leaves the bus idle: about 33% utilization. But a GPU has **independent hardware engines** (a host-to-device copy engine, the compute cores, a device-to-host copy engine) and PCIe is **full-duplex**, so all three can run at once. **Streams** are ordered queues of work whose operations overlap across *different* streams, so splitting the data into chunks and giving each chunk its own stream **pipelines** the copies and compute into near-full utilization. The one catch is that async transfers require **pinned** host memory.
>
> **After reading this you will be able to:**
> - Explain why default-stream execution wastes two-thirds of the GPU's engines.
> - Pipeline a workload across streams so copy and compute overlap.
> - Say why `cudaMemcpyAsync` needs pinned memory and what happens without it.
> - Predict where streams help most (bandwidth-bound work) and least (compute-bound).

![A serial timeline of 12 copy/compute/copy blocks, versus a pipelined timeline where three engines run staggered chunks with a green steady-state band.](diagrams/01-serial-vs-pipeline.svg)
*Serial runs one engine at a time; streams overlap the three engines into a pipeline.*

---

## 1. The motivation: you paid for the whole chip

The default pattern is a trap. `cudaMemcpy` → kernel → `cudaMemcpy`, all on the default stream, runs strictly in order. While the copy engine moves data, the compute cores are idle; while the kernel runs, the bus is idle. If transfer and compute take similar time, you are using each engine only about a third of the time. The hardware to do better is already there.

## 2. What a stream is, and why it can overlap

A **stream** is a queue of GPU operations that execute *in order*. The key fact is that operations in *different* streams have no ordering between them, so the hardware is free to run them at once, on its **separate engines**:

![A GPU chip with three green engines (compute SMs, H2D copy, D2H copy) over a full-duplex PCIe bus to the CPU.](diagrams/02-hardware-engines.svg)
*The copy engines and the compute cores are independent, and PCIe carries H2D and D2H traffic at the same time.*

Most GPUs have one or two copy engines (two enables a simultaneous host-to-device and device-to-host copy) alongside compute engines that can even run kernels from several streams concurrently. Because PCIe is full-duplex, a copy in and a copy out can travel simultaneously. The default stream throws all of this away by serializing everything; explicit streams unlock it.

## 3. The requirement: pinned memory

There is one prerequisite. `cudaMemcpyAsync` is only truly asynchronous from **pinned** (page-locked) host memory. Ordinary `malloc` memory is *pageable*: the OS can move it, so before a DMA the driver must first copy it into a hidden pinned staging buffer, a blocking step that silently turns the async copy synchronous.

![Pageable memory needs a two-step staging copy (red) that blocks; pinned memory DMAs directly to the GPU (green).](diagrams/03-pinned-memory.svg)
*From pinned memory the GPU DMAs directly, so the copy overlaps; from pageable memory it cannot.*

```cuda
float* h;  cudaMallocHost(&h, bytes);   // page-locked: async works
// (malloc here would make every cudaMemcpyAsync secretly synchronous)
```

Pinned memory is a limited resource and slow to allocate, so allocate it **once at startup**, never inside the timed loop.

## 4. The pipeline

With pinned buffers and several streams, the recipe is to chunk the data and issue each chunk's copy-in, kernel, and copy-out on its own stream:

```cuda
for (int i = 0; i < NS; ++i) {
    int off = i * chunk;
    cudaMemcpyAsync(&d_a[off], &h_a[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
    cudaMemcpyAsync(&d_b[off], &h_b[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
    vectorAdd<<<grid, block, 0, s[i]>>>(&d_a[off], &d_b[off], &d_c[off], chunk);
    cudaMemcpyAsync(&h_c[off], &d_c[off], chunkBytes, cudaMemcpyDeviceToHost, s[i]);
}
```

Each stream is internally ordered (its kernel waits for its copy), but the streams overlap: while stream 1 computes, stream 2 copies in and stream 0 copies out. After issuing all the streams, wait on `cudaDeviceSynchronize()` before reading `h_c`, since the async copies have not necessarily finished when the loop returns.

Build and run the standalone demo:

```bash
nvcc -O3 -arch=sm_75 snippets/streams.cu -o streams && ./streams
```

CUDA can't run in this post's environment, so a CPU model (`snippets/streams_model.py`) computes the *makespan* (the total wall-clock time from the first copy to the last result) and shows it fall from `3N` to `N + 2` stage-times; run `python snippets/streams_model.py`:

```text
streams model (3 equal stages: H2D, compute, D2H)
  chunks (streams)   = 4
  serial makespan    = 12 stage-times  (every stage end to end)
  pipelined makespan = 6 stage-times  (= N + 2 once filled)
  speedup            = 2.00x
  engine utilization = H2D 67%, compute 67%, D2H 67%
  with 64 chunks     = 2.91x speedup, compute 97% utilized (-> 3x, 100%)
```

More chunks deepen the pipeline: the fixed fill-and-drain cost amortizes, the speedup approaches `3×`, and utilization approaches 100%.

## 5. The numbers, and when streams help

On an RTX 3080 (compute capability 8.6) over 256 MB, the two changes (pinned memory, then overlap) compound:

| Method | Time | Speedup |
|---|---|---|
| serial (pageable) | 45.2 ms | 1.0× |
| serial (pinned) | 32.1 ms | 1.4× |
| 4 streams (async) | 12.5 ms | 3.6× |

These times were measured separately on an RTX 3080; the committed `snippets/streams.cu` is a smaller standalone demo (`N = 1<<24`, about 67 MB per array, built for `sm_75`) that reports its own streamed time rather than reproducing this whole table.

![A time bar chart: serial pageable 45.2, serial pinned 32.1, four streams 12.5 in green.](diagrams/04-streams-perf.svg)
*Pinned memory speeds the transfer; streams hide the compute behind it.*

Streams help most when **transfer time is comparable to compute time**, which is exactly the bandwidth-bound case (vector add here): the compute hides almost entirely behind the copies. A heavily compute-bound kernel already keeps the cores busy, so its transfers are relatively cheap and streams add little. The sweet spot is **4–8 streams**; far more just adds scheduling overhead, because there are only one or two copy engines.

---

## Common pitfalls

- **Touching the default stream mid-pipeline.** Any operation on the default stream (stream 0) waits for all other streams and blocks them until it finishes. One stray `cudaMemcpy` (no `Async`, no stream) serializes the whole thing. Use explicit streams everywhere.
- **Forgetting pinned memory.** `cudaMemcpyAsync` from `malloc` memory silently runs synchronously, and the overlap vanishes with no error. If Nsight shows no overlap, check the allocation first.
- **Creating too many streams.** Past ~8 there is no more hardware parallelism (one or two copy engines), only overhead.
- **Ignoring cross-stream dependencies.** If stream B reads what stream A wrote, they are not automatically ordered; record a `cudaEvent` in A and `cudaStreamWaitEvent` in B.
- **Allocating pinned memory in the loop.** `cudaMallocHost` is slow; one allocation inside the timed region erases the speedup. Allocate once, reuse.

---

## Further reading

- Harris, M., *"How to Overlap Data Transfers in CUDA C/C++"* (NVIDIA, 2012). The canonical streamed-pipeline walkthrough this post follows (technical, foundational).
- NVIDIA, *"CUDA C++ Programming Guide — Streams"* (current). The reference for stream semantics, the default stream, and events (reference).
- NVIDIA, *"CUDA C++ Best Practices Guide — Asynchronous Transfers"* (current). Pinned memory and overlap guidance (technical).
- NVIDIA, *"Nsight Systems User Guide"* (current). How to confirm overlap on the real timeline (reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 13, CUDA Graphs](../13-cuda-graphs/index.md)**: once copy and compute overlap, the next cost is the CPU launching each piece; graphs record a whole pipeline once and replay it with a single call.
- **[Post 08, Parallel scan](../08-parallel-scan/index.md)**: the last of the single-kernel algorithmic patterns, worth a revisit before the systems posts take over.
