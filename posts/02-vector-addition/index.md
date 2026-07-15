# 02 · Your first CUDA kernel — vector addition and the memory wall

> **TL;DR.** A graphics processing unit (GPU) does not run your code faster by being a quicker central processing unit (CPU); it runs the *same* line across tens of thousands of elements in parallel. The "hello world" that teaches this is adding two arrays, `C[i] = A[i] + B[i]`: on a CPU you write a loop over `i`, on a GPU you delete the loop and launch a **grid** of threads where each thread computes its own index with `blockIdx.x * blockDim.x + threadIdx.x` and handles one element. Because the kernel moves twelve bytes (two 4-byte reads and one 4-byte write) for every single addition, it is fundamentally **memory bound**, so "fast" never means more math, it means feeding the cores at a high fraction of peak memory bandwidth. This post builds the kernel one thread at a time, benchmarks it honestly against a CPU, and shows exactly when the GPU wins and when it loses.
>
> **After reading this you will be able to:**
> - Write a CUDA vector-addition kernel and compute each thread's global index by hand.
> - Explain what a grid, a block, and a thread are, and why a block size should be a multiple of 32.
> - Move data across the host/device boundary with `cudaMalloc`, `cudaMemcpy`, and `cudaFree`, and guard the surplus threads with `if (i < N)`.
> - Predict from arithmetic intensity that vector addition is memory bound, and say what "fast" means for it.

![On the CPU a single worker steps through the array indices in sequence; on the GPU a grid of threads each owns one index and all of them write in parallel, so the grid replaces the loop.](diagrams/01-grid-is-the-loop.svg)
*On a CPU the loop walks the array; on a GPU the grid of threads is the loop, and each thread owns one index.*

---

## 1. The motivation: deleting the loop

On a CPU, adding two arrays is a `for` loop you have written a hundred times:

```python
for i in range(N):
    C[i] = A[i] + B[i]
```

The CPU walks the indices one after another. Even with a few cores it processes a handful of elements at a time, in sequence. That is the right shape for code with branches and dependencies, but vector addition has neither: element 7 does not need element 6, and the operation on every element is identical. It is *embarrassingly parallel*, which is exactly the workload a GPU exists for.

> **What is a kernel?** A **kernel** is a function that runs on the GPU, written from the point of view of a *single* element. You do not write the loop; you write what one worker does, then launch a whole **grid** of those workers, one per element. The hero image above is one launch: a grid of workers, each owning one slot of the output and writing it in parallel.

So we stop thinking about *iterating* and start thinking about *covering*. Instead of one worker stepping through `N` indices, picture `N` workers, each assigned one index, all firing in parallel. The loop disappears, and the rest of this series is learning to think this way: not "loop over the data" but "what does one thread do, and how do I map threads onto data."

## 2. The mechanism: grids, blocks, threads, and the global index

When you launch a kernel, the GPU does not hand you a flat list of threads. It organizes them into a two-level hierarchy, and you must understand it because the index arithmetic depends on it.

> **The three words, defined once.** A **thread** runs one copy of your kernel for one element. A **block** is a group of threads (up to 1024) that run together on one streaming multiprocessor and can share fast on-chip memory. A **grid** is the full set of blocks from one launch. So a launch is one grid, made of blocks, each made of threads, and you choose how many of each.

You pick those two numbers with the triple-angle-bracket launch syntax, the one piece of CUDA that is not ordinary C++:

```cpp
// <<< number of blocks , threads per block >>>
vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);
```

Inside the kernel, four read-only built-in variables tell each thread where it sits. Two locate the thread and vary with its position (`threadIdx.x` within its block, `blockIdx.x` among the blocks); the other two describe the launch and are the same for every thread (`blockDim.x`, `gridDim.x`):

| Built-in | Meaning | Example |
|---|---|---|
| `threadIdx.x` | this thread's index inside its block | `1` |
| `blockIdx.x` | this block's index inside the grid | `1` |
| `blockDim.x` | threads per block (same for all) | `4` |
| `gridDim.x` | blocks in the grid (same for all) | `3` |

`threadIdx.x` restarts at zero inside every block, while `blockIdx.x` identifies which block this is within the grid; to get a single index into the array you flatten the two-level position into one number:

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

Read it as "skip past all the earlier blocks, then step in by my position inside my own block." A thread in block 1 with `blockDim.x = 4` and `threadIdx.x = 1` computes $1 \times 4 + 1 = 5$, so it owns index 5. Every thread lands on a distinct index, and together they tile the array. That single line is the whole kernel:

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        C[i] = A[i] + B[i];                          // the work: one add
}
```

Why the `if`? You launch *whole blocks*, so the grid is almost always slightly larger than `N`. You size it with integer ceiling division:

```cpp
int blockSize = 256;
int blocksPerGrid = (N + blockSize - 1) / blockSize;   // round up, never down
```

For `N = 1000` and `blockSize = 256` that is `(1000 + 255) / 256 = 4` blocks, which is 1024 threads covering 1000 elements. The 24 extra threads have no element to write; `if (i < N)` is false for them, so they touch no memory instead of scribbling past the end of the array.

![A worked example with N equal to 6 and a block size of 4: ceiling division gives 2 blocks and 8 threads, threads 0 to 5 each write one cell, and the two surplus threads 6 and 7 fail the i less than N test and return without writing.](diagrams/03-launch-config.svg)
*Six elements need two blocks of four, which is eight threads. The two surplus threads are caught by `if (i < N)`.*

> **Why a block size that is a multiple of 32?** The hardware executes threads in **warps** of 32 lanes in lockstep. A block of 100 threads still occupies 128 lanes (four warps) and wastes 28. Start with 128 or 256 and benchmark from there; a block can hold up to 1024 threads, but the best size also depends on the kernel's register and shared-memory use.

## 3. Two memory spaces: the copy, launch, copy dance

The CPU and GPU have **separate physical memory** (on a discrete GPU like this one). A pointer into system RAM is meaningless on the device, so you explicitly allocate device memory and copy data across the PCIe bus before and after the kernel.

![The host holds the CPU arrays and the device holds the GPU arrays, joined by the PCIe bus; six numbered steps allocate device memory, copy the inputs over, launch the kernel, synchronize, copy the result back, and free.](diagrams/02-host-device-workflow.svg)
*Allocate on the device, copy inputs over, launch, wait, copy the result back, free. On a discrete GPU, the PCIe bus is the only road between the two memories.*

```cpp
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, bytes);  cudaMalloc(&d_B, bytes);  cudaMalloc(&d_C, bytes);  // 1

cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);                          // 2
cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);                    // 3
cudaDeviceSynchronize();                                                     // 4

cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);                          // 5
cudaFree(d_A);  cudaFree(d_B);  cudaFree(d_C);                                // 6
```

> **A kernel launch is asynchronous.** The CPU *queues* the kernel and immediately runs on to the next line; it does not wait. The result copy on step 5 is still safe without the explicit sync, because a blocking `cudaMemcpy` runs in the same stream as the kernel and waits for it to finish before copying — you never read half-written data. What the asynchrony *does* break is **timing** and **error reporting**: a host clock wrapped around the launch alone measures a kernel that has not run yet (microseconds instead of milliseconds), and a fault inside the kernel surfaces later, blamed on some unrelated call. That is why `cudaDeviceSynchronize()` earns its place here — it gives a host timer a real finish line and a defined point to catch runtime faults. Pair it with `cudaGetLastError()` right after the launch, which catches a bad launch configuration immediately.

In the committed program every call is wrapped so a failure points at its own line. Around the launch that is just two checks:

```cpp
vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);
CUDA_CHECK(cudaGetLastError());        // catch a bad launch configuration now
CUDA_CHECK(cudaDeviceSynchronize());   // wait, then surface any runtime fault
```

## 4. Running it — with or without a GPU

The complete program, with a `CUDA_CHECK` macro wrapping every call, is committed at [snippets/vector_add.cu](snippets/vector_add.cu). Build and run it with:

```bash
cd posts/02-vector-addition
nvcc -O2 snippets/vector_add.cu -o vector_add   # needs a GPU + CUDA toolkit
./vector_add                                     # needs a GPU
python snippets/vector_add.py                    # no GPU: the CPU model, needs only NumPy
```

The committed companion is a faithful CPU model in NumPy that reproduces the exact logic with no GPU required: it simulates a launch of `3907 x 256` threads over `N = 1,000,003` elements (deliberately not a multiple of the block size), runs each through the global-index mapping and the `i < N` guard, and checks that every element of `C` is written exactly once and equals `A + B`. It prints the arithmetic intensity rather than asserting an invented bandwidth. The model lives at [snippets/vector_add.py](snippets/vector_add.py) and needs only NumPy:

```text
vector_add CPU model
  N                 = 1,000,003 (not a multiple of 256)
  launch            = 3,907 blocks x 256 threads = 1,000,192 threads
  surplus threads   = 189 (guarded out by i < N)
  every i written 1x= True
  matches A + B     = True
  bytes per element = 12 (read A, read B, write C)
  arithmetic int.   = 0.0833 FLOP/byte  -> memory bound
```

The 189 surplus threads are the whole point of the boundary check, made concrete.

## 5. Benchmarking, and when the GPU actually wins

Most "GPU versus CPU" numbers are wrong for at least one reason: they time a single run, measure a GPU kernel with a CPU timer (missing the async launch), let the compiler delete unused CPU work, or quietly mix pinned and pageable host memory so the transfer numbers are not comparable. The honest methodology reports **two** GPU numbers: kernel-only (data already on the device, the best case) and end-to-end (host to device, kernel, device to host, the realistic case for one-off work).

| Aspect | Approach |
|---|---|
| Repetitions | many runs, averaged, warm-up discarded |
| GPU timing | CUDA events (`cudaEventElapsedTime`) |
| CPU timing | `std::chrono::steady_clock` |
| Memory | pinned host memory (`cudaMallocHost`): the best-case, reproducible transfer path (faster than pageable, but scarcer and slower to allocate) |
| Verification | checksums so the compiler cannot delete the work |

The full harness that produced the table below — CUDA events, pinned memory, warm-up, checksums, and the OpenMP CPU baseline — is at [benchmark_vector_add.cu](../../examples/src/Vector%20Addition/benchmark_vector_add.cu). Build and run it with:

```bash
# Linux
nvcc -O3 -std=c++17 -Xcompiler -fopenmp "examples/src/Vector Addition/benchmark_vector_add.cu" -o bench && ./bench
# Windows
nvcc -O2 -Xcompiler "/openmp" "examples/src/Vector Addition/benchmark_vector_add.cu" -o benchmark_vector_add.exe
```

These numbers were measured on one laptop; treat the shape, not the digits, as the lesson.

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 (14 SMs, compute capability 7.5, ~192 GB/s) |
| CPU | Intel Core i7-9750H (6 cores / 12 threads), 16 GB DDR4 (~40 GB/s) |
| Toolchain | CUDA 12.6, driver 560.94, `nvcc -O2`, OpenMP 12 threads, PCIe Gen3 x16 |

| N | CPU (1 thread) | CPU (OpenMP 12T) | GPU (with transfer) | GPU (kernel only) |
|---|---|---|---|---|
| 1K | **<1 µs** | 2 µs | 149 µs | 3 µs |
| 10K | 3 µs | **3 µs** | 145 µs | 3 µs |
| 100K | 33 µs | **9 µs** | 139 µs | 11 µs |
| 1M | 1.02 ms | **0.39 ms** | 1.20 ms | 0.08 ms |
| 10M | 13.6 ms | 12.3 ms | 11.5 ms | **0.82 ms** |
| 100M | 146 ms | 128 ms | 123 ms | **8.28 ms** |

![A log-log plot of runtime against array size for the three measured curves; the GPU with transfer is a flat 140 microsecond floor for small N and only overtakes the CPU near one to ten million elements, while the kernel-only curve is fastest once data is resident.](diagrams/04-memory-wall.svg)
*The GPU carries a fixed launch-and-transfer cost that dominates small arrays. It pays off only once `N` is large enough to amortize it. (The plotted CPU line is single-threaded; the OpenMP figures are in the table above.)*

Reading the curves:

- **Small `N` (under 100K): the CPU wins.** The kernel takes microseconds, but you pay roughly 140 µs just to launch and transfer. Do not GPU-accelerate tiny workloads.
- **Medium `N` (around 1M): it is close.** Multi-threaded OpenMP is competitive because the data fits in cache and the CPU boosts aggressively.
- **Large `N` (10M and up): the GPU pulls away.** At 100M elements the kernel alone is about **17× faster** than the single-threaded CPU, and even with transfers the GPU stays ahead.

> **The key insight: this kernel is memory-bandwidth bound, not compute bound.** Each element costs two reads and one write (12 bytes) for a single addition, an arithmetic intensity of about `1 / 12 = 0.083` FLOP/byte. A modern CPU could do the additions in microseconds, but moving 100M elements (1.2 GB) at ~40 GB/s takes tens of milliseconds, and that wait is the runtime. The GPU wins not because it adds faster but because its DRAM delivers ~192 GB/s, roughly five times more. And the kernel genuinely reaches that ceiling: at 100M elements it moves 1.2 GB in 8.28 ms, about **145 GB/s — roughly 76% of the ~192 GB/s peak**. Both machines are waiting on memory; the GPU just waits less.

## 6. When it bites: one-off work versus a pipeline

The decision is not "GPU good, CPU bad," it is whether the work amortizes the transfer:

| Scenario | Best choice | Why |
|---|---|---|
| `N` < 100K, one-off | CPU (single thread) | launch and transfer overhead dominate |
| 100K–10M, one-off | CPU (OpenMP) | competitive and simpler |
| `N` ≥ 10M, one-off | GPU | bandwidth wins despite the transfer |
| Multi-kernel pipeline | GPU | pay the transfer once, run many kernels |
| Data already resident | GPU | usually; no transfer to amortize (though a tiny launch still has fixed overhead) |

The last two rows are why GPUs dominate deep learning. A vector addition over 10M elements moves 30M floats — 120 MB, both inputs and the output — across the bus for about 11 ms once, but a training step then runs hundreds of kernels on data that never leaves the device. You pay the bus tax a single time, then run at kernel-only speed throughout the resident pipeline. Vector addition in isolation is the worst case for the GPU; it is still the right first kernel, because every habit it teaches (one thread per element, the boundary guard, the bandwidth lens) carries straight into the kernels that do pay off.

---

## Common pitfalls

- **Dropping the bounds guard.** Because you launch whole blocks, the grid almost always exceeds `N`. Without `if (i < N)`, the surplus threads read and write past the array: an illegal-access crash or silent corruption. The boundary check is not optional.
- **Assuming a launch reports errors synchronously.** A launch returns before the GPU runs. You need `cudaGetLastError()` for launch-config errors *and* `cudaDeviceSynchronize()` for runtime faults; check only one and failures slip past silently.
- **Timing the first launch.** The first kernel pays one-time context and module initialization (plus just-in-time compilation if the binary carries no native code for your GPU). Run an untimed warm-up, then measure with CUDA events on the GPU timeline, never a host wall-clock that ignores asynchrony.
- **A block size that is not a multiple of 32.** Threads execute in 32-wide warps, so 100 threads round up to 128 lanes and waste 28. Use a multiple of 32 — 128 or 256 to start.
- **Folding `cudaMemcpy` into the timed region (or excluding it when it matters).** Report kernel-only and end-to-end separately; for one-shot work the copy often dominates and hiding it flatters the GPU.
- **Chasing math tricks on a memory-bound kernel.** Cleverer arithmetic does almost nothing here. The main lever is coalesced access (consecutive threads read consecutive addresses, so the hardware fuses their loads into the fewest possible memory transactions) that saturates DRAM bandwidth; across a whole pipeline you cut traffic further by fusing kernels and keeping data resident. Profile achieved bandwidth as a percent of peak, not arithmetic throughput.

---

## Further reading

- Harris, M., *"An Even Easier Introduction to CUDA"* (NVIDIA, 2017). The canonical first kernel in a few pages (beginner-friendly).
- Harris, M., *"CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"* (NVIDIA). The grid-stride pattern that generalizes the surplus-thread guard to grids smaller than the data (technical).
- Harris, M., *"How to Implement Performance Metrics in CUDA C/C++"* (NVIDIA, 2012). Timing kernels correctly with CUDA events (technical).
- NVIDIA, *"CUDA C++ Programming Guide"* (current). The reference for the execution model, built-in variables, and launch syntax (technical, reference).
- Williams, S., Waterman, A., & Patterson, D., *"Roofline: An Insightful Visual Performance Model"* (2009). The origin of the memory-bound versus compute-bound picture used in Section 5 (technical, historical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 01, Introduction to CUDA](../01-introduction-to-cuda/index.md)**: if grids, warps, and the host/device model were new here, this is the hardware tour they come from.
- **[Post 03, Matrix multiplication](../03-matrix-multiplication/index.md)**: the first *compute-bound* kernel, where tiling and shared memory finally matter and the lessons here invert.
- **[Post 04, Reduction](../04-reduction/index.md)**: what changes when threads must cooperate instead of each owning one independent element.
