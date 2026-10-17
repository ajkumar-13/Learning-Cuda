# 02 - Your first CUDA kernel: vector addition and the memory wall

> **TL;DR.** A graphics processing unit (GPU) does not run a loop faster than a central processing unit (CPU) does; it deletes the loop and runs the loop's body once per element, in thousands of threads that each find their element with `blockIdx.x * blockDim.x + threadIdx.x`. Adding two arrays, `C[i] = A[i] + B[i]`, is the first kernel because it shows the whole host program around a launch: allocate device memory, copy the inputs in, launch, wait, copy the result out, free. The kernel moves 12 bytes for every addition, an arithmetic intensity of $1/12$ floating-point operation (FLOP) per byte, so it is **memory bound**: on the author's GTX 1650 it reaches 145 GB/s, 91% of the card's theoretical peak, and the copies across the bus, not the arithmetic, decide whether the GPU wins at all.
>
> **Prerequisites:** [post 01](../01-introduction-to-cuda/index.md) (the thread, block, and grid hierarchy, the global index, warps of 32).
> **Safe to skip?** Skip it if the reader can write a guarded kernel and the host code that allocates, copies, launches, checks, and frees, can time a kernel with CUDA events, and can say from its bytes per FLOP why vector addition is memory bound.
>
> **After reading, you will be able to:**
>
> - Write the vector-addition kernel from one thread's point of view and compute any thread's global index in a launch.
> - Size a launch with a ceiling division, pick a block size that is a multiple of 32, and say what the guard `if (i < N)` does for the surplus threads.
> - Move data across the host and device boundary with `cudaMalloc`, `cudaMemcpy`, and `cudaFree`, and check every call and the launch with `CUDA_CHECK`.
> - Benchmark a kernel honestly with CUDA events, a warm-up, pinned memory, and a check of every element, timing the kernel apart from the copies.
> - Predict from arithmetic intensity that vector addition is memory bound, and judge the kernel by its achieved bandwidth as a fraction of the card's peak.

![Two panels over the same six-element array C. On the left, one CPU thread walks the loop: six rows of time steps, and in each row one cell is filled, element 0 in step 1 up to element 5 in step 6. On the right, six GPU threads each own one index and all six cells are filled in step 1, leaving steps 2 to 6 empty.](diagrams/01-grid-is-the-loop.svg)

*On a CPU the loop walks the array one index per step; on a GPU the grid of threads is the loop, and each thread owns one index.*

---

## 1. The question: what happens to the loop?

On a CPU, adding two arrays is a loop:

```cpp
for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
```

The CPU walks the indices one after another; even with a few cores it handles a handful of elements at a time. That is the right shape for code with branches and dependencies, and vector addition has neither: element 7 does not need element 6, and every element gets the same operation. It is *embarrassingly parallel*, the workload a GPU is built for.

> **What is a kernel?** A **kernel** is a function that runs on the GPU, written from the point of view of *one* element. It contains no loop over the data: it says what one thread does, and the launch starts one thread per element.

So the question changes from *iterating* to *covering*: instead of one worker stepping through `N` indices, `N` threads each own one index. The figure at the top draws six elements both ways. The rest of the series asks the same question of every kernel: what does one thread do, and how do threads map onto data?

## 2. Grids, blocks, threads, and the global index

Post 01 defined the three words: a **thread** runs one copy of the kernel, a **block** is a group of up to 1,024 threads that runs on one streaming multiprocessor (SM), and a **grid** is all the blocks of one launch. The host chooses two numbers, the blocks and the threads per block, with the launch syntax, a CUDA extension to C++, shown here as a fragment:

```cpp
// <<< number of blocks, threads per block >>>
vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);
```

Inside the kernel, four built-in, read-only variables tell each thread where it sits. Two depend on the thread (`threadIdx.x` within its block, `blockIdx.x` among the blocks); two describe the launch and are the same for every thread (`blockDim.x`, the threads per block, and `gridDim.x`, the blocks). A thread flattens its two-level position into one index, `i = blockIdx.x * blockDim.x + threadIdx.x`: skip the earlier blocks, then step in by the position inside its own block. With `blockDim.x = 4`, thread 1 of block 1 computes $1 \times 4 + 1 = 5$ and owns element 5. That one line is the whole kernel, from `snippets/vector_add.cu`:

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        C[i] = A[i] + B[i];                          // the work: one add
}
```

Thread `i` reads `A[i]` and `B[i]` and writes `C[i]`; nothing else. The `if` is there because a launch comes in *whole blocks*, so the grid is almost always slightly larger than `N`. The host sizes it with an integer ceiling division:

```cuda
    const int blockSize = 256;
    const int blocksPerGrid = (N + blockSize - 1) / blockSize;   // round up, never down
```

The figure works the smallest case, which the CPU model prints: $N = 6$ in blocks of 4 needs $(6 + 3) / 4 = 2$ blocks, 8 threads, and the last two threads compute $i = 6$ and $i = 7$, fail the guard, and write nothing.

![A row of eight threads, i from 0 to 7, under two braces for block 0 and block 1, above a six-cell array C. Threads 0 to 5 are orange, each with an arrow to its cell of C, and thread 5 is emphasized. Threads 6 and 7 are outlined in red: they fail i less than N and write nothing. Above, blocksPerGrid equals 6 plus 3, divided by 4, equals 2, for 8 threads and 2 surplus.](diagrams/02-launch-config.svg)

*Six elements need two blocks of four: eight threads, of which the guard stops two.*

The committed program launches the same kernel over $N = 1{,}000{,}003$ elements, deliberately not a multiple of 256: 3,907 blocks, 1,000,192 threads, and 189 surplus threads that the guard stops. Without the guard those 189 threads would read and write past the end of the arrays.

> **Why a block size that is a multiple of 32?** The hardware runs a block's threads in **warps** of 32 lanes. A block of 100 threads still occupies 128 lanes (four warps) and wastes 28. Starting points of 128 or 256 are common; the best size also depends on the kernel's register and shared-memory use, and [post 10](../10-roofline/index.md) counts what limits it.

## 3. Two memory spaces: copy in, launch, copy out

The CPU (the *host*) and a discrete GPU (the *device*) have separate physical memories joined by the Peripheral Component Interconnect Express (PCIe) bus. A host pointer means nothing on the device, so the program allocates device memory, copies the inputs across, launches, and copies the result back. The figure draws the six steps and the two roads data travels.

![On the left the host: the host program and h_A, h_B and h_C in host memory. On the right the device: d_A, d_B and d_C in global memory and the vectorAdd kernel, which reaches global memory at a measured 145.0 GB/s. Between them a purple dashed line is the PCIe 3.0 bus of 16 lanes, measured at 11.6 GB/s from pinned memory and 3.5 GB/s from pageable memory. Numbered steps: 1 allocate and 6 free on the device, 2 copy in, 3 launch, 4 synchronize, 5 copy out.](diagrams/03-host-device-copy.svg)

*The kernel's own road to global memory is more than ten times wider than the bus that fills it.*

These are the six steps in `snippets/vector_add.cu`, each call wrapped in `CUDA_CHECK`:

```cuda
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));                                   // 1. allocate
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));       // 2. copy inputs in
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);             // 3. launch
    CUDA_CHECK(cudaGetLastError());         // a bad launch configuration, reported now
    CUDA_CHECK(cudaDeviceSynchronize());    // 4. wait, then report any fault inside the kernel

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));       // 5. copy the result out
    CUDA_CHECK(cudaFree(d_A));                                             // 6. free
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
```

The `h_` and `d_` prefixes keep host and device pointers apart; `bytes` is `N * sizeof(float)`, never `N`. The macro turns a failed call into a message that names its own line:

```cuda
#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                                 \
            return 1;                                                        \
        }                                                                    \
    } while (0)
```

> **A kernel launch is asynchronous.** The CPU *queues* the kernel and runs on to the next line. The copy in step 5 is still safe without step 4, because a blocking `cudaMemcpy` on the default stream waits for the kernel before it copies. What the asynchrony breaks is **timing** and **error reporting**: a host clock stopped right after the launch measures the queueing, not the kernel, and a fault inside the kernel surfaces at some later call. `cudaGetLastError()` right after the launch reports a bad launch configuration at once; `cudaDeviceSynchronize()` gives a host timer a real finish line and reports a fault inside the kernel. Each catches what the other misses.

## 4. Benchmarking honestly

A GPU against CPU number is wrong if it times a single cold run, stops a host clock before the kernel has run, lets the compiler skip work whose result is never read, or mixes pinned and pageable host memory. Two GPU numbers answer two different questions: the kernel alone, on data already in device memory (the best case, and the case inside a pipeline), and the end-to-end time with the copies (the case for one-off work). `snippets/vector_add_bench.cu` measures both, beside the CPU loop on one thread and on all threads with OpenMP (the compiler directives that split a loop across CPU threads), the method of Harris's "How to Implement Performance Metrics in CUDA C/C++":

| Aspect | Approach |
|---|---|
| Repetitions | an untimed warm-up of every variant, then the average of R repetitions |
| GPU timing | CUDA events on the GPU's own clock (`cudaEventElapsedTime`) |
| CPU timing | `std::chrono::steady_clock` |
| Host memory | pinned (page-locked) memory from `cudaMallocHost` for every copy in the table |
| Verification | every element of `C` compared with `A + B` after every variant |

Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, in a laptop with an Intel Core i7-9750H (6 cores, 12 threads), one 8 GB module of DDR4-2667 memory, and a PCIe 3.0 x16 link (as `nvidia-smi` reports it), the benchmark prints:

```text
GPU: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), 14 SMs, 160 GB/s peak (theoretical)
CPU: 12 OpenMP threads
each time: the average of R repetitions after one untimed warm-up

         N      R    CPU 1 thr   CPU OpenMP   GPU+copies   GPU kernel kernel GB/s  check
      1000   2000       0.1 us       6.4 us     167.3 us       3.6 us         3.4     ok
     10000   2000       1.9 us       6.9 us     141.3 us       3.8 us        31.2     ok
    100000   2000      28.1 us      31.5 us     168.3 us       9.8 us       122.2     ok
   1000000    200     897.4 us     420.9 us      1.30 ms      84.8 us       141.5     ok
  10000000     20     13.66 ms     12.15 ms     12.76 ms     827.8 us       145.0     ok

host to device, 40 MB: pinned 11.6 GB/s, pageable 3.5 GB/s
```

Every number in it is measured on that machine, in the quietest of the runs made that day; a reader's card and CPU will give other digits, and the shape is the lesson. The kernel column repeated within 3% in every run with the GPU otherwise idle (808 to 828 µs at 10 million elements); the CPU columns and the copies, which also depend on the host's memory and on whatever else the laptop is running, varied by up to two times. The sizes stop at 10 million elements so that the three pinned host arrays (120 MB) fit comfortably in the laptop's 8 GB. The chart draws the four time columns on logarithmic axes.

![A log-log chart of time in microseconds against N from 1,000 to 10 million elements. The GPU with copies sits on a floor of about 140 to 170 microseconds up to 100,000 elements and then rises to 12.76 ms; the CPU lines start under 10 microseconds and rise to 13.66 and 12.15 ms; the GPU kernel alone stays near 4 microseconds up to 10,000 elements and rises to 827.8 microseconds, just above the dashed floor that 160 GB/s allows.](diagrams/04-benchmark.svg)

*The copies set a floor of about 140 to 170 µs that the CPU never pays; only the kernel alone pulls away.*

Reading the rows:

- **Up to 100,000 elements the CPU wins outright.** The kernel itself takes 3.6 to 9.8 µs, but the launch and the three copies cost about 140 to 170 µs whatever the size, while one CPU thread finishes 100,000 elements in 28.1 µs. OpenMP is no faster than one thread here: waking the threads and waiting for all of them costs more than the work.
- **At 1 million elements the CPU still wins** (897.4 µs on one thread, 420.9 µs with OpenMP, against 1.30 ms with the copies), while the kernel alone takes 84.8 µs.
- **At 10 million elements the end-to-end race is a tie** (13.66 ms, 12.15 ms, 12.76 ms), and the kernel alone is 16.5 times faster than one CPU thread (13.66 ms against 827.8 µs).

The tie has a cause the next section makes precise: the copies move 120 MB across the bus at about 10 GB/s, and the CPU loop moves the same bytes through its own memory at about the same rate. Neither side is adding; both are waiting on memory.

## 5. The memory wall

Each element costs two 4-byte reads and one 4-byte write, 12 bytes, for one FLOP. Its **arithmetic intensity** $I$, the FLOP per byte moved to and from dynamic random-access memory (DRAM), is

$$I = \frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.0833 \text{ FLOP per byte},$$

far below the ridge point, the intensity at which a card's arithmetic rather than its memory becomes the limit, which post 10 derives for the GTX 1650 from the roofline of Williams, Waterman and Patterson. A kernel this low is **memory bound**: its time is the bytes divided by the bandwidth, and arithmetic tricks change nothing. The CPU model prints the count and the shortest time the card's theoretical peak $\beta = 160$ GB/s allows for each size:

```text
3. traffic and the bandwidth floor
  bytes per element = 12 (read A, read B, write C)
  arithmetic int.   = 1/12 = 0.0833 FLOP/byte  -> memory bound
  fastest possible kernel at 160 GB/s:
       1,000 elements:   0.012 MB moved, at least    0.07 us
      10,000 elements:   0.120 MB moved, at least    0.75 us
     100,000 elements:   1.200 MB moved, at least    7.50 us
   1,000,000 elements:  12.000 MB moved, at least   75.00 us
  10,000,000 elements: 120.000 MB moved, at least  750.00 us
```

Those are derived numbers, from `snippets/vector_add.py`. Set against the measured kernel, they show what "fast" means for this kernel: at 10 million elements the kernel needs 827.8 µs against a floor of 750 µs, so it moves 145.0 GB/s, 91% of $\beta$. At 1,000 and 10,000 elements the kernel takes 3.6 to 3.8 µs against a floor below 1 µs: there the time is the fixed cost of a launch, not memory. The lever for a memory-bound kernel is the bytes: consecutive threads reading consecutive addresses, which this kernel already does and [post 06](../06-matrix-transpose/index.md) breaks on purpose, and fewer trips to DRAM across a whole pipeline, which [post 14](../14-kernel-fusion/index.md) buys by fusing kernels.

The same wall stands in front of the CPU. One DDR4-2667 module moves at most $2{,}667 \times 10^6$ transfers per second $\times$ 8 bytes $= 21.3$ GB/s, and the measured OpenMP loop moved 120 MB in 12.15 ms, about 9.9 GB/s. The GPU's kernel wins because it moves about 15 times the bytes per second that the CPU loop reached (145.0 against 9.9 GB/s), not because it adds faster.

## 6. One-off work against a pipeline

The decision is not "GPU good, CPU bad"; it is whether the work pays for the copies. On the author's laptop, from the table above:

| Situation | Better choice | Why |
|---|---|---|
| one-off, up to 1 million elements | CPU | the launch and copies cost more than the whole CPU loop |
| one-off, around 10 million elements | either | the bus and the CPU's memory both run near 10 GB/s |
| data already on the device | GPU | the kernel alone is 16.5 times faster than one CPU thread at 10 million elements |
| a pipeline of many kernels | GPU | the copies are paid once, then every kernel runs at kernel speed |

The table reads the measured numbers of section 4; on another machine the boundaries move, and the reasons stay.

The last two rows are why GPUs carry deep learning: a training step runs many kernels on data that stays on the device, so the bus is crossed once and every kernel after it runs at kernel speed. Vector addition in isolation is the worst case for the GPU. It is still the right first kernel, because every habit it teaches (one thread per element, the guard, the copies in and out, the bandwidth lens) carries into the kernels that do pay off; [post 12](../12-cuda-streams/index.md) overlaps the copies with compute to hide part of the bus.

## 7. Make it run: the model, the program, and the benchmark

The CPU model needs no GPU. From the series root:

```bash
uv run python posts/02-vector-addition/snippets/vector_add.py
```

It prints the launch over 1,000,003 elements, the launch the figure of section 2 draws, and the traffic table of section 5. The launch block:

```text
1. launch over N = 1,000,003
  N                 = 1,000,003 (not a multiple of 256)
  launch            = 3,907 blocks x 256 threads = 1,000,192 threads
  surplus threads   = 189 (fail the guard i < N)
  every i written 1x= True
  matches A + B     = True
```

The two `.cu` programs need an NVIDIA GPU and the CUDA Toolkit. Each builds with one `nvcc` line from the series root; the benchmark adds OpenMP for its CPU column (`-Xcompiler -fopenmp` on Linux):

```bash
nvcc -O3 -arch=sm_75 posts/02-vector-addition/snippets/vector_add.cu -o vector_add && ./vector_add
nvcc -O3 -arch=sm_75 -Xcompiler /openmp posts/02-vector-addition/snippets/vector_add_bench.cu -o vector_add_bench && ./vector_add_bench
```

Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, `vector_add` prints:

```text
vectorAdd<<<3907, 256>>> over N = 1000003: 1000192 threads, 189 surplus
C[0] = 0.5, C[N-1] = 1000002.5
elements equal to A + B: 1000003 of 1000003
```

The benchmark prints the table of section 4 in about four seconds. On Windows the programs are `vector_add.exe` and `vector_add_bench.exe`, and `nvcc` may need a `-ccbin` argument to find the host compiler; [RUNNING.md](../../RUNNING.md) covers both and the `-arch` value for other cards.

## 8. What can go wrong?

The mistakes a first host program makes, each reproduced on the author's GTX 1650 with a variant of `vector_add.cu`.

- **A size in elements where bytes are meant.** `cudaMemcpy(d_A, h_A, N, ...)` copies `N` bytes, a quarter of the array. Every call returns success, and the check reports `250000 of 1000003` elements equal to `A + B`.
- **Checking only the synchronize.** A launch of 2,048 threads per block never runs, yet `cudaDeviceSynchronize()` returns `no error`; only `cudaGetLastError()` reports `invalid configuration argument`. A program that skips it reads back whatever `d_C` held.
- **A host clock around the launch alone.** Stopped right after `vectorAdd<<<...>>>` at 10 million elements, `std::chrono` reads 12 to 15 µs, the time to queue the launch; with `cudaDeviceSynchronize()` before the stop it reads about 875 µs.
- **Timing the first launch.** The first launch of `vectorAdd` over 10 million elements took 15.5 ms by CUDA events and the next three 0.81 ms each: the first pays one-time start-up costs, such as loading the kernel onto the device. The benchmark's untimed warm-up is there for this.
- **Pageable host memory in a transfer benchmark.** The same 40 MB copy runs at 3.5 GB/s from ordinary `new` memory and 11.6 GB/s from `cudaMallocHost` memory, because the driver first copies pageable memory into a pinned staging buffer (Harris, "How to Optimize Data Transfers in CUDA C/C++"). Comparing a pinned run with a pageable one compares two different buses.

## 9. Summary

Vector addition deletes the loop: thread `i = blockIdx.x * blockDim.x + threadIdx.x` adds one element, a ceiling division sizes the grid, and the guard `if (i < N)` stops the surplus threads. Around the launch the host allocates, copies in, launches, synchronizes, copies out, and frees, checking every call. At 12 bytes per FLOP the kernel is memory bound: it reaches 91% of the GTX 1650's peak bandwidth, and whether the GPU wins end to end depends on how often the data crosses the bus.

---

## Common pitfalls

1. **Dropping the guard.** The grid almost always exceeds `N`, so without `if (i < N)` the surplus threads read and write past the arrays; the damage can stay silent until it lands on another allocation.
2. **Reporting only one GPU number.** The kernel alone flatters one-off work and the end-to-end time hides what a pipeline gains; report both, as the benchmark does.
3. **A block size that is not a multiple of 32.** A block of 100 threads occupies four warps, 128 lanes, and wastes 28; start at 128 or 256.
4. **Optimizing the arithmetic of a memory-bound kernel.** At $I = 1/12$ the time is the bytes divided by the bandwidth; judge the kernel by its achieved bandwidth as a percentage of $\beta$, not by its FLOP/s.
5. **Pinning too much host memory.** Pinned memory speeds the copies, but the operating system cannot page it out, so every pinned megabyte is taken from everything else; Harris advises against over-allocating it, and the benchmark keeps its pinned arrays to 120 MB.

---

## Further reading

- Harris, M., "An Even Easier Introduction to CUDA" (NVIDIA Technical Blog, 25 January 2017). The first kernel and launch in a few pages.
- Harris, M., "How to Implement Performance Metrics in CUDA C/C++" (NVIDIA Technical Blog, 7 November 2012). Timing with CUDA events, and why a host timer needs a synchronize.
- Harris, M., "How to Optimize Data Transfers in CUDA C/C++" (NVIDIA Technical Blog, 4 December 2012). Pinned against pageable host memory, and the staging copy behind the difference.
- Harris, M., "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops" (NVIDIA Technical Blog, 22 April 2013). The grid-stride loop, which lets a grid smaller than the data cover it.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, consulted October 2026. The execution model, the built-in variables, the launch syntax, and asynchronous execution.
- Williams, S., Waterman, A. and Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (*Communications of the ACM* 52(4), April 2009). The memory-bound against compute-bound picture that post 10 draws.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 03 - Matrix multiplication: from a memory-bound dot product to shared-memory tiling](../03-matrix-multiplication/index.md):** the next kernel reuses each loaded value many times, and shared memory turns that reuse into speed.
- **[Post 11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy](../11-profiling-debugging/index.md):** puts this benchmark under a profiler, which reports the achieved bandwidth directly and catches the unguarded thread.
