# 12 - CUDA streams: overlapping copy and compute for free throughput

> **TL;DR.** On the default stream a program's copy in, kernel and copy out run one after another, so the copy engines and the streaming multiprocessors (SMs) take turns being idle. A **stream** is an ordered queue of GPU work, and work in different streams may run at the same time, so cutting the data into chunks, each copied in, computed and copied out on its own stream, **pipelines** copies and kernels. Asynchronous copies need **pinned** host memory: on the author's GTX 1650, vector addition over $2^{24}$ floats with its copies took 53.60 ms from pageable memory and 18.95 ms from pinned memory, the largest single gain. A pipeline hides only what can run beside something else, and on this laptop the copies in and out mostly took turns, so four or more streams hid the kernel and the run settled near the copies' own 17.80 ms.
>
> **Prerequisites:** [post 02](../02-vector-addition/index.md).
> **Safe to skip?** Skip it if the reader can already pipeline copies and kernels across streams from pinned memory, say what the default stream orders, and predict from a workload's stage times how much a pipeline can hide on a given card.
>
> **After reading, you will be able to:**
>
> - Explain why work on the default stream leaves the copy engines and the SMs idle in turn, and compute an engine's busy share from the stage times.
> - Pipeline a workload across streams with `cudaMemcpyAsync` and per-stream launches, so that one chunk's copies overlap another chunk's kernel.
> - Say why `cudaMemcpyAsync` needs pinned host memory to overlap, and what happens from pageable memory.
> - Predict from a workload's measured stage times, with the streams model, how much a pipeline can hide and where streams help most and least.
> - Check a card's copy engines with `asyncEngineCount` and a two-way copy test before counting on the two directions to overlap.

![Two timelines on one millisecond axis, each with lanes for the copy in, the kernel and the copy out. Top, serial from pinned memory: the copy in, then the kernel, then the copy out, one lane busy at a time. Bottom, four streams: four chunks, the kernel of one chunk running while the next chunk is copied in, finishing near the time the two copies take end to end.](diagrams/01-serial-vs-pipeline.svg)

*Measured on the GTX 1650: four streams let the kernel run under the copies, but the copies themselves still follow one another.*

---

## 1. The question: who is idle while a copy runs?

[Post 02](../02-vector-addition/index.md) wrote the whole host program around a kernel: allocate, copy in, launch, wait, copy out, every call on the **default stream**, each starting only when the one before it had finished. The GPU, though, is several machines in one package: the SMs run kernels, and one or more **copy engines**, direct memory access (DMA) units, move data across the Peripheral Component Interconnect Express (PCIe) bus without the SMs. In a serial program each works only during its own stage.

`snippets/streams.cu` measures the three stages alone for vector addition over $N = 2^{24}$ floats, 64 MiB (67.1 MB) per array, from pinned host memory on the author's GTX 1650: copying `A` and `B` in takes 11.01 ms, the kernel 1.37 ms, copying `C` out 6.79 ms. End to end that is 19.17 ms, and the SMs are busy for 1.37 of them, 7% of the run; the copy engine that brings data in is busy for 57%. Vector addition is the extreme case, a kernel that moves 12 bytes per addition and so finishes long before the bus has delivered its inputs. The program also runs a second kernel with the same copies and the same result, whose 64 extra rounds of `c = sqrtf(c * c)` per element (exact for the small integers the arrays hold, so `c` does not change) make it take 9.44 ms in the run of section 6, nearly as long as the copy in.

The question of this post is how much of the idle time streams can fill, and what decides it.

## 2. What a stream is, and what the default stream does

A **stream** is a sequence of commands that run in order. The CUDA C++ Programming Guide (CUDA 12.3 edition, section 3.2.8.5) states the rule both ways: commands in one stream "execute in order", while different streams "may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness". A stream is created with `cudaStreamCreate(&s)` and named as the last argument of a copy (`cudaMemcpyAsync(dst, src, bytes, kind, s)`) or the fourth parameter of a launch (`kernel<<<grid, block, 0, s>>>(...)`).

The **default stream** is the stream a call uses when it names none. Without a `--default-stream` flag, `nvcc` compiles for the legacy default stream, which the Guide calls the NULL stream and makes special: in its list of implicit synchronizations (section 3.2.8.5.4), two commands from different streams cannot run concurrently if, between them, the host issues "a page-locked host memory allocation", "a device memory allocation", "a device memory set", "a memory copy between two addresses to the same device memory", or "any CUDA command to the NULL stream". A plain `cudaMemcpy` is such a command. Compiling with `--default-stream per-thread` turns the default stream into an ordinary stream per host thread, and a stream created with `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` does not synchronize with the legacy one.

![Two timelines with lanes for stream 1, stream 2 and the default stream. Top: the two streams each copy in, run a kernel and copy out in order, and stream 2's work overlaps stream 1's. Bottom: the same work with one cudaMemcpy on the default stream issued after stream 1's kernel; it waits for the work before it, all later work in both streams waits for it, and the run ends later.](diagrams/02-default-stream.svg)

*A stream orders its own commands; the legacy default stream orders everything around it.*

The figure draws the consequence: one stray `cudaMemcpy` in the middle of a pipeline is a fence across every stream. Whether two streams' commands then really run at once depends on the hardware the Guide names: a device copies to or from the GPU during a kernel when its `asyncEngineCount` property is greater than zero, it runs kernels of different streams together when `concurrentKernels` is 1, and it "can overlap copies to and from the device" when `asyncEngineCount` "is equal to 2". The harness prints both properties first: the GTX 1650 reports `asyncEngineCount 6, concurrentKernels 1`. The Guide's rule for copies in both directions names only the value 2, and the runtime reference describes the property as the "Number of asynchronous engines", so section 6 tests what the 6 is worth.

## 3. Pinned memory: the requirement for an asynchronous copy

Ordinary host memory from `malloc`, `new` or `std::vector` is **pageable**: the operating system may move or page out any of it, so a DMA engine cannot be pointed at it. The driver first copies each piece into a **pinned** (page-locked) staging buffer it owns and then lets the DMA engine copy that, as Harris describes in "How to Optimize Data Transfers in CUDA C/C++" (NVIDIA Technical Blog, 4 December 2012). Memory from `cudaMallocHost` (or `cudaHostAlloc`, or a range locked with `cudaHostRegister`) is pinned from the start, and the DMA engine reads it directly.

![Host on the left and device on the right of a purple dashed line. Top, pageable memory: the CPU copies each piece of the array into a pinned staging buffer owned by the driver, in red, and only then does DMA carry it to global memory. Bottom, pinned memory from cudaMallocHost: DMA reads the array directly. Each row gives the measured serial and four-stream times of vector addition with its copies.](diagrams/03-pinned-memory.svg)

*The staging copy costs time on its own, and it also takes away the asynchrony a stream needs.*

The staging copy costs twice. It is CPU work at the speed of the host's memory, and it makes the copy synchronous: the Guide's section 3.2.8.1 lists copies "suffixed with Async" among the operations that return before the device finishes, then warns that "Async memory copies might also be synchronous if they involve host memory that is not page-locked", and section 3.2.8.3 requires, for a copy to overlap a kernel, that "If host memory is involved in the copy, it must be page-locked". In the harness, vector addition took 53.60 ms from pageable memory and 18.95 ms from pinned memory, both serial; [post 02](../02-vector-addition/index.md) measured the same gap on a single 40 MB copy, 3.5 against 11.6 GB/s. Four streams from pageable memory took 50.01 ms: nothing in the run was asynchronous to overlap.

Pinned memory is taken from everything else on the machine, since the operating system can no longer page it out, as post 02's pitfalls said, and the Best Practices Guide calls pinning "a heavyweight operation compared to most normal system memory allocations". The harness pins its three arrays, 201 MB, once at startup, outside every timed region, on a laptop with one 8 GB memory module.

## 4. The pipeline: chunks on streams

With pinned arrays, the recipe is to cut the work into `NS` chunks and give chunk `i` its own stream, on which it is copied in, computed and copied out. The kernel is post 02's, one thread per element, with the optional extra rounds; thread `i` handles element `i` of its chunk, with the global index `blockIdx.x * blockDim.x + threadIdx.x` and the usual guard:

```cuda
__global__ void vectorAdd(const float* A, const float* B, float* C, int n, int extra) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's element of the chunk
    if (i < n) {
        float c = A[i] + B[i];
        for (int k = 0; k < extra; ++k)               // extra = 0: plain vector addition
            c = sqrtf(c * c);                         // exact for these small integers, so c is unchanged
        C[i] = c;
    }
}
```

The serial version puts everything on the default stream:

```cuda
void serial(Host h, Dev d, int extra) {
    CUDA_CHECK(cudaMemcpy(d.A, h.A, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.B, h.B, BYTES, cudaMemcpyHostToDevice));
    vectorAdd<<<(N + THREADS - 1) / THREADS, THREADS>>>(d.A, d.B, d.C, N, extra);
    CUDA_CHECK(cudaMemcpy(h.C, d.C, BYTES, cudaMemcpyDeviceToHost));
}
```

The pipelined version issues each chunk's four commands on its stream, with an optional CUDA event after each one for the timeline of section 6:

```cuda
void pipelined(Host h, Dev d, int extra, cudaStream_t* s, int NS, cudaEvent_t (*ev)[4] = nullptr) {
    const int chunk = N / NS;                        // N is a multiple of every NS used
    const size_t chunkBytes = (size_t)chunk * sizeof(float);
    const int grid = (chunk + THREADS - 1) / THREADS;
    for (int i = 0; i < NS; ++i) {
        int off = i * chunk;
        cudaMemcpyAsync(&d.A[off], &h.A[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
        if (ev) cudaEventRecord(ev[i][0], s[i]);
        cudaMemcpyAsync(&d.B[off], &h.B[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
        if (ev) cudaEventRecord(ev[i][1], s[i]);
        vectorAdd<<<grid, THREADS, 0, s[i]>>>(&d.A[off], &d.B[off], &d.C[off], chunk, extra);
        if (ev) cudaEventRecord(ev[i][2], s[i]);
        cudaMemcpyAsync(&h.C[off], &d.C[off], chunkBytes, cudaMemcpyDeviceToHost, s[i]);
        if (ev) cudaEventRecord(ev[i][3], s[i]);
    }
    CUDA_CHECK(cudaGetLastError());
}
```

Inside one stream the kernel waits for its two copies and the copy out waits for the kernel; across streams nothing is ordered, so chunk 1 can be copied in while chunk 0 computes. The loop returns before any command has run, so `h.C` may be read only after a synchronization (`cudaStreamSynchronize`, `cudaDeviceSynchronize` or an event). The harness times each run with events on the legacy default stream, which by the rule of section 2 wait for every stream. Issuing each chunk's commands together is the "version 1" of Harris's "How to Overlap Data Transfers in CUDA C/C++" (NVIDIA Technical Blog, 13 December 2012); his "version 2" issues all copies in, then all kernels, then all copies out. Version 2 was faster on his Tesla C1060, which has one copy engine, version 1 on the C2050, which has two, and on the K20c the two tied. The harness measures version 1 only.

## 5. The model: what a pipeline can hide

`snippets/streams_model.py` schedules the three stages of every chunk on the engines: a stage starts when the stage before it in its chunk has finished and its engine is free. Its first part is the textbook case, three equal stages and an engine for each:

```bash
uv run python posts/12-cuda-streams/snippets/streams_model.py
```

```text
streams model (3 equal stages: H2D, compute, D2H)
  chunks (streams)   = 4
  serial makespan    = 12 stage-times  (every stage end to end)
  pipelined makespan = 6 stage-times  (= N + 2 once filled)
  speedup            = 2.00x
  engine utilization = H2D 67%, compute 67%, D2H 67%
  with 64 chunks     = 2.91x speedup, compute 97% utilized (-> 3x, 100%)
```

Serial, $N$ chunks take $3N$ stage-times and each engine is busy a third of them; pipelined, they take $N + 2$, two to fill the pipeline and one per chunk after that, with every engine busy $N$ of them. More chunks amortize the fill and drain, and the speed-up approaches 3. That ceiling holds only if the stages are equal and the copies in and out really run at the same time.

The second part feeds the model the stage times the harness measured (constants at the top of the script, dated) and predicts the pipelined time under two assumptions about the bus: the copies in and out run at full speed together, or one copy runs at a time and the copy engine takes whichever copy is ready first:

```text
measured stage times on the GTX 1650 (ms), and the pipelined time the model predicts
  vector add: copy in 11.01, kernel 1.37, copy out 6.79; stages end to end 19.17
    chunks   both directions at once   one copy at a time
         1      19.17 ms   1.00x         19.17 ms   1.00x
         2      15.09 ms   1.27x         17.80 ms   1.08x
         4      13.05 ms   1.47x         17.80 ms   1.08x
         8      12.03 ms   1.59x         17.80 ms   1.08x
        16      11.52 ms   1.66x         17.80 ms   1.08x
        32      11.26 ms   1.70x         17.80 ms   1.08x
  64 extra rounds: copy in 11.01, kernel 9.44, copy out 6.79; stages end to end 27.24
    chunks   both directions at once   one copy at a time
         1      27.24 ms   1.00x         27.24 ms   1.00x
         2      19.12 ms   1.42x         19.12 ms   1.42x
         4      15.07 ms   1.81x         18.46 ms   1.48x
         8      13.04 ms   2.09x         18.13 ms   1.50x
        16      12.02 ms   2.27x         17.97 ms   1.52x
        32      11.52 ms   2.37x         17.88 ms   1.52x
```

Unequal stages change the ceiling. With both directions running at once, the pipeline can never beat its longest stage, here the copy in of 11.01 ms, so vector addition is capped at 11.26 ms and 1.70 times however many streams it gets. With one copy at a time, the floor is the two copies end to end, 17.80 ms, for both kernels, and the only stage a pipeline can hide is the kernel.

## 6. Measured on the GTX 1650

`snippets/streams.cu` runs everything on the same pinned and pageable arrays: the stages alone, a copy each way one after the other and then at once on two streams, and the table for both kernels, each cell the fastest of 5 passes after about a sixth of a second of untimed kernels, with every element of `C` checked, and the SM clock before and after (`clock64()` cycles over CUDA-event time, as in posts 09 and 11). Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), asyncEngineCount 6, concurrentKernels 1
N = 16777216 floats, 67.1 MB per array; fastest of 5 passes; GB/s = MB over the time in ms
stages alone, pinned: copy A and B in 11.01 ms (12.2 GB/s), copy C out 6.79 ms (9.9 GB/s)
kernels alone: vector add 1.37 ms (147.0 GB/s on the device), with 64 extra rounds 9.44 ms
one copy each way, 67.1 MB: one after the other 11.89 ms, at the same time on two streams 10.10 ms
                      vector add                    64 extra rounds
method                time (ms)   GB/s  speed-up    time (ms)  speed-up  check
serial, pageable          53.60    3.8     1.00x        57.60     1.00x  ok
serial, pinned            18.95   10.6     2.83x        26.62     2.16x  ok
1 stream, pinned          18.82   10.7     2.85x        26.65     2.16x  ok
2 streams, pinned         17.51   11.5     3.06x        19.94     2.89x  ok
4 streams, pinned         17.00   11.8     3.15x        18.11     3.18x  ok
8 streams, pinned         16.66   12.1     3.22x        17.34     3.32x  ok
16 streams, pinned        16.77   12.0     3.20x        17.22     3.35x  ok
32 streams, pinned        16.76   12.0     3.20x        17.14     3.36x  ok
4 streams, pageable       50.01    4.0     1.07x        58.39     0.99x  ok
timeline, 4 streams, vector add, ms from the start to the end of: A in, B in, kernel, C out
  stream 0:   1.46   2.86   3.26   5.91
  stream 1:   5.11   6.92   7.31   9.97
  stream 2:   9.21  10.97  11.35  13.94
  stream 3:  13.21  14.94  15.28  16.92
timeline, 4 streams, extra rounds, ms from the start to the end of: A in, B in, kernel, C out
  stream 0:   1.41   2.86   5.25   7.89
  stream 1:   4.28   6.03   8.43  11.16
  stream 2:   8.37  11.02  13.40  15.34
  stream 3:  12.49  14.31  16.68  18.32
SM clock: 1589 MHz before the timed rows, 1005 MHz after
```

Every number in it is measured, and the SM clock line needs reading: the run started at 1,589 MHz, but the GPU, near 85 degrees C while other programs shared it, was held at about 1,005 MHz, as the second reading shows; by inference the limit was its temperature, since `nvidia-smi` gives 87 degrees C as the card's maximum operating temperature. The kernel with 64 extra rounds took 9.44 ms alone here, against 5.5 to 6.5 ms in runs whose clock read about 1,700 MHz; vector addition and the copies do not depend on the SM clock, but the copies do depend on the laptop's memory and on whatever else it runs. Four runs taken within ten minutes agreed within 4% on the serial pinned row and the 4, 8 and 16-stream rows; the other rows varied by up to 13%. Effective GB/s is the 201 MB that cross the bus, two arrays in and one out, divided by the time.

Four things stand out.

**Pinned memory is the largest single gain**, 2.83 times on the serial run before any stream is involved, and streams cannot recover it from pageable memory (section 3).

**The pipeline hides the kernel.** The stages measured alone add up to 19.17 ms for vector addition and 27.24 ms with the extra rounds; the serial runs took 18.95 and 26.62 ms, and the best pipelines 16.66 ms (8 streams) and 17.14 ms (32 streams), both close to 17.80 ms, the two copies end to end: the kernel, 1.37 ms in one case and 9.44 ms in the other, disappeared under the copies. The hero is the fastest 4-stream pass of the second kernel, which ended at 18.32 ms: each chunk's kernel runs while the next chunk is copied in.

**The two copy directions overlap only a little on this laptop.** One copy in and one copy out of 64 MiB took 11.89 ms one after the other and 10.10 ms at the same time on two streams, so running them together saved 15%. That is why, by inference, every pinned pipeline lands near the two copies end to end, 17.80 ms, the floor of the model's "one copy at a time" column (vector addition's 16.66 ms is a little under it, as a 15% overlap allows), and well above the 11 ms its "both directions at once" column allows. The PCIe 3.0 x16 link (as `nvidia-smi` reports it) carries both directions at once, and the card reports `asyncEngineCount 6`, but neither says that both directions run at full speed together on this machine. The measurement is consistent with a limit on the host side, where post 02 counted one DDR4-2667 module at most $2{,}667 \times 10^6 \times 8$ bytes $= 21.3$ GB/s and the two directions alone ask for 12.2 plus 9.9 GB/s, but the harness does not separate the causes; it shows only that on this machine a pipeline cannot count on the copies overlapping each other.

**Past four streams, more change little.** From 4 to 32 streams the pinned times stayed within 2% of each other for vector addition and 6% for the other kernel, while the chunks shrank from 16 MiB to 2 MiB; the 1-stream row is the serial work issued asynchronously on one stream, and it overlaps nothing: it took 18.82 ms against the serial row's 18.95.

![Two horizontal bar charts of end-to-end time, vector addition on the left and the kernel with 64 extra rounds on the right, for serial pageable, serial pinned, 1, 2, 4, 8, 16 and 32 streams pinned, and 4 streams pageable. The pageable rows are red and far the longest; the pinned pipelines cluster near a dashed line marking the two copies measured alone.](diagrams/04-streams-perf.svg)

*Pinned memory takes most of the time out; streams then trim the kernel, down to the copies' own floor.*

## 7. When streams help most, and least

The model and the table give one rule: a pipeline can hide every stage except the longest engine's total, and how many engines there are is a property of the card and the machine, measured, not assumed.

- **Most:** when the stages are comparable and run on separate engines, where a pipeline approaches the ideal speed-up of 3 of section 5. On the GTX 1650 the kernel with extra rounds is the nearest case, and its pipeline took 9.48 ms off a serial run of 26.62 ms, no more, because the copies overlapped each other only a little.
- **Little:** when one stage dominates. Vector addition's kernel is under a tenth of its copies, so even a perfect pipeline saves only that tenth; the lever there is to cross the bus less, by keeping data on the device across kernels (post 02's training-step argument) or by fusing kernels ([post 14](../14-kernel-fusion/index.md)).
- **Little, the other way:** a kernel far longer than its copies already keeps the SMs busy; hiding its copies saves their small share and no more.
- **Nothing, or worse:** pageable memory, a command on the legacy default stream in the middle of the loop, or chunks so small that the fixed cost of each command adds up (32 chunks of 2 MiB were not yet that small here).

Before relying on an overlap, a program can check it as the harness does: print `asyncEngineCount` and `concurrentKernels`, time a copy each way alone and together, and compare a serial run with a pipelined one. Nsight Systems would show the overlap directly, but on this laptop `nsys` needs administrator rights ([post 11](../11-profiling-debugging/index.md)), so the timeline here comes from CUDA events.

## 8. Make it run

The harness needs an NVIDIA GPU and the CUDA Toolkit, builds from the series root with `-arch` set to the card's compute capability, and runs in about 10 seconds; the model needs only Python:

```bash
nvcc -O3 -arch=sm_75 posts/12-cuda-streams/snippets/streams.cu -o streams && ./streams
uv run python posts/12-cuda-streams/snippets/streams_model.py
```

On Windows the program is `streams.exe`, and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both. The program pins 201 MB of host memory and allocates 201 MB on the device; it exits with status 1 if any element of `C` is wrong. To feed the model the reader's own card, copy the printed stage times into `MEASURED` at the top of `streams_model.py`.

## 9. What can go wrong?

- **Timing a card that has dropped its clock.** The host-side check between rows leaves the GPU idle for about a fifth of a second; a first version of the harness, warmed up only at the start, gave identical work times more than two times apart. Untimed kernels before every row fixed it, and the printed SM clock catches a run taken while the laptop's charger flapped, the card was hot, or another program shared the GPU.
- **Reading results before the streams finish.** `cudaMemcpyAsync` returns at once; the host must synchronize before it reads `h.C`.
- **Freeing or reusing a buffer a stream still uses.** Freeing pinned memory with `cudaFreeHost` or overwriting a host chunk while its copy is in flight corrupts the copy; an event recorded after the copy and waited on with `cudaEventSynchronize` or `cudaStreamWaitEvent` orders it.

---

## Common pitfalls

1. **Copying from pageable memory.** `cudaMemcpyAsync` from `malloc` or `std::vector` memory is staged by the CPU and runs synchronously, with no error; the pipeline then gains nothing.
2. **A stray default-stream command inside the loop.** A `cudaMemcpy`, `cudaMemset` or kernel with no stream on the legacy default stream waits for all other streams and makes them wait for it.
3. **Allocating inside the loop.** `cudaMallocHost` and `cudaMalloc` are among the Guide's implicit synchronizations and are slow; allocate once, before the timed work.
4. **Ordering streams by hope.** If stream B reads what stream A wrote, record a `cudaEvent` in A and call `cudaStreamWaitEvent` in B; different streams are never ordered for free.
5. **Assuming the textbook speed-up of 3.** A pipeline hides all but its longest engine's work, and the card or the machine may not overlap copies in both directions; measure the stages and run the model first.

---

## Further reading

- Harris, M., "How to Overlap Data Transfers in CUDA C/C++" (NVIDIA Technical Blog, 13 December 2012). The chunked pipeline, the two issue orders, and their results on three Tesla cards.
- Harris, M., "How to Optimize Data Transfers in CUDA C/C++" (NVIDIA Technical Blog, 4 December 2012). Pinned against pageable memory and the staging copy.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, CUDA 12.3 edition, sections 3.2.6 "Page-Locked Host Memory" and 3.2.8 "Asynchronous Concurrent Execution", consulted October 2026. Streams, the default stream, implicit synchronization, and the device properties that decide overlap.
- NVIDIA, *CUDA C++ Best Practices Guide*, section "Asynchronous and Overlapping Transfers with Computation", consulted October 2026. Staged copies and execution with streams.
- NVIDIA, *Nsight Systems Documentation*, consulted October 2026. How a timeline shows copies overlapping kernels.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 13 - Cutting launch overhead: CUDA Graphs and stream-ordered allocation](../13-cuda-graphs/index.md):** once copies and kernels overlap, the next cost is the launch itself, the host's issue time and a gap of about 1.5 µs on the GPU before each kernel, which a graph narrows and issues with one host call.
- **[Post 11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy](../11-profiling-debugging/index.md):** the tools that show where a pipeline's time goes, and why the timeline here comes from CUDA events.
