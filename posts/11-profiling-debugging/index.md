# 11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy

> **TL;DR.** Before a kernel is timed it is checked: on the author's GTX 1650, `compute-sanitizer` named the thread that read past a 24-byte array, the line of a shared-memory race, and a barrier that half a block never reached, two of them bugs the programs never noticed. Then it is measured against post 10's model: a SAXPY with 10 registers per thread has 100% theoretical occupancy from 64 threads per block up and reaches 145.2 GB/s, 91% of the card's peak and of its roofline ceiling, but at 32 threads per block the 16-block limit halves its warps and it loses a quarter of that. Nsight Systems, the timeline, and Nsight Compute, the per-kernel counters, both refused to run without administrator rights on this laptop, so the post gives their commands, their errors, and what unlocks them.
>
> **Prerequisites:** [post 10](../10-roofline/index.md).
> **Safe to skip?** Skip it if the reader can already run memcheck, racecheck and synccheck and read their reports, compute a kernel's theoretical occupancy from its register count and check it with the occupancy API, and choose between Nsight Systems and Nsight Compute.
>
> **After reading, you will be able to:**
>
> - Run the profile, don't guess loop: check a kernel with compute-sanitizer, measure it, read its bottleneck, fix one limiter, and repeat.
> - Catch out-of-bounds accesses with memcheck, shared-memory races with racecheck, and divergent barriers with synccheck, and read their reports.
> - Choose Nsight Systems for the timeline and Nsight Compute for one kernel's counters, run each, and unlock the counters when the driver denies them.
> - Compute a kernel's theoretical occupancy from its register count by hand, and check it with the CUDA runtime's occupancy API.
> - Read a measured bandwidth against the roofline ceiling and against the theoretical occupancy of each block size.

![A loop of five cards: write the kernel, saxpy; check it with memcheck and racecheck, 0 errors; measure it with CUDA events, 145.2 GB/s; read the bottleneck, memory bound at 91% of its roof; fix one limiter and nothing else; and an arrow from the fix back to the check.](diagrams/01-profiling-loop.svg)

*Correctness is checked before anything is timed, and every fix goes back through the check and the measurement.*

---

## 1. The question: is it correct, and how close is it to the ceiling?

Every post so far ended with a measured number: post 02's vector addition at 145 GB/s, 91% of the GTX 1650's 160 GB/s theoretical peak; post 04's warp-shuffle reduction at 154.6 GB/s, 96.6%; post 06's padded transpose at 89.5% of a copy kernel. Each was timed with CUDA events around a kernel checked on the host. A host check catches a wrong answer, not a thread that wrote past an array and happened to hit nothing, and a time does not say why a kernel is slow.

[Post 10](../10-roofline/index.md) derived the ceilings by counting: the bandwidth roof $\beta = 160$ GB/s, the compute roof of 3,118 GFLOP/s (billions of floating-point operations, FLOP, per second) at the 1,740 MHz boost clock, the ridge at 19.5 FLOP per byte, and the occupancy that registers, shared memory and block size allow. This post measures a kernel against that model in the loop the figure draws: write, check, measure, read the bottleneck, fix one thing, repeat. The check comes first, because a kernel that touches the wrong memory can post a fast, meaningless number.

## 2. Correctness first: `compute-sanitizer`

`compute-sanitizer` runs an unmodified CUDA program under instrumentation and reports each bad access with its kernel, thread, block and address. It ships with the CUDA Toolkit (version 2023.3.0 with CUDA 12.3) and replaced CUDA-MEMCHECK, which the CUDA 12.0 release notes list as "removed from CUDA 12.0, and has been replaced with Compute Sanitizer". Its documentation describes four tools:

| Tool | Flag | Reports |
|---|---|---|
| memcheck | `--tool memcheck` (the default) | out-of-bounds and misaligned accesses to global, local and shared memory, hardware exceptions, and leaks with `--leak-check full` |
| racecheck | `--tool racecheck` | shared-memory data access hazards |
| synccheck | `--tool synccheck` | invalid uses of `__syncthreads()`, `__syncwarp()` and their Cooperative Groups counterparts |
| initcheck | `--tool initcheck` | reads of global memory that nothing initialized |

The first bug is post 02's missing guard. `snippets/missing_guard.cu` adds $N = 6$ floats in blocks of 4 threads, so the grid of 2 blocks has 8 threads and threads 6 and 7 have no element. With `--no-guard` it runs the kernel without `if (i < N)`:

```cuda
__global__ void vectorAddNoGuard(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];              // BUG: no guard; threads 6 and 7 read and write past the end
}
```

On its own, the broken program notices nothing:

```text
vectorAddNoGuard<<<2, 4>>> over N = 6: 8 threads, 2 surplus
elements equal to A + B: 6 of 6
```

The surplus threads touch bytes 24 to 31 of 24-byte arrays, memory no other array uses. Built with `-lineinfo`, so that reports can name source lines, and run under memcheck, the same program gives:

```text
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at vectorAddNoGuard(const float *, const float *, float *, int)+0x70 in C:/Users/admin/Desktop/Learning Cuda/posts/11-profiling-debugging/snippets/missing_guard.cu:38
=========     by thread (2,0,0) in block (1,0,0)
=========     Address 0x502c00018 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x502c00000 of size 24 bytes
=========
========= Invalid __global__ read of size 4 bytes
=========     at vectorAddNoGuard(const float *, const float *, float *, int)+0x70 in C:/Users/admin/Desktop/Learning Cuda/posts/11-profiling-debugging/snippets/missing_guard.cu:38
=========     by thread (3,0,0) in block (1,0,0)
=========     Address 0x502c0001c is out of bounds
=========     and is 5 bytes after the nearest allocation at 0x502c00000 of size 24 bytes
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========
CUDA error unspecified launch failure at C:\Users\admin\Desktop\Learning Cuda\posts\11-profiling-debugging\snippets\missing_guard.cu:57
========= Target application returned an error
========= ERROR SUMMARY: 3 errors
```

The report names the kernel, line 38, and the threads by their index in their block: threads 2 and 3 of block 1, that is `i = 1 * 4 + 2 = 6` and `i = 7`. Their addresses lie 24 and 28 bytes from the start of a 24-byte allocation, the first float past the end (the tool counts "1 bytes after" from the allocation's last byte). The first bad access is a read; memcheck stops the kernel there, so the launch fails and the writes never happen. With the guard back, memcheck counts nothing:

```text
========= COMPUTE-SANITIZER
vectorAdd<<<2, 4>>> over N = 6: 8 threads, 2 surplus
elements equal to A + B: 6 of 6
========= ERROR SUMMARY: 0 errors
```

![Eight threads, two blocks of four, above a six-element array of 24 bytes. Threads 0 to 5 reach elements 0 to 5. Threads 6 and 7, thread 2 and thread 3 of block 1, reach past the end into a dashed red region, which memcheck reports as invalid reads at bytes 24 and 28.](diagrams/02-sanitizer-catch.svg)

*The two surplus threads run past the 24-byte arrays; memcheck names each by block and thread and gives its distance past the allocation.*

The documentation states a limit: buffers can sit back to back, and then "an overflow access into the first buffer will simply happen in the second buffer and not be detected as out-of-bounds"; the `--padding` option leaves a gap after every allocation.

## 3. Races and barriers: racecheck and synccheck

`snippets/shared_race.cu` sums 256 values per block in shared memory, 64 blocks checked against the host, and carries two bugs from [post 09](../09-synchronization/index.md) behind flags. With `--race`, the last six steps run in one warp with no barrier, the plain tail post 09 broke:

```cuda
        if (MODE == RACE) {                       // no barrier: assumes the warp runs in lockstep
            for (int off = 32; off > 0; off >>= 1)
                s[tid] += s[tid + off];
```

Every block sum comes out wrong, and racecheck says where:

```text
========= COMPUTE-SANITIZER
========= Warning: Race reported between Read access at void blockSum<(int)1>(const int *, int *)+0x200 in C:/Users/admin/Desktop/Learning Cuda/posts/11-profiling-debugging/snippets/shared_race.cu:53
=========     and Write access at void blockSum<(int)1>(const int *, int *)+0x240 in C:/Users/admin/Desktop/Learning Cuda/posts/11-profiling-debugging/snippets/shared_race.cu:53 [7936 hazards]
=========
plain tail, no barrier: wrong block sums 64 of 64  (breaks the rule)
========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)
```

The read of `s[tid + off]` and the write of `s[tid]` on line 53 race between lanes of one warp: 7,936 hazards over the 64 blocks, 124 per block, reported as one warning. Post 09's fix, `__syncwarp()` between every read and every write (the default mode), gives `0 hazards displayed (0 errors, 0 warnings)` and right sums.

With `--divergent-barrier` the kernel runs the whole tree with its `__syncthreads()` inside the branch:

```cuda
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
            __syncthreads();                      // reached only by the threads with tid < stride
        }
    }
    __syncthreads();                              // every thread waits here before thread 0 writes the sum
```

At stride 128, threads 0 to 127 wait at the barrier inside the branch while threads 128 to 255 go on to the final one. Every block sum came out right, and synccheck still reports the break:

```text
========= COMPUTE-SANITIZER
========= Barrier error detected. Divergent thread(s) in block.
=========     at blockSumDivergent(const int *, int *)+0x420 in C:/Users/admin/Desktop/Learning Cuda/posts/11-profiling-debugging/snippets/shared_race.cu:80
=========     by thread (128,0,0) in block (19,0,0)
=========
========= Barrier error detected. Divergent thread(s) in block.
=========     at blockSumDivergent(const int *, int *)+0x420 in C:/Users/admin/Desktop/Learning Cuda/posts/11-profiling-debugging/snippets/shared_race.cu:80
=========     by thread (129,0,0) in block (19,0,0)
=========
========= Warning: Error buffer overflow has been detected. Some records have been dropped
=========
__syncthreads inside if (tid < stride): wrong block sums 0 of 64  (breaks the rule)
========= ERROR SUMMARY: 257 errors
========= ERROR SUMMARY: 254 errors were not printed. Use --print-limit option to adjust the number of printed errors
=========
========= 16064 errors were skipped (error buffer overflow).
```

Each record is one thread waiting at the final barrier on line 80 while others of its block wait at another (which thread and block come first changes from run to run; a rerun named thread 192 of block 6); the tool counted 257 errors, printed 2, and dropped 16,064 more records when its buffer filled. It reports this class "for a barrier that does not support this on the current architecture": from compute capability 7.0 a barrier waits only for the threads that have not exited (post 09), so a version without the final barrier, whose skipping threads simply exit, gave 0 errors. The Programming Guide's rule, that the condition must evaluate identically across the block, is broken either way.

## 4. What the compiler reports: registers, and the occupancy they allow

The demo, `snippets/occupancy_demo.cu`, runs SAXPY (single-precision $a$ times $x$ plus $y$), `y = a * x + y`, over 16,777,216 floats, one thread per element:

```cuda
__global__ void saxpy(const float* x, float* y, float a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's element
    if (i < N)                                       // guard the surplus threads
        y[i] = a * x[i] + y[i];                      // one fused multiply-add
}
```

The flag `-Xptxas=-v` makes `ptxas`, the assembler that turns PTX (parallel thread execution, the virtual instruction set) into machine code, print each kernel's resources. For `saxpy`:

```text
ptxas info    : Compiling entry function '_Z5saxpyPKfPffi' for 'sm_75'
ptxas info    : Function properties for _Z5saxpyPKfPffi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 376 bytes cmem[0]
```

Ten registers per thread and no shared memory feed post 10's occupancy model for `sm_75`: 32 warp slots, at most 16 blocks and 65,536 registers per streaming multiprocessor (SM), registers allocated per warp in units of 256 from four quarters of 16,384 that each hold whole warps. For blocks of 256 threads, 8 warps:

- registers: $\lceil 10 \times 32 / 256 \rceil \times 256 = 512$ per warp, so a quarter holds $16{,}384 / 512 = 32$ warps and the SM 128, room for 16 blocks;
- warp slots: $32 / 8 = 4$ blocks; the block limit: 16; shared memory: no limit;
- so 4 blocks, 32 warps, a **theoretical occupancy** of $32 / 32 = 100\%$, set by the warp slots.

For blocks of 32 threads, one warp each, the 16-block limit binds first: 16 warps, 50%. The CUDA runtime's occupancy application programming interface (API) does the same arithmetic on the compiled kernel: `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, saxpy, blockSize, 0)` returns the blocks per SM, which the Programming Guide's section "Occupancy Calculator" turns into warps and a percentage, and `cudaOccupancyMaxPotentialBlockSize` returns the block size with the highest occupancy. Neither needs performance counters, and both give a ceiling on the warps resident, not a speed.

## 5. Measured: occupancy from the runtime, and bandwidth

`occupancy_demo.cu` prints the runtime's theoretical occupancy for six block sizes and times SAXPY at each, the fastest of 5 launches after half a second of untimed ones, as 12 bytes per element (`x` and `y` read, `y` written) over the time. It measures the SM clock around the timed rows, as `clock64()` cycles over CUDA-event time, and checks every element on the host. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), 14 SMs, 32 warp slots per SM
saxpy: 10 registers per thread, 0 bytes of static shared memory, 0 bytes of local memory
saxpy over N = 16777216 floats; theoretical occupancy from the runtime;
time the fastest of 5 launches; GB/s = 12 bytes per element over the time; peak 160 GB/s
block  blocks/SM  warps/SM  theoretical  time (ms)    GB/s  % of peak
   32         16        16          50%      1.874   107.4        67%
   64         16        32         100%      1.386   145.3        91%
  128          8        32         100%      1.387   145.2        91%
  256          4        32         100%      1.388   145.1        91%
  512          2        32         100%      1.386   145.2        91%
 1024          1        32         100%      1.385   145.3        91%
cudaOccupancyMaxPotentialBlockSize: block 1024, at least 14 blocks to fill the GPU
SM clock: 1740 MHz before the timed launches, 1741 MHz after
check: 16777216 of 16777216 elements equal 4.0  ok
```

The theoretical column equals the model's, 50% at 32 threads per block and 100% from 64 up. The times are measured at the 1,740 to 1,741 MHz SM clock the program printed; under a sustained kernel the card sits at its software power cap (about 35 W), and the reader's card will give other digits.

From 64 to 1,024 threads per block the kernel moves 145.1 to 145.3 GB/s, 91% of the peak, as post 02's vector addition did. At 32 threads per block, with half the warp slots empty, it falls to 107.4 GB/s, 26% less. Post 10's latency-hiding argument predicts a loss from the empty warp slots, but this run alone does not separate that from the cost of launching and retiring twice as many one-warp blocks.

**Achieved occupancy**, the warps actually resident averaged over the run, can only be lower than theoretical, and it is a counter reading: Nsight Compute reports it as `sm__warps_active.avg.pct_of_peak_sustained_active`, which this machine denies (section 7). The occupancy measured here is the runtime's theoretical figure for the compiled kernel, beside the bandwidth the kernel reached.

![Left: occupancy against threads per block from 32 to 1,024. The model and the runtime both give 50% at 32 and 100% from 64 up. Right: measured bandwidth, 107.4 GB/s at 32 threads per block and 145.1 to 145.3 GB/s from 64 to 1,024, against the 160 GB/s peak.](diagrams/03-occupancy-measured.svg)

*Theoretical occupancy limits only the one-warp blocks, and that is the only launch the bandwidth notices.*

## 6. Reading the measurement against the roofline

SAXPY does one fused multiply-add per element, 2 FLOP, and moves 12 bytes: $I = 2 / 12 \approx 0.167$ FLOP per byte, 117 times below the ridge of 19.5. Its roofline ceiling is $I \beta = 160 / 6 \approx 26.7$ GFLOP/s, under 1% of the compute roof. The measured 145.2 GB/s is $145.2 / 6 \approx 24.2$ GFLOP/s, 91% of that ceiling: memory bound, close to its roof, with nothing to gain from its arithmetic. The lever is fewer bytes, by fusing it with its neighbors ([post 14](../14-kernel-fusion/index.md)). A kernel at 91% of its roof has little left to win; one at 26%, like post 06's naive transpose (30% of a copy kernel), has a limiter worth a profiler's time.

## 7. Two profilers: a timeline and a microscope

Two profilers answer two questions (Nsight Systems 2023.3.3 and Nsight Compute 2023.3.0 are installed here).

**Nsight Systems (`nsys`) is the timeline.** It records the whole program across time: every CUDA API call, kernel and copy, on which stream, and with `--stats=true` summary tables of time per kernel and per copy. It answers which kernel or copy dominates and whether copies overlap kernels, the question of [post 12](../12-cuda-streams/index.md). On this laptop it did not start:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=true -o occ_timeline ./occupancy_demo --check
```

```text
ReflexStatsTraceLoggingProvider provider requires administrator privileges
```

The ETW (Event Tracing for Windows) provider it enables needs administrator rights. `nsys` exited with status 1, wrote no report, and left the program it had started suspended. No timeline was measured for this post.

**Nsight Compute (`ncu`) is the microscope.** It replays one kernel many times under hardware performance counters and reports that kernel alone. For one launch of the demo's kernel, every section (the output below leaves out the first line, which names the program's path):

```bash
ncu --set full -k saxpy -c 1 -o occ_kernel ./occupancy_demo --check
```

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), 14 SMs, 32 warp slots per SM
saxpy: 10 registers per thread, 0 bytes of static shared memory, 0 bytes of local memory
check: 16777216 of 16777216 elements equal 4.0  ok
==PROF== Disconnected from process 6092
==WARNING== No kernels were profiled.
```

`ERR_NVGPUCTRPERM` means the driver denies the counters to users who are not administrators. NVIDIA's page for the error gives the Windows fix: in the NVIDIA Control Panel, under Desktop, check "Enable Developer Settings", then under Developer, Manage GPU Performance Counters, select "Allow access to the GPU performance counter to all users", which itself needs the Control Panel run as administrator; or run `ncu` from an elevated shell. Neither was done here, so nothing below is a measurement. By its documentation and installed section files, `ncu` reports:

- **GPU Speed Of Light Throughput:** "Memory Throughput" and "Compute (SM) Throughput", each "the achieved percentage of utilization with respect to the theoretical maximum"; SAXPY should show memory far above compute.
- **Occupancy:** the block limits of section 4 and theoretical occupancy, beside achieved occupancy.
- **Warp State Statistics:** the cycles warps spend stalled per issued instruction, by reason; waiting on a global load is "long scoreboard".
- **GPU Speed Of Light Roofline Chart:** the kernel on the card's roofline, the measured form of post 10's chart (in the `full`, `detailed` and `roofline` sets; the default set has the first two sections).

So `nsys` goes first, over the whole program, to find the kernel that matters, and `ncu` second, on that kernel, to find why.

## 8. Make it run

The three programs need an NVIDIA GPU and the CUDA Toolkit, and build from the series root with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 -Xptxas=-v posts/11-profiling-debugging/snippets/occupancy_demo.cu -o occupancy_demo && ./occupancy_demo
nvcc -O3 -arch=sm_75 -lineinfo posts/11-profiling-debugging/snippets/missing_guard.cu -o missing_guard
nvcc -O3 -arch=sm_75 -lineinfo posts/11-profiling-debugging/snippets/shared_race.cu -o shared_race
compute-sanitizer --tool memcheck --show-backtrace no ./missing_guard --no-guard
compute-sanitizer --tool racecheck --show-backtrace no ./shared_race --race
compute-sanitizer --tool synccheck --show-backtrace no --print-limit 2 ./shared_race --divergent-barrier
compute-sanitizer ./occupancy_demo --check
```

On Windows the programs are `.exe` files, and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both. Each builds in 3 to 5 seconds; `occupancy_demo` runs in about a second, and its `--check` flag runs one launch and the host check, for the sanitizer. Without a flag, `missing_guard` and `shared_race` run their correct versions and exit 0. `nsys` is found under the Nsight Systems install folder rather than on `PATH`.

## 9. What can go wrong?

Each of these happened on the author's GTX 1650 while this post was measured.

- **Timing a card that is not at its clock.** A run of `occupancy_demo` while the laptop's power source kept changing printed an SM clock of 211 MHz before and 248 MHz after, and 74.3 GB/s at 256 threads per block; the printed clock is how such a run is caught and discarded.
- **Trusting a clean exit.** `missing_guard --no-guard` printed `6 of 6` and exited 0, and the divergent barrier gave right sums in all 64 blocks.
- **Expecting `__launch_bounds__(256)` to cap registers.** In a test copy of the demo, SAXPY with it still used 10 registers: with one argument the compiler only keeps a 256-thread block possible, and the Programming Guide says a kernel "will fail to launch if it is executed with more threads per block than its launch bound". A 512-thread launch of that copy failed with "invalid argument". Capping registers takes the second argument, `minBlocksPerMultiprocessor`, or `-maxrregcount`.
- **Leaving a profiler's target behind.** Every failed `nsys` run left a suspended copy of the program; it has to be found and stopped before the next timing run.

---

## Common pitfalls

1. **Profiling before the sanitizer is clean.** A kernel that touches the wrong memory can post a fast, meaningless time; run memcheck, and racecheck on any shared-memory kernel, first.
2. **Reading theoretical occupancy as achieved.** The register arithmetic and the occupancy API give a ceiling on the warps resident; the warps actually resident are a counter reading, Nsight Compute's achieved occupancy.
3. **Choosing a block size by habit.** At 32 threads per block the 16-block limit halved SAXPY's warps and it lost 26% of its bandwidth (the run does not say how much of that the empty slots cost); from 64 to 1,024 every size gave 100% and the same 145 GB/s.
4. **Running `ncu --set full` over a whole program.** It replays every kernel under counters; find the kernel with `nsys`, then point `ncu` at it with `-k` and `-c`.
5. **Taking a sanitizer's silence as proof.** memcheck misses an overflow into a neighboring allocation without `--padding`, and synccheck misses a divergent barrier whose other threads exit.

---

## Further reading

- NVIDIA, *Compute Sanitizer Documentation*, version 2023.3 (with CUDA 12.3), consulted October 2026. The four tools, their reports, padding and the print limit.
- NVIDIA, *CUDA Toolkit 12.0 Release Notes*, consulted October 2026. The removal of CUDA-MEMCHECK.
- NVIDIA, *Nsight Compute Documentation*, version 2023.3, consulted October 2026. The sections, the metrics and the replay.
- NVIDIA, *Nsight Systems Documentation*, version 2023.3, consulted October 2026. The timeline and its statistics.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, sections "Occupancy Calculator" and "Launch Bounds", consulted October 2026 (CUDA 12.3 edition).
- NVIDIA, "ERR_NVGPUCTRPERM: Permission issue with Performance Counters", NVIDIA Developer, consulted October 2026. How to allow access to the counters.
- NVIDIA, *CUDA C++ Best Practices Guide*, consulted October 2026. Measuring before optimizing.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 12 - CUDA streams: overlapping copy and compute for free throughput](../12-cuda-streams/index.md):** the overlap of copies and kernels that a timeline such as Nsight Systems shows.
- **[Post 14 - Kernel fusion: the fastest kernel is the one never launched](../14-kernel-fusion/index.md):** the lever for a memory-bound kernel like SAXPY, fewer bytes per result.
