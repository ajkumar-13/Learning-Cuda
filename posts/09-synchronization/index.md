# 09 - When threads share data: barriers, atomics, and the memory model

> **TL;DR.** A plain `counter += 1` from many threads is a read, an add and a write that other threads can slip between: on the author's GTX 1650 it kept 548 of 6,553,600 increments, and `atomicAdd` kept all of them. `__syncthreads()` orders a block only when every thread of the block reaches it; inside a warp the `_sync` primitives order their own lanes, and `__syncwarp()` orders lanes that talk through shared memory. An atomic guarantees that no update is lost and nothing about when other writes become visible, which is the job of a **memory fence** at a **scope** (block, device or system), each wider scope costing more on this card. No barrier spans the grid inside an ordinary kernel; a new launch is the grid-wide barrier.
>
> **Prerequisites:** [post 05](../05-histogram/index.md), [post 08](../08-parallel-scan/index.md).
> **Safe to skip?** Skip it if the reader can already say what `__syncthreads()`, `__syncwarp()`, a `_sync` shuffle, an atomic and a fence each guarantee, which of them orders memory, and why a kernel cannot wait for all of its blocks.
>
> **After reading, you will be able to:**
>
> - Explain the lost-update race of a plain read-modify-write, and use the race model to count what each schedule loses.
> - Place `__syncthreads()` where every thread of the block reaches it, and say why a barrier inside divergent code is undefined.
> - Order the lanes of one warp: a `_sync` primitive with a mask derived from the program's logic, or `__syncwarp()` between shared-memory reads and writes.
> - Use `atomicAdd` and `atomicCAS`, and separate what an atomic guarantees (no lost update) from what it does not (the order of other writes).
> - Publish data to other threads with a fence or an acquire-release atomic at the narrowest scope that reaches them: block, device, or system.
> - Explain why no barrier spans the grid inside an ordinary kernel, and why a new launch is the grid-wide barrier.

![Top: threads 0 and 1 both read 7 and both write 8, so the counter ends at 8, not 9, the second write red. Bottom: with atomicAdd, thread 1 waits while thread 0 adds, and the counter reaches 9.](diagrams/01-race-condition.svg)

*Both threads read 7 before either writes, so one increment is lost; an atomic makes the second thread wait, and both count.*

---

## 1. The question: an update that vanishes

Part II leaned on synchronization rules stated in passing: the trees of [post 04](../04-reduction/index.md) and [post 08](../08-parallel-scan/index.md) put a `__syncthreads()` between levels, and the histogram of [post 05](../05-histogram/index.md) used `atomicAdd`. This post asks what exactly `__syncthreads()`, `__syncwarp()`, atomics, fences and scopes each guarantee, and what none of them guarantees.

Start with the bug. `counter += 1` is one line of source, but a thread runs it as three steps: READ the value into a private register, add one, WRITE the register back. If two threads READ before either WRITEs, both compute the same value and the second WRITE lands on the first: two increments, one counted. This is a **race condition**, and its symptom is a **lost update**, drawn at the top for a counter holding 7.

The central processing unit (CPU) model of this post, `snippets/race_model.py`, replays the read, add and write of `T` = 1,024 threads doing `K` = 100 increments each under fixed schedules and prints that two-thread trace and this table:

```text
  final counter by schedule (higher is more correct):
    schedule                         final      lost  fraction lost
    serial (race-free order)       102,400         0        0.000
    warp-lockstep (32 collide)       3,200    99,200        0.969
    full-lockstep (all collide)        100   102,300        0.999
    atomicAdd (indivisible)        102,400         0        0.000
```

These counts are derived by the model. When the 32 lanes of each warp read, add and write together and the warps take turns, each warp step adds 1, so 3,200 of the 102,400 increments survive; when all 1,024 threads collide on every step, 100 do. The atomic row has the same arrival order, with each lane's three steps made one.

The graphics processing unit (GPU) does the same. Part 1 of [`snippets/sync.cu`](snippets/sync.cu) launches 65,536 threads that each add one to a global counter 100 times, with a plain read-modify-write and with `atomicAdd`, against the host's 6,553,600:

```cuda
__global__ void raceKernel(volatile int* c) {
    for (int k = 0; k < ITERS; ++k)
        *c = *c + 1;                      // load, add, store: three steps, not one
}
```

Without the `volatile`, the compiler merges the loop into one load, one add of 100 and one store (`cuobjdump -sass` shows `IADD3 R0, R0, 0x64`). One run on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, printed:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5)
1. one counter, 65536 threads x 100 increments
  expected            = 6553600
  plain (*c = *c + 1) = 548       updates lost
  atomicAdd           = 6553600   ok
```

The plain count changes from run to run (eight runs printed 504 to 557), because which reads fall between which writes depends on timing no program controls; it lies near the model's full-lockstep 100 for this launch, not its warp-lockstep 204,800, as fits a card with up to 14,336 threads resident at once, 1,024 on each of its 14 streaming multiprocessors (SMs, [post 01](../01-introduction-to-cuda/index.md)). The atomic count was right on every run.

## 2. The block barrier: `__syncthreads()`

Within a block, `__syncthreads()` is a **barrier**: the CUDA C++ Programming Guide (section "Synchronization Functions") says it "waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to `__syncthreads()` are visible to all threads in the block". In a tree, the wait keeps step `s + 1` from starting early and the visibility makes step `s`'s writes readable.

The same section gives the rule: the barrier "is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, otherwise the code execution is likely to hang or produce unintended side effects". In this fragment only the first 32 threads reach it:

```cuda
if (threadIdx.x < 32) {
    // ... some work ...
    __syncthreads();   // WRONG: only 32 threads of the block reach this barrier
}
```

What happens is undefined. From compute capability 7.0 on, the Guide's section "Independent Thread Scheduling" says, the barrier "will not succeed until reached by all non-exited threads in the block", so the outcome depends on the other threads: if they exit, the waiting threads go on; if they reach a later `__syncthreads()`, that arrival counts and the block passes with its phases paired wrongly; if they stay alive waiting for the first 32, the block hangs. Two such variants ran to completion on the author's GTX 1650, which is the danger: the bug passes a test and hangs when the code around it changes.

![Left: of eight threads, threads 0 to 3 take a branch and wait, red-edged, at a purple barrier inside it, while threads 4 to 7 skip it and never arrive: undefined. Right: the barrier after the branch, where all eight arrive and continue together.](diagrams/02-barrier.svg)

*A barrier inside a branch waits for threads that never come; after the branch, every thread arrives.*

The fix moves the barrier out of the branch:

```cuda
if (threadIdx.x < 32) { /* some work */ }
__syncthreads();       // every thread of the block reaches this barrier
```

The same rule covers a loop whose trip count differs between threads.

## 3. Inside one warp: `_sync` primitives and `__syncwarp()`

Older code assumed that a warp's 32 lanes run in lockstep: Harris's fifth reduction kernel (post 04) finished its last 32 values in `volatile` shared memory with no barrier. Since compute capability 7.0, **independent thread scheduling** lets the lanes of a warp run apart, and the Guide states that "assumptions that code is executed in lockstep or that reads/writes from separate threads are visible across a warp without synchronization are invalid". The GTX 1650 has it. Two tools replace the assumption, with different jobs:

- **The `_sync` primitives** (`__shfl_down_sync`, `__ballot_sync` and the rest) take a mask of the lanes that take part. Lin and Grover ("Using CUDA Warp-Level Primitives", NVIDIA Technical Blog, 15 January 2018) write that "these primitives first synchronize the threads if they are not already synchronized", and the Guide requires every lane in the mask to execute the same intrinsic with the same mask. A shuffle passes data between registers inside the call, so consecutive shuffles need no `__syncwarp()`, and post 04's `warpReduce` is correct as written. They "do not imply a memory barrier".
- **`__syncwarp(mask)`** waits for every lane in the mask and "guarantees memory ordering among threads participating in the barrier": the tool for lanes that exchange data through memory.

Part 2 of `sync.cu` sums 256 values per block in shared memory, with `__syncthreads()` down to 64 values, and does the last six steps in one warp four ways:

```cuda
    if (tid < 32) {
        if (MODE == PLAIN) {              // no barrier: assumes the warp runs in lockstep
            for (int off = 32; off > 0; off >>= 1)
                s[tid] += s[tid + off];
        } else if (MODE == VOLATILE) {    // the pre-2018 idiom: volatile, still no barrier
            volatile int* v = s;
            for (int off = 32; off > 0; off >>= 1)
                v[tid] += v[tid + off];
        } else if (MODE == SYNCWARP) {    // every read before any write, every write before the next read
            int v = s[tid];
            for (int off = 32; off > 0; off >>= 1) {
                v += s[tid + off];
                __syncwarp();
                s[tid] = v;
                __syncwarp();
            }
        } else {                          // SHUFFLE: registers only; the _sync shuffle orders its lanes
            int v = s[tid] + s[tid + 32];
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) s[0] = v;
        }
    }
```

The `__syncwarp()` version separates each step's read from its write, as Lin and Grover's corrected listing does; with one `__syncwarp()` per step, a lane's read of `s[tid + off]` and its neighbor's write of it sit between the same two barriers, still a race. Over 16,384 blocks and 20 launches:

```text
2. the last six steps of a 256-thread block sum, 16384 blocks x 20 launches
  plain shared memory, no barrier    wrong block sums:   327680 of 327680  (breaks the rule)
  volatile, no barrier               wrong block sums:        0 of 327680  (breaks the rule)
  __syncwarp between read and write  wrong block sums:        0 of 327680  ok
  __shfl_down_sync                   wrong block sums:        0 of 327680  ok
```

The plain tail is wrong in every block: with nothing ordering the warp, the compiler loads all seven values before its one store (`cuobjdump -sass` shows seven `LDS`, then one `STS`). The `volatile` tail was right here and still breaks the rule; Lin and Grover call such code unsafe because it depends on behavior that "may change from one hardware architecture to another, from one CUDA toolkit release to another ... or even from one run-time execution to another". A version of `sync.cu` cut to 64 blocks and one launch, run under `compute-sanitizer --tool racecheck`, reported races in the plain and `volatile` tails and none in the other two.

The mask comes from the program's logic, computed before any branch: `__ballot_sync(0xffffffff, threadIdx.x < n)` names the lanes that take part when only the first `n` do. `__activemask()` is no substitute: it returns only the lanes that happen to be converged at the call, which need not be the lanes the branch admits. The full mask is right only when every lane reaches the call.

## 4. Atomics: correct counts, not ordered writes

An **atomic** performs a read-modify-write on one 32-bit or 64-bit word "in one atomic transaction" (the Guide, section "Atomic Functions"), so no thread can slip between the read and the write. Besides `atomicAdd`, the most general one is **compare-and-swap**, `atomicCAS(addr, expected, desired)`: it stores `desired` only if `*addr` equals `expected`, and returns the old value. Any atomic can be built from it in a retry loop, such as a `float` maximum, which `atomicMax` (integers only) does not offer. As a fragment, with `f` the update:

```cuda
// build any atomic read-modify-write out of atomicCAS: retry until it sticks
int old = *addr, assumed;
do {
    assumed = old;
    int want = f(assumed);                 // the update to install
    old = atomicCAS(addr, assumed, want);  // install it only if *addr is unchanged
} while (assumed != old);                  // another thread got there first: retry
```

An atomic guarantees **atomicity**, not **ordering**: the Guide states that the atomic functions "have ordering `cuda::memory_order_relaxed`". If thread 0 writes `data[7] = 42` and then runs `atomicAdd(&flag, 1)`, a thread of another block that sees the new `flag` may still read the old `data[7]`. Atomics have scopes too: `atomicAdd` is atomic across the device, `atomicAdd_block` within a block, `atomicAdd_system` also against the host and peer GPUs (compute capability 6.0 and later).

Atomics on one address also take turns: post 05 measured global atomics counting 16,777,216 values at 13.5 GB/s on random data and 6.6 GB/s when every value hit one bin, about twice as slow. The compiler can pay part of the price: `cuobjdump -sass` shows part 1's `atomicKernel` counting the warp's active lanes and issuing one `RED.E.ADD` per warp per iteration, so 6,553,600 increments reach memory as 204,800 atomic adds.

## 5. Ordering writes: fences, release and acquire

The Guide's section "Memory Fence Functions" starts from a weakly ordered model: the order in which a thread writes "is not necessarily the order in which the data is observed being written by another CUDA or host thread". A **memory fence** restores an order: after `__threadfence()`, no thread of the device observes a write the calling thread made after the fence as occurring before one it made before the fence. A fence waits for no other thread, and fences "do not, by themselves, ensure that these memory operations are visible to other threads", which is why the Guide's example reads the published data through a `volatile` pointer.

Publishing has two halves: the writer stores the data, fences and sets a flag (a **release**); the reader sees the flag, fences and reads the data (an **acquire**). Part 3 of `sync.cu` is the Guide's example: each block sums 256 values, thread 0 stores the partial sum, fences, and counts the block done with `atomicInc`; the block that receives `gridDim.x - 1` is the last, and it adds the partial sums:

```cuda
    if (tid == 0) {
        partial[blockIdx.x] = s[0];                      // publish the data
        __threadfence();                                 // ... before the flag below
        unsigned int value = atomicInc(&count, gridDim.x);   // the flag: one more block done
        isLastBlockDone = (value == gridDim.x - 1);
    }
    __syncthreads();                                     // every thread reads isLastBlockDone
```

Without the fence the counter could reach `gridDim.x - 1` before a partial sum is written. The reader's half is lighter, as in the Guide: no second fence, the partial sums read through a `volatile` pointer. Every total is checked against the host:

```text
3. the last-block sum of 4194304 ints in 16384 blocks, 20 launches
  expected 12582918, last launch 12582918, wrong totals: 0 of 20  ok
```

No block waits for another, so the pattern is safe under any schedule; the figure draws it with and without fences.

![A writer in block 0 and a reader in block 1. Top, no fences: the flag is seen before the data, and the reader reads stale data 0, in red. Bottom: data, purple fence, flag; the reader sees the flag, fences, reads 42.](diagrams/03-fence-acquire-release.svg)

*A fence before the flag on the writer's side and after it on the reader's: a reader that sees the flag then sees the data.*

The modern form puts the order on the atomic. CUDA 12.3 ships libcu++ 2.2.0, whose `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>` offers `fetch_add(1, cuda::memory_order_acq_rel)`, a release and an acquire in one call, in the terms of the C++ memory model (Boehm and Adve, 2008). Part 3 written that way was right on 200 of 200 launches here; its machine code puts a `MEMBAR.ALL.GPU` before the atomic (release) and a `CCTL.IVALL`, which invalidates the SM's L1 cache, after it (acquire).

## 6. Scopes: block, device, system

A fence has a **scope**, the threads whose view of the order it fixes:

- `__threadfence_block()` orders the calling thread's writes as seen by the threads of its **block**;
- `__threadfence()` as seen by every thread of the **device**;
- `__threadfence_system()` also by host threads and peer devices, the whole **system**, which a kernel needs when it talks to the CPU through mapped host memory.

The Guide's rule: two threads of one block need only `__threadfence_block()`, threads of different blocks `__threadfence()`, threads on different devices `__threadfence_system()`. For `sm_75` the three compile to `MEMBAR.SC.CTA`, `MEMBAR.SC.GPU` and `MEMBAR.SC.SYS`, the last two followed by an L1 invalidation. Part 4 of `sync.cu` times a loop in which each of 14,336 threads stores a value and fences, 1,000 times, the fastest of 5 launches after 400 untimed ones, and prints the SM clock it measured:

```text
4. store then fence, 14336 threads x 1000 iterations, fastest of 5 launches
  scope                     ns per iteration
  no fence                        39
  __threadfence_block()          166
  __threadfence()                328
  __threadfence_system()        2919
  SM clock: 1754 MHz before the timed launches, 1754 MHz after
```

These numbers are measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026; the reader's card will differ, and the order of the rows is the lesson. Each scope cost more than the one inside it: 2.0 times from block to device, 8.9 times from device to system. The printed SM clock matters: the loop is latency bound, and the same harness run while the laptop was on battery or its charger kept disconnecting measured about 250 MHz and every row four to seven times longer (269, 1,180, 1,800 and 7,778 ns), so fence costs are compared only within one run whose printed clock is near the card's 1,740 MHz boost. The rule: the narrowest scope that reaches every thread that must see the write.

![Three scopes with their fence, atomic and reach: block, 166 ns per iteration; device, 328 ns; system, which also reaches host threads and peer GPUs, 2,919 ns; no fence, 39 ns.](diagrams/04-scope-ladder.svg)

*Each wider scope orders the writes for more threads and, on this card, takes longer.*

## 7. There is no barrier across the grid inside a kernel

Blocks are independent (post 01): they run in any order, only as many at a time as the SMs hold, so a block that waits for another may wait forever. Sorensen, Donaldson, Batty, Gopalakrishnan and Rakamarić ("Portable Inter-Workgroup Barrier Synchronisation for GPUs", OOPSLA 2016) show that a software barrier across blocks can deadlock for this reason, and make one safe by first discovering how many blocks can be resident.

The barrier the model provides is the kernel boundary. The Guide's section "Grid Synchronization" says that before Cooperative Groups "the CUDA programming model only allowed synchronization between thread blocks at a kernel completion boundary": every block of a launch finishes before the next launch in the stream begins. Post 08's scan puts a launch between its phases because block $b$ needs the totals of the blocks before it; post 04's reduction needed one launch, because `atomicAdd` may take the block totals in any order.

Cooperative Groups' `grid.sync()` is a grid-wide barrier, but only in a kernel launched with `cudaLaunchCooperativeKernel` and no more blocks than fit at once; this series names it and stops.

## 8. Make it run

The CPU model needs only the standard library. From the series root:

```bash
uv run python posts/09-synchronization/snippets/race_model.py
```

The kernel needs an NVIDIA GPU and the CUDA Toolkit, and builds from the series root with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 posts/09-synchronization/snippets/sync.cu -o sync && ./sync
```

On Windows the program is `sync.exe`, and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both. It builds in about 15 seconds, prints the four parts quoted above, and exits with status 1 if a row that keeps the rules disagrees with the host or a CUDA call fails. [Post 11](../11-profiling-debugging/index.md) covers `compute-sanitizer`, whose racecheck flagged the `volatile` tail here although its sums were right.

## 9. What can go wrong?

Each mistake below was reproduced on the author's GTX 1650 with a one-line change to `sync.cu`.

- **Dropping the barrier after `isLastBlockDone`.** Threads of the last block read the flag before thread 0 wrote it, and 148 of 200 totals were wrong.
- **Not resetting the counter.** Without `count = 0`, the next launch counts on from the last, and 18 of 20 totals were wrong.
- **Dropping the fence before `atomicInc`.** All 500 totals were still right on this card; the ordering guarantee is gone, so the next card or compiler may differ.
- **One `__syncwarp()` per step.** Every block sum was right here, and the read and the neighbor's write still race between the same two barriers.

---

## Common pitfalls

1. **A plain shared counter.** `counter += 1` from many threads loses updates; use `atomicAdd`, or privatize as post 05 does.
2. **`__syncthreads()` where only some threads go.** A barrier in a branch or loop that not every thread takes is undefined: it can hang, or pass with its phases paired wrongly.
3. **Trusting a warp to run in lockstep.** Put `__syncwarp()` between each shared-memory read and write, and compute a `_sync` primitive's mask from the program's logic, never from `__activemask()`.
4. **Reading atomicity as ordering.** An atomic is relaxed; publish data with a fence before the flag and after it, or with `cuda::atomic_ref` and acquire-release order.
5. **Waiting for another block inside a kernel.** Blocks may not be resident together; split the phases across launches or let the last block do the work.

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, sections "Synchronization Functions", "Memory Fence Functions", "Atomic Functions", "Warp Vote Functions", "Warp Shuffle Functions" and "Grid Synchronization", consulted October 2026 (CUDA 12.3 edition).
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Independent Thread Scheduling", consulted October 2026. Why lockstep warp code broke at compute capability 7.0.
- Lin, Y. and Grover, V., "Using CUDA Warp-Level Primitives" (NVIDIA Technical Blog, 15 January 2018). The `_sync` primitives, the mask, `__activemask()` and `__syncwarp()`.
- Sorensen, T., Donaldson, A. F., Batty, M., Gopalakrishnan, G. and Rakamarić, Z., "Portable Inter-Workgroup Barrier Synchronisation for GPUs" (OOPSLA, 2016). Why a barrier across blocks can deadlock.
- Boehm, H.-J. and Adve, S. V., "Foundations of the C++ Concurrency Memory Model" (PLDI, 2008). The model whose orders `cuda::atomic_ref` uses.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy](../10-roofline/index.md):** with correctness settled, how fast a correct kernel can go and which resource caps it.
- **[Post 11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy](../11-profiling-debugging/index.md):** how `compute-sanitizer` finds the races and barrier mistakes of this post before profiling begins.
