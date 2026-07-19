# 09 · When threads share data — barriers, atomics, and the memory model

> **TL;DR.** When many threads touch one location, correctness stops being automatic: a plain `counter += 1` from several threads is a **race condition** where each reads the old value before any writes the new one, so most increments are silently lost. `__syncthreads()` is a **block barrier** that makes every thread wait until all have arrived, but put it inside divergent control flow and the block deadlocks. An **atomic** like `atomicAdd` makes a read-modify-write indivisible so no update is lost, yet atomicity is not ordering: it says nothing about when other threads *see* a different write, which is the job of a **memory fence** (`__threadfence`) and a **thread scope** (block, device, or system). And there is no global barrier inside a running kernel: separate kernel launches are the only guaranteed grid-wide sync.
>
> **After reading this you will be able to:**
> - Explain the lost-update race and simulate how much a non-atomic increment loses.
> - Place `__syncthreads()` correctly, and say why a barrier in divergent code deadlocks a block.
> - Use `atomicAdd` and `atomicCAS`, and separate what an atomic guarantees (no lost update) from what it does not (ordering).
> - Reason about `__threadfence` and block/device/system scopes with an acquire/release intuition, and know why only a new kernel launch is a global barrier.

![Two threads each read a shared counter holding 7, both add one, and both write 8; the second write lands on top of the first, so one increment shown in red is lost and the counter reads 8 instead of the correct 9.](diagrams/01-race-condition.svg)
*Both threads read 7 before either writes, so one update is overwritten. The lost increment, in red, is the race.*

---

## 1. The race: an update that vanishes

Every pattern in Part II shared data between threads, and each leaned on a synchronization rule stated in passing: reduction and scan put a `__syncthreads()` between tree steps ([Post 04](../04-reduction/index.md), [Post 08](../08-parallel-scan/index.md)), and the histogram reached for `atomicAdd` to keep a shared count correct ([Post 05](../05-histogram/index.md)). This post gathers those rules into one place and says exactly what each one guarantees, because the next time you write a kernel that shares state, guessing is a bug.

Start with the bug itself. Suppose a thousand threads each want to add one to a shared `counter`. In source that is one line, `counter += 1`, but the hardware runs it as three steps: READ the value from memory into a private register, add one, WRITE the register back. Nothing makes those three steps indivisible. If two threads READ before either WRITEs, both see the same old value, both compute the same new value, and the second WRITE lands on top of the first. Two increments, one counted. This is a **race condition**, and its symptom here is a **lost update**.

A standard-library model makes the loss concrete. It replays the read-modify-write of `T` threads doing `K` increments each under a few fixed schedules, from a lucky race-free order to the worst case where every thread collides on every step, and compares against the indivisible atomic version. It is deterministic (fixed constructions, no clocks or randomness), needs no GPU, and lives at [`snippets/race_model.py`](snippets/race_model.py). Running it prints:

```text
race model
  threads (T)               = 1024
  increments per thread (K) = 100
  intended final counter    = 102,400   (T x K, every increment counted)
  warp size                 = 32

  trace  (4 threads, 1 increment each, fully interleaved):
    READ : all 4 threads read counter = 0
    INC  : all compute 0 + 1 = 1
    WRITE: all write 1 -> counter = 1     (3 of 4 updates lost)

  final counter by schedule (higher is more correct):
    schedule                         final      lost  fraction lost
    serial (race-free order)       102,400         0        0.000
    warp-lockstep (32 collide)       3,200    99,200        0.969
    full-lockstep (all collide)        100   102,300        0.999
    atomicAdd (indivisible)        102,400         0        0.000
```

Read the two extremes. Under the worst-case schedule the 102,400 intended increments collapse to 100, losing 99.9% of the work, because every step overwrites all but one thread's update. The atomic version counts all 102,400, every time. A real race lands between these: the warp-lockstep row (32 lanes of a warp collide, but warps run one after another) still keeps only 3,200, throwing away 97%. The point is not the exact figure, which depends on timing you do not control; it is that a non-atomic shared increment is wrong, and unpredictably so.

The companion kernel shows the same failure on real hardware. [`snippets/sync.cu`](snippets/sync.cu) launches 65,536 threads that each increment one global counter 100 times, once with a plain `*c += 1` and once with `atomicAdd`, and checks both against the host reference of 6,553,600. Only the atomic run matches; the racy run reports a smaller, run-to-run-varying number. Build and run the kernel and the CPU model with:

```bash
nvcc -O3 -arch=sm_75 snippets/sync.cu -o sync && ./sync
python snippets/race_model.py
```

## 2. The block barrier: `__syncthreads()`

The race in Section 1 is about the *atomicity* of one location. The other half of synchronization is *ordering*: making one thread wait until others have finished a phase. Within a block, that is `__syncthreads()`, a **barrier**. Every thread that reaches it stops until every thread in the block has reached it; only then do any proceed. It is what makes the reduction and scan trees correct: each step writes shared memory that the next step reads across threads, and the barrier guarantees all of step `s`'s writes are done and visible before step `s+1` reads them.

![All threads of a block advance toward a horizontal __syncthreads line; the threads that arrive first wait in gray at the line while stragglers catch up, and only when the last thread arrives do all cross together in green.](diagrams/02-barrier.svg)
*A barrier releases no thread until every thread in the block has arrived. The slowest thread sets the pace.*

The guarantee has a sharp edge: `__syncthreads()` must be reached by *every* thread in the block, or by none. Put it inside divergent control flow, where only some threads take the branch, and the threads that skip it never arrive, so the threads that did arrive wait for an event that cannot happen. The result is undefined, and typically a hang:

```cuda
if (threadIdx.x < 32) {
    // ... some work ...
    __syncthreads();   // WRONG: only 32 threads reach this barrier,
}                      // the rest never do, so these 32 wait forever
```

The fix is to lift the barrier out of the branch, so the whole block reaches the same one:

```cuda
if (threadIdx.x < 32) { /* some work */ }
__syncthreads();       // every thread reaches this barrier
```

The same rule bans a `__syncthreads()` inside a loop whose trip count differs across threads, and after an early `return` that only some threads take. A barrier is a property of the whole block; place it only where the whole block agrees to be.

## 3. Warps are not free barriers: `__syncwarp()` and the active mask

A block barrier is often more than you need. The 32 lanes of a **warp** run together, which used to let programmers skip the barrier and rely on lanes being implicitly in step: the old *warp-synchronous* trick, common in pre-2018 reduction code that dropped the sync for the last 32 elements. On modern GPUs (Volta and later) that assumption is unsafe. **Independent thread scheduling** lets lanes of one warp diverge and reconverge on their own, so nothing guarantees that lane 5 has executed its write before lane 6 reads it, even inside a single warp.

The correct tool is `__syncwarp()`, a barrier scoped to a warp, together with the **active mask** it takes. Not every lane is always active (some may have branched away), so warp-level primitives take a 32-bit mask naming which lanes participate. `__activemask()` returns the lanes currently active, and `__syncwarp(mask)` synchronizes exactly those:

```cuda
// warp-level steps of a shuffle reduction: order the participating lanes
val += __shfl_down_sync(0xffffffff, val, 16);
__syncwarp();                    // not optional on Volta and later
val += __shfl_down_sync(0xffffffff, val, 8);
```

The mask `0xffffffff` says all 32 lanes take part, correct when the warp is converged; if the call sits under a branch, pass `__activemask()` so the primitive names only the lanes that actually arrived. The rule that replaces the old assumption: a warp shuffle or vote is defined only for the lanes in its mask, and only once a `__syncwarp` has ordered them. [Post 04](../04-reduction/index.md) uses `__shfl_down_sync` in anger to sum the last warp of a reduction.

## 4. Atomics: correct counts, not ordered writes

Back to atomicity. An **atomic** operation performs a read-modify-write as one indivisible step: no other thread can slip between the read and the write, so no update is lost. `atomicAdd(&x, v)` is the common one, but the family is larger, and its most general member is **compare-and-swap**, `atomicCAS(addr, expected, desired)`: it stores `desired` only if `*addr` currently equals `expected`, and returns the old value either way. Every other atomic can be built from it in a retry loop, which is how you get an atomic the hardware does not provide directly (a `float` maximum on older cards, say):

```cuda
// build any atomic read-modify-write out of atomicCAS: retry until it sticks
int old = *addr, assumed;
do {
    assumed = old;
    int want = f(assumed);                 // whatever update you need
    old = atomicCAS(addr, assumed, want);  // install it only if unchanged
} while (assumed != old);                  // someone raced us: loop and retry
```

Here is the distinction that trips people up. An atomic guarantees **atomicity**: the increment is not lost. It does *not* guarantee **ordering**: it says nothing about when a *different* store this thread made becomes visible to other threads. If thread 0 writes `data[7] = 42` and then `atomicAdd(&flag, 1)`, another thread that observes the new `flag` may still read the old `data[7]`, because the two writes can reach memory in either order. Atomicity keeps one location self-consistent; ordering across locations is a separate guarantee, and it is the subject of Section 5.

One practical cost rides along. An atomic on a contended address is *serial*: when many threads target the same location, the hardware processes them one at a time. A histogram where every pixel lands in one bin therefore crawls, even though it moves almost no data. That is a first-class performance problem, and relieving it by privatizing the contention into shared memory is the whole story of [Post 05](../05-histogram/index.md). Correctness and contention are different axes: `atomicAdd` fixes the first and can wreck the second.

## 5. Making writes visible: fences and scopes

Ordering is enforced by a **memory fence**. `__threadfence()` is a one-way gate: every memory write a thread issued *before* the fence becomes visible to other threads before any write it issues *after* the fence. Pair it with the flag pattern from Section 4 and the bug closes: write the data, `__threadfence()`, then set the flag. The reader does the mirror image: see the flag, `__threadfence()`, then read the data, and it is guaranteed to see what the writer published. This is the **release/acquire** shape: the writer *releases* (fence, then publish the flag), and the reader *acquires* (see the flag, then fence, then read the data).

A fence has a **scope**: the set of threads whose view of memory it orders. CUDA names three, nested like rings:

- `__threadfence_block()` orders writes as seen by other threads in the same **block**.
- `__threadfence()` orders them for all threads on the **device** (the whole GPU).
- `__threadfence_system()` orders them for the device, the host, and peer GPUs (the whole **system**), which is what a kernel needs when it talks to CPU code through mapped memory.

![Three concentric rings labelled block, device, and system around a central write; the write propagates outward, with the inner block ring highlighted in green as the cheap default scope and the outer system ring in gray as the widest and most expensive reach.](diagrams/03-atomic-vs-scope.svg)
*A fence publishes a write only as far as its scope ring. Wider rings reach more threads and cost more.*

A wider scope costs more, so choose the narrowest ring that still reaches the threads which must see the write. The same scope idea extends to atomics on newer architectures (the `cuda::atomic` library takes an explicit scope), but the mental model stays fixed: atomics keep one location consistent, fences order writes across locations, and a scope says how far that ordering reaches.

## 6. There is no global barrier inside a kernel

One barrier is conspicuously missing from this whole post: a grid-wide one. There is no default `__syncgrid()`, because blocks are **independent** by design ([Post 01](../01-introduction-to-cuda/index.md)): the scheduler may run them in any order, or a few at a time on a small GPU, so a block that waits for another block can wait forever. `__syncthreads()` stops at the block boundary on purpose.

So how do you synchronize the whole grid? End the kernel. A kernel launch is itself a global barrier: every block of launch `A` finishes, and all its global-memory writes become visible, before any block of launch `B` begins. This is why multi-phase algorithms (the hierarchical scan and reduction of Posts 08 and 04, which reduce per block and then combine the partials) are split across separate launches. The launch boundary *is* the grid-wide sync that does not exist inside a kernel.

There is one escape hatch. **Cooperative Groups** can issue a launch whose blocks are guaranteed co-resident, after which `grid.sync()` is a genuine global barrier; it requires a cooperative launch and enough SM room to hold every block at once, so it is a specialist tool rather than the default. For almost everything, the rule from [Post 01](../01-introduction-to-cuda/index.md) holds: coordinate within a block with barriers, and across the grid with a new launch. With correctness settled, [Post 10](../10-roofline/index.md) turns to how fast a correct kernel can go and which resource sets the limit.

---

## Common pitfalls

- **Non-atomic shared counter.** `counter += 1` from many threads is a read-modify-write race, and most increments vanish. Use `atomicAdd`, or reduce and privatize so that few threads ever contend.
- **`__syncthreads()` in divergent code.** A barrier that only some threads reach is undefined and typically hangs the block. Lift it out of any `if`, early `return`, or per-thread-count loop so the whole block reaches the same barrier.
- **Trusting implicit warp-synchronous execution.** Since Volta, the lanes of a warp are not guaranteed to move in lockstep. Use `__syncwarp()` and pass the right mask (`__activemask()` under a branch); never just drop the sync.
- **Confusing atomicity with ordering.** An atomic stops a lost update; it does *not* make your other writes visible in order. Publish data with a `__threadfence()` before you set a flag, not with the atomic alone.
- **Over-wide fence scope.** `__threadfence_system()` where `__threadfence_block()` would do is correct but slow. Match the scope to the threads that actually need to see the write.
- **Expecting a global barrier inside a kernel.** Blocks are independent and unordered, so a block cannot wait on another. Split the phases across kernel launches, or use a cooperative launch with `grid.sync()`.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — Memory Fence Functions and Atomic Functions"* (current). The reference for `__threadfence`, `atomicAdd`, `atomicCAS`, and the memory scopes (reference).
- NVIDIA, *"CUDA C++ Programming Guide — Independent Thread Scheduling"* (current). Why the warp-synchronous assumption broke at Volta and what `__syncwarp` replaces it with (technical).
- Lin, Y. & Grover, V., *"Using CUDA Warp-Level Primitives"* (NVIDIA, 2018). The active-mask rules for shuffles, votes, and `__syncwarp` (technical, reference).
- Sorensen, T. et al., *"Portable Inter-Workgroup Barrier Synchronisation for GPUs"* (OOPSLA, 2016). Why a safe device-wide barrier is hard and what it costs (technical, foundational).
- Boehm, H. & Adve, S., *"Foundations of the C++ Concurrency Memory Model"* (PLDI, 2008). The acquire/release model that CUDA's fences follow (historical, foundational).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 10, Roofline and occupancy](../10-roofline/index.md)**: with correctness settled, turn to how fast a kernel can go and which resource caps it.
- **[Post 05, Histogram](../05-histogram/index.md)**: atomic contention as a first-class performance problem, and how privatization relieves it.
- **[Post 04, Reduction](../04-reduction/index.md)**: `__syncthreads()` and warp-shuffle used in anger to sum an array.
