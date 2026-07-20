# 15 · Asynchronous copy and software pipelining — cp.async, double buffering, and hiding memory latency

> **TL;DR.** The tiled matrix multiply from post 03 spends most of its time waiting: it loads a tile, calls `__syncthreads()`, computes, and repeats, so while a tile crawls in from global memory the compute units sit idle for hundreds of cycles. The fix is to stop waiting. On an Ampere graphics processing unit (GPU), the `cp.async` instruction copies global memory straight into shared memory in the background, bypassing the register file, so you can keep **two** shared-memory tiles and prefetch the next tile while computing the current one. This post builds that double-buffered, software-pipelined kernel, derives exactly how much latency it hides with a CPU model, and shows it is the same trick cuBLAS and CUTLASS use under the hood.
>
> **After reading this you will be able to:**
> - Point at the stall in a synchronous tiled kernel and explain why the compute units wait.
> - Use the `cp.async` programming model: issue copies, `commit`, and `wait` until only N groups remain in flight.
> - Write a double-buffered (ping-pong) tile loop that prefetches tile k+1 while computing tile k.
> - Predict the latency-hiding speedup from a load-to-compute ratio, and state the Ampere (`sm_80`) requirement.

![Two timelines for a four-tile loop. The synchronous one alternates load, a red stall bubble where compute waits, then compute, four times. The pipelined one runs loads on their own row staggered ahead of computes, with a green band marking the steady state where load and compute overlap, finishing much sooner.](diagrams/01-stall-vs-pipeline.svg)
*Synchronous tiling waits for every load (the red bubbles); pipelining prefetches the next tile so the load hides behind compute.*

---

## 1. The motivation: the stall hiding inside tiled matmul

Post 03 cut the matrix multiply's global traffic with tiling: each block loads a `TILE x TILE` block of `A` and `B` into shared memory once, then every thread reuses it. The inner loop walks the shared dimension `K` in tiles, and its shape is fixed:

```cuda
for (int t = 0; t < numTiles; ++t) {
    // load tile t of A and B into shared memory
    sA[ty][tx] = A[...];  sB[ty][tx] = B[...];
    __syncthreads();                 // wait: every thread finished loading
    for (int k = 0; k < TILE; ++k)   // compute this tile's contribution
        sum += sA[ty][k] * sB[k][tx];
    __syncthreads();                 // wait: every thread finished reading
}
```

Read the timing, not the correctness. The compute loop reads the *very tile the load just wrote*, and a `__syncthreads()` barrier sits between them, so the compute **cannot begin** until the load has fully landed. A load from global memory (off-chip dynamic random-access memory, DRAM) takes hundreds of cycles; the few multiply-adds in the compute loop take far fewer. So the streaming multiprocessor (SM) loads a tile, then **stalls** waiting for it, then computes a little, then stalls again. The red bubbles in the hero image are that wait, repeated once per tile.

> **Why the stall is not "just" hidden by other warps.** GPUs hide latency by switching to another ready warp when one stalls. That works when you have enough independent warps resident. But in a shared-memory tiled kernel every warp in the block hits the *same* `__syncthreads()` at the *same* time, so they all stall together on the same load. There is no other warp to switch to inside the block; the barrier serializes load and compute. We have to overlap them by hand.

The work is not the problem; the *waiting* is. If we could keep the compute units busy on tile k while the load of tile k+1 ran in the background, the load latency would disappear behind useful math. That is exactly what the rest of this post builds.

## 2. The mechanism: cp.async, the asynchronous copy

Ampere (compute capability 8.0, the `sm_80` architecture) added an instruction for exactly this: `cp.async` copies from global memory to shared memory **asynchronously**, and crucially it does **not** route the data through registers. An ordinary load goes global memory into a register, then a separate store moves the register into shared memory (SMEM), occupying a register and stalling the thread the whole time. `cp.async` goes global memory through the level-two (L2) cache straight into shared memory, in the background, touching no register at all.

![Two data paths. The synchronous path, grayed and dashed, goes global memory into the register file then out to shared memory. The cp.async path, solid green, goes global memory through the L2 cache straight into shared memory, with the register file drawn faded and crossed out to show it is bypassed.](diagrams/02-cp-async-path.svg)
*An ordinary load detours through a register and stalls the thread; `cp.async` copies global to shared directly and runs in the background.*

Two consequences fall out of bypassing the register file. First, the copy frees the registers a manual load would have burned, which raises occupancy. Second, because it runs in the background, the thread issues the copy and **keeps going**; it only waits when it actually needs the data. You manage that wait with a three-call programming model:

| Call (the `cuda_pipeline.h` wrapper) | Underlying PTX | What it does |
|---|---|---|
| `__pipeline_memcpy_async(dst, src, bytes)` | `cp.async` | issue one asynchronous global-to-shared copy |
| `__pipeline_commit()` | `cp.async.commit_group` | bundle the issued copies into one *group* |
| `__pipeline_wait_prior(n)` | `cp.async.wait_group n` | block until at most `n` groups are still in flight |

> **Read `wait_prior(n)` as "wait until only the most recent `n` groups remain."** You commit a group per tile. If you have prefetched the next tile (one group in flight) and want the *current* tile's copy to be done, you call `__pipeline_wait_prior(1)`: it waits until only the one newest group (the prefetch) is still running, which means the older group (the current tile) has landed. The prefetch keeps running across the barrier. That single number is how you keep exactly one load ahead.

The same logic is available through the higher-level `cuda::pipeline` C++ application programming interface (API) and, at the lowest level, by emitting the `cp.async` PTX directly. We use the `__pipeline_*` intrinsics here because they map one-to-one onto the instruction and read clearly.

## 3. Double buffering: prefetch the next tile

With an asynchronous copy in hand, the overlap is a question of storage: you cannot compute from a buffer that is simultaneously being overwritten by a load. So keep **two** shared-memory tiles per matrix and ping-pong between them. While the compute units read buffer A (tile k), `cp.async` fills buffer B (tile k+1). Next iteration, swap: compute reads B, the copy fills A.

![Two shared-memory tile buffers A and B. In step k, compute reads buffer A (traced green) while cp.async fills buffer B from global memory (blue). A swap arrow leads to step k+1 where the roles reverse: compute reads B and cp.async fills A.](diagrams/03-double-buffer.svg)
*Trace buffer A: read in step k, refilled in step k+1. One buffer is always full while the other fills, so the load never blocks compute.*

In code this doubles the shared-memory declaration and selects a buffer by the low bit of the tile index:

```cuda
__shared__ float s_A[2][TILE][TILE];     // two ping-pong buffers
__shared__ float s_B[2][TILE][TILE];

// prologue: start loading tile 0 into buffer 0, then commit the group
__pipeline_memcpy_async(&s_A[0][ty][tx], &A[...], sizeof(float));
__pipeline_memcpy_async(&s_B[0][ty][tx], &B[...], sizeof(float));
__pipeline_commit();

for (int t = 0; t < numTiles; ++t) {
    int cur = t & 1, next = (t + 1) & 1;          // which buffer is which
    if (t + 1 < numTiles) {                        // prefetch the NEXT tile
        __pipeline_memcpy_async(&s_A[next][ty][tx], &A[...], sizeof(float));
        __pipeline_memcpy_async(&s_B[next][ty][tx], &B[...], sizeof(float));
        __pipeline_commit();
    }
    int keep = (t + 1 < numTiles) ? 1 : 0;   // last tile: no prefetch, so drain its own copy
    __pipeline_wait_prior(keep);  // current tile has landed; a live prefetch keeps running
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k)                 // compute from `cur`
        sum += s_A[cur][ty][k] * s_B[cur][k][tx];
    __syncthreads();
}
```

The prefetch of tile `t+1` is issued *before* the compute of tile `t`, so the copy is already in flight when the multiply-adds run. `__pipeline_wait_prior(keep)` then blocks only on the current tile, never on a live prefetch; and on the final tile, where no prefetch is outstanding, `keep = 0` drains that last copy before the compute reads it. The load of tile k+1 now overlaps the compute of tile k, which is the entire point.

How much does that buy? It depends on the load time relative to the compute time, and you can compute the answer exactly. Model the loop as `K` tiles, each with a `load_t` and a `comp_t`. Synchronous makespan is the full sum; the pipelined makespan pays the first load (the prologue), then one `max(load, compute)` per overlapped step, then the last compute (the epilogue):

$$T_\text{sync} = \sum_k (\text{load}_k + \text{comp}_k), \qquad T_\text{pipe} = \text{load}_0 + \sum_{k} \max(\text{load}_{k+1}, \text{comp}_k) + \text{comp}_\text{last}$$

The committed model [snippets/pipeline_model.py](snippets/pipeline_model.py) computes both for a few ratios. This model is the **verified artifact**: it is exact arithmetic, not a timing, and it prints

```text
software-pipelining model (double-buffered tile loop)

  balanced  (load == compute)
    per tile          : load=10, compute=10  (K=8 tiles)
    synchronous        = 160  units   (sum of all loads + computes)
    pipelined          = 90  units   (load_0 + overlap + compute_last)
    speedup            = 1.78x
    hidden per step    = 10 units of load tucked behind compute
```

When load and compute cost the same, each overlapped step costs `max(10, 10) = 10` instead of `20`, so eight tiles fall from 160 to 90 units, a `1.78x` speedup that climbs toward `2x` as the loop lengthens (more on that in section 4). Run the kernel and the model with:

```bash
nvcc -O3 -arch=sm_80 snippets/async_pipeline.cu -o async_pipeline && ./async_pipeline
python snippets/pipeline_model.py
```

The kernel [snippets/async_pipeline.cu](snippets/async_pipeline.cu) is the full double-buffered matmul with a `CUDA_CHECK` macro, a CPU reference, and a correctness check (`max abs error vs CPU` printed at the end); it refuses to run below compute capability 8.0 because `cp.async` does not exist there.

## 4. Software pipelining, generalized

Double buffering is the two-stage case of a general idea: **software pipelining**, where you stagger a loop so that stage *s* of iteration *k* runs alongside stage *s−1* of iteration *k+1*. A pipeline has three phases. The **prologue** issues the first load(s) with nothing to compute yet, filling the pipe. The **steady state** does the overlapped work: prefetch tile k+1 while computing tile k, both engines busy. The **epilogue** computes the last tile with no load left to issue, draining the pipe.

![A pipeline chart over five tile iterations with a load row and a compute row. The prologue issues the first load alone, the steady state (green band) runs load of tile k+1 alongside compute of tile k with both engines busy, and the epilogue computes the last tile with no load left.](diagrams/04-pipeline-stages.svg)
*The load row stays one step ahead of the compute row; prologue fills the pipe, steady state overlaps, epilogue drains.*

The model makes the asymptotics precise. For the balanced case as the tile count `K` grows:

```text
  balanced, K=   8: speedup = 1.7778x  (-> 2.0000x as K -> infinity)
  balanced, K=  64: speedup = 1.9692x  (-> 2.0000x as K -> infinity)
  balanced, K=1024: speedup = 1.9980x  (-> 2.0000x as K -> infinity)
```

The prologue and epilogue are fixed overheads of one tile each; amortized over a long loop they vanish, and the steady state runs at the cost of the slower stage. That gives the rule for when pipelining helps: **the load latency is fully hidden as long as there is enough compute to cover it.** The model's other two cases show both sides. When the load is shorter than the compute (`load=4, compute=10`), the entire load hides and the loop runs at compute speed. When the load is longer (`load=10, compute=4`), compute hides instead and the load latency now sets the pace; you still win, but the win is capped by bandwidth, not arithmetic. This is the roofline picture from post 10 again: a compute-bound tile can absorb its loads, a memory-bound one cannot fully.

> **`cp.async` needs Ampere. Older GPUs emulate it.** The asynchronous copy is an `sm_80` instruction. On pre-Ampere hardware (for example the `sm_75` Turing GPU used elsewhere in this series) you get the same overlap by hand: in the steady state, issue an ordinary *load to registers* for tile k+1 while computing tile k from shared memory, then store the registers into the other shared buffer after the compute. It costs the registers `cp.async` would have saved and is fiddlier to write, but the pipeline structure (prologue, overlapped steady state, epilogue) is identical. `cp.async` is the hardware finally doing for free what people had been hand-rolling for years.

## 5. The payoff and the bridge

Hand-written tiling (post 03) gets you load-once-reuse. Pipelining gets you the next thing the production libraries do: it is precisely how cuBLAS and CUTLASS structure their general matrix multiply (GEMM). They run **multi-stage** pipelines (often three to five `cp.async` stages deep, not just two), so several tiles are in flight at once and even a long load latency is buried under the compute of several earlier tiles. CUTLASS exposes this directly as a `Stages` template parameter; the post 03 observation that "cuBLAS does double buffering" is this kernel, generalized and tuned.

Two threads run from here. Post 17 (Tensor cores) replaces the scalar multiply-add inner loop with a hardware matrix-multiply unit that consumes tiles far faster, which makes the load even more likely to be the bottleneck and makes `cp.async` pipelining essential rather than merely nice. Post 19 (CUTLASS and Triton) is where you stop writing this by hand: the library composes `cp.async` staging, Tensor-core math, and tile scheduling for you, and the reason it is worth using is that it has already solved the pipeline you just built. This post is the missing rung between the two: the manual tiled kernel of post 03 and the "the library does this for you" of post 19.

---

## Common pitfalls

- **Computing from a buffer still being filled.** The whole reason for two buffers is that a `cp.async` into a buffer races with any read of it. Single-buffer plus `cp.async` is a data race; you must prefetch into the *other* buffer and only read a buffer whose copy you have waited on.
- **Mismatching `commit` and `wait_prior`.** `wait_prior(n)` counts *committed groups*, so its meaning depends on how many `__pipeline_commit()` calls you have made. Commit exactly one group per tile and the `wait_prior(1)` reads as "current tile done, one prefetch still running." Forget a `commit` and the counting silently drifts.
- **Dropping `__syncthreads()` because the copy is "async."** `cp.async` is per-thread asynchronous, but the tile in shared memory is read by the *whole block*, so you still need a `__syncthreads()` after the wait to guarantee every thread sees the completed tile, and another after the compute before the buffer is reused.
- **Building for the wrong architecture.** `cp.async` is `sm_80`. Compile with `-arch=sm_75` and the intrinsic will not lower; run an `sm_80` binary on an older card and the launch fails. The kernel guards this with a `prop.major < 8` check at startup.
- **Pipelining a kernel with too little compute.** If a tile's compute is much shorter than its load, the load cannot hide (the model's memory-bound case): you stay bandwidth limited and pipelining only recovers the prologue. Pipelining hides latency; it does not raise peak bandwidth.
- **Copying a size `cp.async` cannot do in one shot.** The hardware path is fastest for 4-, 8-, and 16-byte aligned copies. Odd sizes or misaligned addresses fall back to a slower path; match your per-thread copy to a `float`, `float2`, or `float4`.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — Asynchronous Data Copies"* (current). The reference for `cp.async`, the `cuda::pipeline` API, and the `__pipeline_*` intrinsics (technical, reference).
- NVIDIA, *"NVIDIA A100 Tensor Core GPU Architecture"* whitepaper (2020). Introduces the Ampere asynchronous copy and why it bypasses the register file (technical, historical).
- NVIDIA, *"CUTLASS: Efficient GEMM"* documentation (current). How a production GEMM composes multi-stage `cp.async` pipelines, named by the `Stages` parameter (technical, reference).
- Allan, V. H., Jones, R. B., Lee, R. M., & Allan, S. J., *"Software Pipelining"* (ACM Computing Surveys, 1995). The compiler-theory origin of prologue, steady state, and epilogue scheduling (technical, historical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 14, Kernel fusion](../14-kernel-fusion/index.md)**: the other half of "stop waiting on memory," by removing whole round-trips to DRAM instead of hiding them.
- **[Post 16, Floating-point precision](../16-numerical-precision/index.md)**: the formats those tiles are stored and accumulated in (FP16, BF16, TF32) and why the accumulator's precision decides whether the answer is trustworthy.
- **[Post 03, Matrix multiplication](../03-matrix-multiplication/index.md)**: the synchronous tiled kernel this post upgrades, where the stall first appears.
