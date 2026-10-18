# 15 - Asynchronous copy and software pipelining: cp.async, double buffering, and hiding memory latency

> **TL;DR.** Post 03's tiled matrix multiply idles each block while its next tile loads, a stall other resident blocks cover until too few fit on a streaming multiprocessor. Software pipelining keeps two shared-memory tiles and loads tile $t + 1$ while computing tile $t$; Ampere's `cp.async` does that copy without registers, and a CPU model predicts the gain from the load-to-compute ratio (1.78 times for eight balanced tiles, 2 at most). The `cp.async` kernel was compiled but not run: built for the author's GTX 1650 (compute capability 7.5), the same source becomes an ordinary copy and gains nothing, while a pipeline staged through registers ran 1.61 times faster with one block per multiprocessor and within 3% of the synchronous kernel with four.
>
> **Prerequisites:** [post 03](../03-matrix-multiplication/index.md).
> **Safe to skip?** Skip it if the reader can already write a double-buffered tile loop with `__pipeline_memcpy_async`, `__pipeline_commit` and `__pipeline_wait_prior`, place its barriers, and say when such a pipeline speeds a kernel up and on which cards `cp.async` exists.
>
> **After reading, you will be able to:**
>
> - Point at the stall in a synchronous tiled kernel and say when other resident blocks hide it and when they do not.
> - Use the cp.async model of issue, commit and wait, and read `__pipeline_wait_prior(n)` as a count of committed groups.
> - Write a double-buffered tile loop that issues the copy of tile t + 1 before computing tile t, with the barriers it needs.
> - Predict the latency-hiding speedup from the load-to-compute ratio and the tile count, and read a measured speedup through that model.
> - State the Ampere (sm_80) requirement of cp.async and what the same intrinsics compile to on an older card.

![Two timelines in model units, each with a load lane and a compute lane, for eight tiles that take 10 units to load and 10 to compute. Synchronous: loads and computes alternate and the compute lane stalls in red during every load, 160 units. Pipelined: the load of tile t + 1 runs beside the compute of tile t, between a prologue load and a closing compute, 90 units.](diagrams/01-stall-vs-pipeline.svg)

*Synchronous tiling idles the arithmetic during every load; the pipelined loop pays for one load and hides the rest behind the previous tile's math.*

---

## 1. The question: where does the tiled kernel wait?

[Post 03](../03-matrix-multiplication/index.md) cut the matrix multiply's global traffic with tiling. A block of `TILE x TILE` threads, `TILE` 16, computes a tile of `C: [M, N]` from `A: [M, K]` and `B: [K, N]`; thread `(row, col)`, with `col = blockIdx.x * blockDim.x + threadIdx.x` and `row` the same in y, owns `C[row * N + col]`. The block walks `K` one tile at a time. Post 03's kernel is the first of three in `snippets/async_pipeline.cu`, and its loop is:

```cuda
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;              // coalesced load of A
        s_A[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;              // coalesced load of B
        s_B[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                // tiles are loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();                                // done before the next load
    }
```

The assignment `s_A[...] = A[...]` is a load from global memory into a register and a store from it into shared memory. The store needs the value, so the thread stops there, and the CUDA C++ Programming Guide puts an off-chip operand "typically hundreds of clock cycles" away (section "Multiprocessor Level"). The first `__syncthreads()` then holds the block until the slowest load lands, so its arithmetic idles for every tile's load: the red stalls of the hero figure.

Whether the streaming multiprocessor (SM) idles too depends on what else it holds. A GPU hides latency by issuing from another warp, and the Guide says the warps needed depend "on the kernel code and its degree of instruction-level parallelism". Inside a block there is no other work: all 8 warps of a 256-thread block wait at one barrier for one tile. But a GTX 1650 SM holds 32 warps, four such blocks, and while one waits another computes. Section 5 measures it: with four blocks per SM the stall is mostly covered, with one it costs nearly half the speed. Library kernels, with large tiles and many registers, fit few blocks per SM, so they remove the stall inside the block by starting the next load before computing the current tile.

## 2. `cp.async`: a copy the thread does not wait for

The instruction for it arrived with the A100, compute capability 8.0; NVIDIA's A100 architecture whitepaper (V1.0, 2020) says the A100 GPU "includes a new asynchronous copy instruction that loads data directly from global memory into SM shared memory, eliminating the need for intermediate register file (RF) usage", and the copy "can be done in the background while the SM is performing other computations". In PTX, the parallel thread execution virtual instruction set, it is `cp.async`: a thread issues it, continues, and waits only when it asks to.

The C-style interface used here is three intrinsics in `cuda_pipeline.h`, the Programming Guide's "Pipeline Primitives Interface". Built for `sm_80`, they become the PTX and machine code (SASS) below, read from the program's PTX and from `cuobjdump -sass`:

| Intrinsic | PTX | SASS on `sm_80` | What it does |
|---|---|---|---|
| `__pipeline_memcpy_async(dst, src, 4)` | `cp.async.ca.shared.global` | `LDGSTS` | issue one 4-byte copy from global to shared memory |
| `__pipeline_commit()` | `cp.async.commit_group` | `LDGDEPBAR` | close the copies issued so far into one group |
| `__pipeline_wait_prior(n)` | `cp.async.wait_group n` | `DEPBAR.LE SB0, n` | wait until at most `n` of this thread's groups are in flight |

The Guide states the wait exactly: number a thread's commits $0, 1, \ldots, L$; `__pipeline_wait_prior(N)` waits for "completion of batches at least up to and including L-N". With one group per tile and one tile issued ahead, `__pipeline_wait_prior(1)` means every group but the newest has landed: the current tile is in shared memory and the prefetch still runs. The count is per thread, while the block reads a tile that 256 threads copied, so a `__syncthreads()` still follows the wait.

![Two copy paths from global memory to shared memory. Left, the sm_75 build: LDG loads into the register file and STS stores from it, and the warp waits at the STS for the loaded value. Right, the sm_80 build: one LDGSTS instruction, cp.async.ca for these 4-byte copies, writes shared memory with the register file unused, and the warp waits only at DEPBAR, the cp.async.wait_group before the math.](diagrams/02-copy-path.svg)

*The same source line is a load, a store and a stall on `sm_75`, and one instruction that the warp does not wait for on `sm_80`.*

Three rules matter. A copy is 4, 8 or 16 bytes, aligned to its size (128-byte alignment performs best, says the Guide). The header emits `cp.async.ca`, which may also cache in L1, for 4- and 8-byte copies and `cp.async.cg`, L2 only, for 16-byte ones; the whitepaper's "optionally bypassing L1 cache" is `.cg`, so this post's one-float copies do not bypass L1. And between issuing a copy and waiting for it, reading the destination or storing to either side is a race (Guide, "memcpy_async Primitive").

The copy saves a register and a stall, not a barrier: built for `sm_80`, `ptxas` gives the pipelined kernel 30 registers per thread and post 03's 32, too few to move an occupancy limit here.

## 3. Double buffering: issue tile t + 1, then compute tile t

A copy in flight needs a place that nobody reads, so the block keeps two tiles per operand, `s_A[2][TILE][TILE]` and `s_B[2][TILE][TILE]`, and alternates: while the multiply-adds read buffer `t & 1`, the copies of tile $t + 1$ fill the other. The second kernel of `snippets/async_pipeline.cu`:

```cuda
// Issue this thread's copies of tile t into buffer `buf`: one float of A and one of B. An element outside the matrix is a
// plain zero store, which no outstanding copy targets.
__device__ void issueTile(float (*s_A)[TILE][TILE], float (*s_B)[TILE][TILE], int buf, int t,
                          const float* __restrict__ A, const float* __restrict__ B,
                          int row, int col, int M, int N, int K)
{
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    if (row < M && aCol < K) __pipeline_memcpy_async(&s_A[buf][threadIdx.y][threadIdx.x], &A[row * K + aCol], sizeof(float));
    else                     s_A[buf][threadIdx.y][threadIdx.x] = 0.0f;
    if (bRow < K && col < N) __pipeline_memcpy_async(&s_B[buf][threadIdx.y][threadIdx.x], &B[bRow * N + col], sizeof(float));
    else                     s_B[buf][threadIdx.y][threadIdx.x] = 0.0f;
}

__global__ void matmulPipelined(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[2][TILE][TILE];                // two ping-pong buffers per operand
    __shared__ float s_B[2][TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int numTiles = (K + TILE - 1) / TILE;
    float sum = 0.0f;

    issueTile(s_A, s_B, 0, 0, A, B, row, col, M, N, K); // prologue: tile 0 into buffer 0
    __pipeline_commit();                                // one group per tile

    for (int t = 0; t < numTiles; ++t) {
        int cur = t & 1, next = (t + 1) & 1;
        if (t + 1 < numTiles) {                         // steady state: issue tile t + 1 before computing tile t
            issueTile(s_A, s_B, next, t + 1, A, B, row, col, M, N, K);
            __pipeline_commit();
        }
        __pipeline_wait_prior(t + 1 < numTiles ? 1 : 0);// this thread's copies of tile t have landed
        __syncthreads();                                // and every other thread's

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[cur][threadIdx.y][k] * s_B[cur][k][threadIdx.x];
        __syncthreads();                                // done reading `cur` before tile t + 2 is issued into it
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```

The prologue commits tile 0 as group 0. Iteration $t$ issues tile $t + 1$ into the other buffer as group $t + 1$, in flight before any arithmetic on tile $t$. `__pipeline_wait_prior(1)` releases the thread once group $t$ has landed; on the last tile the count is 0 and drains the pipe. The first barrier makes the other threads' copies visible; the second keeps iteration $t + 1$ from issuing tile $t + 2$ into `cur` while a slower warp still reads it. Outside the matrix a plain zero store is legal, since no copy targets that slot. The figure follows five tiles by the Guide's rule.

![A table of the pipelined loop over five tiles, one thread's view. Columns: the prologue and iterations t = 0 to 4. Rows: the group committed, 0 to 4 and then none, into buffers 0, 1, 0, 1, 0; the wait count, 1 four times and then 0; the group still in flight after the wait, the newest one; and the buffer computed from, 0, 1, 0, 1, 0, holding tile t.](diagrams/03-commit-groups.svg)

*A wait count of 1 lets only the newest group run on, so the tile each iteration computes from is always complete.*

## 4. Software pipelining: prologue, steady state, epilogue

Double buffering is the two-stage case of software pipelining, a schedule from compiler practice (Allan, Jones, Lee and Allan, "Software Pipelining", ACM Computing Surveys, September 1995) in which consecutive iterations overlap. A **prologue** issues the first load with nothing to compute; a **steady state** loads tile $t + 1$ beside the compute of tile $t$; an **epilogue** computes the last tile with nothing left to load. With $n$ tiles, a load time $L_t$ and a compute time $C_t$ per tile:

$$T_\text{sync} = \sum_{t=0}^{n-1} (L_t + C_t), \qquad T_\text{pipe} = L_0 + \sum_{t=0}^{n-2} \max(L_{t+1}, C_t) + C_{n-1}.$$

For equal tiles the speedup $n(L + C) / (L + (n - 1)\max(L, C) + C)$ tends to $(L + C) / \max(L, C)$: 2 when load and compute balance, less when either dominates. `snippets/pipeline_model.py` evaluates the sums; its numbers are derived by the model in abstract units and assume an overlap that is perfect and free:

```text
software-pipelining model (double-buffered tile loop), times in model units
  case           load  compute  tiles  synchronous  pipelined   speedup
  balanced         10       10      8          160         90     1.78x
  compute-bound     4       10      8          112         84     1.33x
  memory-bound     10        4      8          112         84     1.33x
  balanced         10       10     64         1280        650     1.97x
  balanced         10       10   1024        20480      10250     2.00x
  a measured 1.61x over 64 tiles: the model's speedup when compute takes 0.63 of a load's time
  at that share many tiles tend to 1.63x; the fit assumes a perfect overlap, so it cannot test one
```

Balanced, each overlapped step costs `max(10, 10) = 10` instead of 20, and eight tiles fall from 160 to 90 units, 1.78 times faster; the prologue and epilogue cost one tile each, so 64 tiles gain 1.97 and 1,024 tiles 2.00. The unbalanced cases mirror each other: when compute is longer the whole load hides, and when the load is longer the compute hides instead and the load sets the pace. Overlap hides latency; it adds no bandwidth, which is post 10's roofline again. The last two lines read section 5's measurement through the model.

## 5. Measured on a Turing card: the fallback, and a pipeline without `cp.async`

The author's GTX 1650 has compute capability 7.5, and `cp.async` needs 8.0. The Programming Guide says the `memcpy_async` APIs built on `cuda::barrier` and `cuda::pipeline` need compute capability 7.0 and that from 8.0 they "can benefit from hardware acceleration" (section "memcpy_async API"). The CUDA 12.3 header shows what happens below 8.0: `cuda_pipeline_helpers.h` sets `_CUDA_PIPELINE_HAS_ASYNC_COPY` to 1 only when `__CUDA_ARCH__ >= 800`, and otherwise `__pipeline_memcpy_async` calls `pipeline_memcpy_sync`, a plain copy, while commit and wait have empty bodies. The steady-state lines of `cuobjdump -sass` for `matmulPipelined`, encodings removed, show both builds:

```text
sm_80 build:
/*0460*/               @P3 LDGSTS.E [R5], [R18.64] ;
/*04e0*/                   LDGSTS.E [R7], [R4.64] ;
/*0500*/                   LDGDEPBAR ;
/*0510*/                   DEPBAR.LE SB0, 0x1 ;
/*0530*/                   BAR.SYNC 0x0 ;
sm_75 build:
/*0340*/               @P2 LDG.E.CONSTANT.SYS R7, [R26] ;
/*0350*/               @P3 LDG.E.CONSTANT.SYS R4, [R4] ;
/*03f0*/               @P2 STS [R6], R7 ;
/*0410*/               @P3 STS [R9], R4 ;
/*0430*/                   BAR.SYNC 0x0 ;
```

The fallback is real and sound: the logic runs and can be checked on Turing, but the copy of tile $t + 1$ completes, through a register, before tile $t$'s arithmetic. On `sm_75` the "pipelined" kernel is post 03's loop with a second buffer.

A pipeline that works on Turing must keep the load in flight across the math by itself. The third kernel, `matmulPrefetchRegs`, loads tile $t + 1$ into two registers, computes tile $t$, and only then stores them into the other buffer, so each warp waits for its load at the store, after the multiply-adds. It is the form of CUTLASS's mainloop for cards without `cp.async` (`MmaPipelined`, whose "Pair of fragments used to overlap global memory loads and math instructions" are these registers). Its loop, after a prologue that loads tile 0:

```cuda
    for (int t = 0; t < numTiles; ++t) {
        int cur = t & 1;
        int aCol = (t + 1) * TILE + threadIdx.x;        // tile t + 1; past the last tile both loads are skipped
        int bRow = (t + 1) * TILE + threadIdx.y;
        float a = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;   // issued now; nothing waits for them yet
        float b = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE; ++k)                  // compute tile t while those loads are in flight
            sum += s_A[cur][threadIdx.y][k] * s_B[cur][k][threadIdx.x];
        s_A[cur ^ 1][threadIdx.y][threadIdx.x] = a;     // first use of a and b: the warp waits here, after the math
        s_B[cur ^ 1][threadIdx.y][threadIdx.x] = b;
        __syncthreads();                                // tile t + 1 is complete, and tile t is no longer read
    }
```

One barrier per tile suffices: the store into `cur ^ 1` follows the previous iteration's barrier, by which every thread had finished reading that buffer, and the closing barrier makes the new tile visible.

`async_pipeline_sm75.cu` is the same program built with `-arch=sm_75`. It checks the kernels at $M = N = K = 1{,}000$, not a multiple of `TILE` (post 03's against a double-precision reference on 4,096 samples, the others against post 03's), then times them at $1{,}024^3$, $2MNK$ floating-point operations (FLOP) each, as launched and with 40 KB of unused dynamic shared memory: compute capability 7.5 offers at most 64 KB per SM, so that leaves one block, 8 warps, per SM. Each row is the fastest of 5 passes of 10 launches, every pass timing all six rows, after half a second of untimed launches, with the SM clock measured around them. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026; the `__pipeline_*` rows are the software fallback on compute capability 7.5: the copy is not asynchronous, so they measure the double-buffer structure, not `cp.async`:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), 14 SMs
kernels compiled for compute capability 7.5: __pipeline_memcpy_async is the software fallback,
  a load into a register and a store to shared memory; commit and wait do nothing
check at M = N = K = 1000 (not a multiple of TILE = 16):
  synchronous (post 03)      largest relative error on 4096 samples: 1.8e-06  ok
  pipelined, __pipeline_*    elements that differ from the synchronous kernel: 0 of 1000000  ok
  pipelined, via registers   elements that differ from the synchronous kernel: 0 of 1000000  ok
time at M = N = K = 1024: ms per launch, the fastest of 5 passes of 10 launches; GFLOP/s = 2 * M * N * K over the time
  kernel                      blocks/SM  warps/SM   time (ms)  GFLOP/s  vs synchronous
  no extra shared memory:
  synchronous (post 03)               4        32       6.469      332           1.00x
  pipelined, __pipeline_*             4        32       7.263      296           0.89x
  pipelined, via registers            4        32       6.647      323           0.97x
  40 KB of extra shared memory, one block per SM:
  synchronous (post 03)               1         8      12.310      174           1.00x
  pipelined, __pipeline_*             1         8      12.260      175           1.00x
  pipelined, via registers            1         8       7.640      281           1.61x
SM clock: 1445 MHz before the timed launches, 1500 MHz after
every kernel matches the synchronous kernel: PASS
```

These are measured numbers at the printed 1,445 to 1,500 MHz; the card runs below its 1,740 MHz boost under a sustained kernel, three other clean runs gave the same ratios within 0.02, and the reader's card will differ. The kernels sum in the same order, so both pipelines match post 03's bit for bit; post 03 reported that kernel at 305 GFLOP/s as a median, not the fastest pass.

![Two bar charts of GFLOP/s at 1,024 cubed. Four blocks per SM: synchronous 332, the __pipeline_ fallback 296, the register pipeline 323. One block per SM: synchronous 174, fallback 175, register pipeline 281, 1.61 times the synchronous kernel.](diagrams/04-measured-turing.svg)

*With four blocks per SM other blocks already cover the stall; with one, only a pipeline whose load stays in flight recovers it.*

With four blocks per SM the pipelines gain nothing: other blocks already hide the stall, the fallback runs a tenth slower and the register pipeline within 3% of the synchronous kernel. With one block per SM the synchronous kernel falls from 332 to 174 GFLOP/s as its 8 warps stall together; the fallback gains nothing; the register pipeline runs 1.61 times faster, at 281 GFLOP/s, near the four-block rate. (The 64 KB of shared memory may also shrink L1, which shares 96 KB with it, in all three rows.) Read through section 4's model, which assumes a perfect overlap and here a longer load, 1.61 times over 64 tiles puts compute at 0.63 of a load's time, and at that share many tiles would tend to 1.63 times; the fit assumes the overlap, so it cannot show how complete the real one was. What `cp.async` adds on an Ampere card is not measured in this edition.

## 6. Multi-stage pipelines in the libraries

Two buffers keep one tile in flight. When a load outlasts one tile's compute, a deeper pipeline keeps more: with $s$ stages the block issues tile $t + s - 1$ while computing tile $t$ and waits with `__pipeline_wait_prior(s - 1)`. CUTLASS, NVIDIA's open template library for matrix multiplication, makes the depth a template parameter: `cutlass::gemm::device::Gemm` takes `int Stages`, "Number of stages used in the pipelined mainloop", and its default configurations use 3 stages for Ampere (`Sm80`) Tensor Core kernels and 2 for Turing, Volta and plain-arithmetic kernels (`default_gemm_configuration.h`, consulted October 2026). cuBLAS does not publish its kernels; CUTLASS is the public view of how such a multiply is built.

[Post 17](../17-tensor-cores/index.md) replaces the scalar multiply-adds with Tensor Core instructions that consume a tile far faster, which makes the load the longer stage and the pipeline essential. [Post 19](../19-cutlass-triton/index.md) stops writing it by hand: CUTLASS and Triton compose the staging, the Tensor Core math and the tile schedule.

## 7. Make it run

Both programs build from the series root; on Windows they are `.exe` files and `nvcc` may need `-ccbin` ([RUNNING.md](../../RUNNING.md) covers both):

```bash
nvcc -O3 -arch=sm_80 posts/15-async-copy-pipelining/snippets/async_pipeline.cu -o async_pipeline && ./async_pipeline
nvcc -O3 -arch=sm_75 posts/15-async-copy-pipelining/snippets/async_pipeline_sm75.cu -o async_pipeline_sm75 && ./async_pipeline_sm75
uv run python posts/15-async-copy-pipelining/snippets/pipeline_model.py
```

On an Ampere card or newer the first line builds the `cp.async` path, and the program's first lines say so; `sm_86` suits an RTX 3080. The `sm_80` build compiled here, but on the GTX 1650 it stops at its first launch with `the kernels cannot run on this card (no kernel image is available for execution on the device): build with -arch=sm_75` and exits 1, so no `cp.async` timing exists for this edition. The second line is section 5's Turing build; it runs in about 7 seconds. The model needs only Python.

## 8. What can go wrong?

Each of these happened while this post's harness was written and run.

- **The compiler moved the prefetch after the math.** With loads and stores both inside `if (t + 1 < numTiles)`, `cuobjdump -sass` showed the loads after 12 of the 16 multiply-adds, so nothing overlapped. Unconditional stores (zeros after the last tile, into a buffer nobody reads) put them back at the top; instruction order is read from the SASS.
- **Timing the rows one after another.** Timed kernel by kernel, later rows met a different SM clock (1,469 MHz before, 1,679 after); timing all six rows inside each pass made the ratios repeat within 0.02.
- **A power-source flap during a run.** One run printed 1,469 MHz before and 250 MHz after, with the register pipeline 23% slower than the synchronous kernel; the printed clock catches such a run.
- **Running the `sm_80` build on a 7.5 card.** It compiles, and every launch fails with "no kernel image is available"; the program checks with one launch and names the flag.

---

## Common pitfalls

1. **Reading a buffer that is still being filled.** Between `__pipeline_memcpy_async` and its wait, a load from the destination is a race; compute only from the buffer whose group the wait has completed.
2. **Counting commits wrong.** `__pipeline_wait_prior(n)` counts this thread's committed groups, so commit one group per tile; with nothing issued on the last tile, the final wait must be `__pipeline_wait_prior(0)`.
3. **Dropping a barrier because the copy is asynchronous.** The wait covers one thread's copies; the `__syncthreads()` after it completes the block's tile, and the one after the math keeps the next copy out of a buffer still being read.
4. **Expecting a speedup at full occupancy.** With four blocks per SM the GTX 1650 already hid the stall, and neither pipeline gained more than 3%; pipelining pays when few warps share an SM.
5. **Calling the `sm_75` build asynchronous.** Below compute capability 8.0 `cuda_pipeline.h` compiles the copy to a load and a store and the wait to nothing; its times say nothing about `cp.async`.

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, sections "Asynchronous Data Copies", "Asynchronous Data Copies using cuda::pipeline" (with "Pipeline Primitives Interface") and "Multiprocessor Level", consulted October 2026 (CUDA 12.3 edition). The intrinsics, the wait rule, the size and alignment rules, and the latency of off-chip operands.
- NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture*, whitepaper V1.0, 2020. The asynchronous copy, and why it needs no register.
- NVIDIA, "Efficient GEMM in CUDA", CUTLASS documentation, with `include/cutlass/gemm/device/default_gemm_configuration.h` and `include/cutlass/gemm/threadblock/mma_pipelined.h`, consulted October 2026. Double buffering, the `Stages` parameter, and the register-staged mainloop.
- Allan, V. H., Jones, R. B., Lee, R. M. and Allan, S. J., "Software Pipelining", *ACM Computing Surveys* 27(3), 367 to 432, September 1995. Prologue, steady state and epilogue as compiler scheduling.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 17 - Tensor cores: matrix-at-once hardware and mixed precision](../17-tensor-cores/index.md):** the Tensor Core instructions that shrink each tile's compute time, which makes the load the longer stage.
- **[Post 19 - CUTLASS and Triton: why nobody writes raw CUDA for matmul anymore](../19-cutlass-triton/index.md):** the libraries that build this pipeline, several stages deep, for every card.
