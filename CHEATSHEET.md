# Cheatsheet — CUDA, from first kernel to FlashAttention, on one page

---

## The one mental model

A GPU runs the **same instruction across thousands of data elements at once**. You write a **kernel** from one element's point of view; the launch covers the whole array.

```
grid  ── made of ──▶ blocks ── made of ──▶ threads (run in warps of 32, in lockstep)
```

A launch is `kernel<<<blocksPerGrid, blockSize>>>(args)`.

---

## The global index (every 1-D kernel starts here)

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's element
if (i < N) C[i] = ...;                            // guard the surplus threads
int blocksPerGrid = (N + blockSize - 1) / blockSize;   // ceiling division, never down
```

- **Block size a multiple of 32** (warps are 32-wide). Defaults: 128, 256, 512.
- You launch *whole blocks*, so the grid overshoots `N` — the `if (i < N)` guard is not optional.

---

## Memory, fastest to slowest

| Space | Scope | Use it for |
|---|---|---|
| registers | one thread | scalars, accumulators |
| shared memory | one block | a hand-managed scratchpad (tiling, reductions) |
| L2 / global (DRAM) | whole grid | the data; minimize how often you touch it |

**Coalescing:** when the 32 threads of a warp read **consecutive** addresses, the hardware serves them as one 128-byte transaction. Strided access costs up to 32×.

---

## Always check errors

```cpp
kernel<<<g, b>>>(...);
cudaGetLastError();          // bad launch config — reported immediately
cudaDeviceSynchronize();     // async runtime fault — and waits for the result
```

A launch is **asynchronous**: the CPU queues it and runs on. Synchronize before you read results or time the kernel.

---

## Timing, honestly

```cpp
cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
kernel<<<g,b>>>(...); cudaDeviceSynchronize();      // untimed warm-up
cudaEventRecord(a); /* many iters */ cudaEventRecord(b); cudaEventSynchronize(b);
float ms; cudaEventElapsedTime(&ms, a, b);
```

Discard the first run; average many; report kernel-only **and** end-to-end (with transfer) separately.

---

## Memory-bound or compute-bound?

**Arithmetic intensity** `I = FLOPs / bytes moved`. Compare it to the device's
FLOP-to-bandwidth ratio.

- Low `I` (vector add ≈ 0.08, transpose, reduction) → **memory-bound**: the only lever is coalesced access that saturates DRAM bandwidth. More math is free.
- High `I` (tiled matmul, attention) → **compute-bound**: reuse data in shared memory and registers; feed the math units.

---

## Pick your `-arch`

| GPU | Compute capability | `-arch` |
|---|--:|---|
| GTX 16xx / RTX 20xx (Turing) | 7.5 | `sm_75` |
| A100 (Ampere) | 8.0 | `sm_80` |
| RTX 30xx (Ampere) | 8.6 | `sm_86` |
| RTX 40xx (Ada) | 8.9 | `sm_89` |
| H100 (Hopper) | 9.0 | `sm_90` |

Tensor Cores need ≥ 7.0; `bf16` and `cp.async` need ≥ 8.0.

---

## Build and run

```bash
nvcc -O3 -arch=sm_75 kernel.cu -o kernel && ./kernel
```

Full setup in [SETUP.md](SETUP.md); fixes in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
