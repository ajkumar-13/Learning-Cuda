# Cheatsheet: CUDA, from first kernel to FlashAttention, on one page

Print it, pin it. Every line here is taught, with its reasons, in the post named beside it; the names follow the [notation guide](notation_guide.md).

---

## The one mental model (post 01)

A GPU runs the same instruction across thousands of data elements at once. A kernel is written from one element's point of view, and the launch covers the whole array: on the GPU the grid is the loop.

```
grid    -> made of blocks
block   -> made of threads, up to 1024
threads -> scheduled in warps of 32 that share one instruction
```

A launch is `kernel<<<blocksPerGrid, blockSize>>>(args)`.

---

## The global index (posts 01 and 02)

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;         // this thread's element
if (i < N) C[i] = A[i] + B[i];                          // guard the surplus threads
int blocksPerGrid = (N + blockSize - 1) / blockSize;    // round up, never down
```

- Make the block size a multiple of 32, because warps are 32 wide; 128, 256 and 512 are the usual choices.
- The launch is whole blocks, so the grid overshoots `N`: the `if (i < N)` guard is not optional.
- In 2D: `int row = blockIdx.y * blockDim.y + threadIdx.y;` and `int col = blockIdx.x * blockDim.x + threadIdx.x;`.

---

## Memory, fastest to slowest (post 01)

| Tier | Scope | Use it for |
|---|---|---|
| registers | one thread | scalars, accumulators |
| shared memory | one block | a hand-managed scratchpad: tiles, reductions, transposes |
| L2 cache | whole GPU | managed by the hardware over global memory |
| global memory (DRAM) | whole GPU and the host | the data; touch it as few times as possible |

Coalescing (posts 02 and 06): when the 32 threads of a warp touch consecutive addresses, the hardware serves the warp with as few 128-byte transactions as the data needs; a strided pattern can cost up to 32 times the traffic. Shared memory has 32 banks: pad a tile to `[TILE][TILE + 1]` to break bank conflicts.

---

## Always check errors (post 02)

```cpp
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

kernel<<<g, b>>>(args);
CUDA_CHECK(cudaGetLastError());        // a bad launch configuration, reported at once
CUDA_CHECK(cudaDeviceSynchronize());   // a fault inside the kernel, and waits for it to finish
```

A launch is asynchronous: the CPU queues it and runs on. Synchronize before reading results or stopping a timer.

---

## Synchronization (posts 03, 04 and 09)

| Call | What it guarantees |
|---|---|
| `__syncthreads()` | every thread of the block arrives before any continues; never inside a branch some threads skip |
| `__syncwarp(mask)` | the lanes in `mask` meet; needed since compute capability 7.0, where a warp is not guaranteed to run in lockstep |
| `atomicAdd(&x, v)` | one indivisible read-modify-write: no lost update, but no ordering of other writes |
| `__threadfence()` | this thread's writes are seen in order across the device; `_block` and `_system` set other scopes |
| a new kernel launch | the only barrier across the whole grid |

---

## Timing, honestly (post 02)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
kernel<<<g, b>>>(args); cudaDeviceSynchronize();       // untimed warm-up
cudaEventRecord(start);
for (int r = 0; r < reps; ++r) kernel<<<g, b>>>(args);
cudaEventRecord(stop); cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);       // divide by reps
```

Discard the first run, average many, and report the kernel alone and the end-to-end time with the copies separately.

---

## Memory bound or compute bound? (post 10)

Arithmetic intensity $I$ is FLOP per byte moved to and from DRAM. The roofline caps attainable performance at the smaller of the compute peak and $I \cdot \beta$, where $\beta$ is the card's peak bandwidth.

- Low $I$ (vector addition, about 0.083; transpose; reduction): memory bound. The lever is coalesced access that saturates DRAM bandwidth; extra arithmetic is free.
- High $I$ (tiled matrix multiplication, attention): compute bound. Reuse data in shared memory and registers, and feed the arithmetic units.

---

## Pick the `-arch` (the [running guide](RUNNING.md))

| GPU | Compute capability | `-arch` |
|---|--:|---|
| V100 (Volta) | 7.0 | `sm_70` |
| GTX 16xx, RTX 20xx (Turing) | 7.5 | `sm_75` |
| A100 (Ampere) | 8.0 | `sm_80` |
| RTX 30xx (Ampere) | 8.6 | `sm_86` |
| RTX 40xx (Ada) | 8.9 | `sm_89` |
| H100 (Hopper) | 9.0 | `sm_90` |

Tensor Cores need compute capability 7.0 or above; BF16 arithmetic and `cp.async` need 8.0 or above.

---

## Build and run

```bash
nvcc -O3 -arch=sm_75 kernel.cu -o kernel && ./kernel
```

On Windows, add `-ccbin` with the `Hostx64\x64` folder of the installed MSVC toolset, or build from a Developer Command Prompt. The [running guide](RUNNING.md) has the full recipe and the fixes for the build failures that still occur.
