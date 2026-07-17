# 06 · Matrix transpose — coalescing, shared memory, and bank conflicts

> **TL;DR.** Transposing a matrix is pure data movement (`B[col][row] = A[row][col]`) with no arithmetic to optimize, yet a naive GPU version runs at about 17% of a plain copy's bandwidth, because **coalescing** makes 32 threads reading a row cost one 128-byte transaction while 32 threads reading a *column* cost up to 32. A transpose must touch a row on one side and a column on the other, so one access is always strided unless you stage a tile in **shared memory** and turn it there: load coalesced, store coalesced, and let the strided access happen on-chip. That introduces a 32-way **bank conflict** on the column read, which a one-column **pad** removes. The result recovers ~88% of a plain copy's bandwidth.
>
> **After reading this you will be able to:**
> - Explain coalesced versus strided access in terms of 128-byte transactions.
> - Diagnose why a naive transpose wastes most of its memory bandwidth.
> - Use a shared-memory tile to make both global accesses coalesced.
> - Recognize and remove a shared-memory bank conflict with padding.

![Two panels: a coalesced row read served by one 128-byte transaction, versus a strided column read that touches a separate cache line per thread.](diagrams/01-coalesced-vs-strided.svg)
*Coalesced access shares one cache line; strided access wastes most of every line it touches.*

---

## 1. The motivation: a problem with no math

After matmul and reduction, transpose looks trivial: swap indices, no FLOPs. That is exactly what makes it the purest test of the **memory system**. The GPU reads and writes global memory in 128-byte lines. When the 32 threads of a warp touch consecutive addresses, the hardware **coalesces** them into one transaction and uses all 128 bytes. When they touch addresses a large stride apart, each lands in its own line and uses only 4 of the 128 bytes, so the effective bandwidth drops by up to 32×. A model counts the gap directly: a coalesced row read is one transaction per warp, a strided column read is 32.

A transpose is stuck between the two: reading row-major input by rows is coalesced, but writing the transposed output means writing *columns*, which is strided (or vice versa). You cannot coalesce both sides at once with a direct copy.

## 2. The naive kernel, and where it leaks

The simple version reads coalesced and writes strided:

```cuda
__global__ void transposeNaive(float* out, const float* in, int W, int H) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < W && y < H)
        out[x * H + y] = in[y * W + x];   // read row (coalesced), write column (strided)
}
```

The read is fine, but each thread's *write* lands `H` floats from its neighbor's, so a warp's 32 writes scatter across 32 cache lines. On an RTX 3080 (compute capability 8.6) this naive transpose reaches about **60 GB/s, only 17%** of a plain copy's 350 GB/s, measured separately; the shipped `transpose.cu` checks correctness only. The arithmetic is zero; the addressing is the whole problem.

## 3. The fix: tile through shared memory

The trick is to **decouple the read pattern from the write pattern** with a shared-memory tile. Load a 32×32 tile from global memory by rows (coalesced) into shared memory; then write the output by rows (also coalesced), reading the tile *transposed*. Shared memory has no coalescing requirement, so the strided access is confined on-chip.

![A flow: load a tile by rows into a green shared tile, then store by rows while reading columns from the tile, with the index-swap load and store lines.](diagrams/02-tiled-transpose.svg)
*Both global accesses are coalesced; the transpose is the index swap `tile[tx][ty]` inside shared memory.*

```cuda
__shared__ float tile[32][33];                 // note the +1 pad (see below)
tile[threadIdx.y][threadIdx.x] = in[y * W + x]; // load by rows: coalesced
__syncthreads();                                // all threads finish writing the
                                                // tile before any reads a column
x = blockIdx.y * 32 + threadIdx.x;             // recompute transposed block coords
y = blockIdx.x * 32 + threadIdx.y;
out[y * H + x] = tile[threadIdx.x][threadIdx.y]; // store by rows; read tile column
```

The `__syncthreads()` barrier is required because every thread must finish writing the tile before any thread reads a *column* of it: without the barrier a lane could read a tile slot a sibling has not yet stored. The two index conventions matter. The *block* coordinates swap (`blockIdx.x` and `blockIdx.y` exchange roles for the output), and *inside* the tile the thread indices swap (`tile[tx][ty]` rather than `tile[ty][tx]`), which is what actually performs the transpose. This is the same load-once-reuse idea as tiled matmul, repurposed to fix coalescing instead of arithmetic intensity.

## 4. The new problem: bank conflicts

Coalescing the global accesses creates a shared-memory hazard. Shared memory is split into **32 banks**, and a 4-byte word at address `A` lives in bank `A % 32`. Reading a *column* of an unpadded `tile[32][32]` means addresses `0, 32, 64, ...`, every one of which is bank 0: all 32 lanes hit one bank, a **32-way conflict** the hardware serializes.

![Two panels: an unpadded tile where a column read hits bank 0 for every row (32-way conflict, red), and a padded tile where the column spreads diagonally across all 32 banks (conflict-free, green).](diagrams/03-bank-conflict.svg)
*One extra column changes the stride from 32 to 33, so a column lands on 32 different banks.*

The fix is to **pad** the tile by one column, `tile[32][33]`. Now a column's addresses are `0, 33, 66, ...`, whose banks are `0, 1, 2, ...`, all distinct. The model confirms the column read goes from a 32-way conflict to conflict-free, at the cost of one extra column (~3% more shared memory):

```text
transpose model
  index-swap == A.T   : True
  global transactions per 32-thread warp:
    coalesced (a row)    : 1   (one 128-byte line)
    strided   (a column) : 32  (one line per thread)
  shared-memory bank conflicts on a column read (lower is better):
    tile[32][32] unpadded: 32-way conflict
    tile[32][33] padded  : 1-way (conflict-free)
```

Build and run the kernel (it verifies correctness), and the CPU model (it prints the counts above):

```bash
nvcc -O3 -arch=sm_75 snippets/transpose.cu -o transpose && ./transpose
python snippets/transpose_model.py
```

## 5. The numbers

Each fix targets one hazard, and they compound. On an RTX 3080 transposing 4096²:

| Implementation | Bandwidth | % of copy |
|---|---|---|
| naive transpose | 60 GB/s | 17% |
| shared memory | 220 GB/s | 63% |
| shared + padding | 310 GB/s | 88% |

These bandwidths were measured separately on an RTX 3080 (compute capability 8.6); the shipped `transpose.cu` only verifies correctness, it does not produce this table.

![A bar chart: naive transpose 60 in red, shared 220, shared+padding 310 in green, against a dashed plain-copy line at 350.](diagrams/04-transpose-perf.svg)
*Tiling fixes global coalescing (3.7×); padding fixes the bank conflict (1.4×); together 5.2× over naive.*

Tiling alone is a **3.7×** win (global coalescing), padding adds **1.4×** (shared-memory banks), and combined they reach **88% of copy bandwidth**, about the practical ceiling for a memory-bound kernel. The transferable lesson: when a kernel does no arithmetic, performance *is* the access pattern, and shared memory is the tool for reshaping it.

---

## Common pitfalls

- **Not swapping the block indices for the output.** Using the input `blockIdx` for the output writes the wrong tile location; `blockIdx.x` and `blockIdx.y` must exchange roles.
- **Not swapping the tile indices.** Reading `tile[ty][tx]` on the way out copies instead of transposing; the swap to `tile[tx][ty]` is the actual transpose.
- **Using the wrong output stride.** Output `B` has `height` columns (the input's rows); indexing it with the input `width` corrupts a rectangular transpose.
- **Forgetting to pad.** An unpadded `tile[32][32]` reintroduces a 32-way bank conflict on the column read; `tile[32][33]` fixes it for ~3% more shared memory.
- **Maxing out the block at 32×32 with a heavy kernel.** 32×32 is 1024 threads, the hard limit; a register-hungry kernel may spill. For complex kernels use 16×16 tiles and a per-thread loop.

---

## Further reading

- Harris, M., *"An Efficient Matrix Transpose in CUDA C/C++"* (NVIDIA, 2013). The canonical walkthrough of tiling and the padding fix, with the same benchmark structure (technical, foundational).
- NVIDIA, *"CUDA C++ Programming Guide — Device Memory Accesses"* (current). The definition of coalescing and the 128-byte transaction model (reference).
- NVIDIA, *"CUDA C++ Best Practices Guide — Shared Memory and Bank Conflicts"* (current). The bank model and padding guidance (technical, reference).
- Volkov, V., *"Better Performance at Lower Occupancy"* (GTC, 2010). Why memory access patterns often matter more than occupancy (technical, historical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 05, Histogram](../05-histogram/index.md)**: the other shared-memory pattern, where it solves contention rather than coalescing.
- **[Post 07, Convolution](../07-convolution/index.md)**: shared-memory tiling again, now with *halo* regions, plus constant memory for the filter.
