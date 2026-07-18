# 07 · Convolution — constant memory, shared-memory halos, and separable filters

> **TL;DR.** Convolution makes each output pixel a weighted sum of a small **neighborhood** of the input under a filter, and adjacent windows overlap, so a naive kernel re-reads the same input many times. Two specialized memories fix it: the small read-only **filter** goes in **constant memory**, which broadcasts one value to a whole warp in a single transaction, and the **pixels** go in a **shared-memory tile** that loads an 18×18 input for a 16×16 output because edge threads need a one-pixel **halo**. With both, the kernel loads each input value once instead of re-reading it. A final trick, **separable filters**, turns a K×K filter into two cheap 1D passes.
>
> **After reading this you will be able to:**
> - Express a convolution as a 2D stencil and see why neighbors cause redundant reads.
> - Put a filter in constant memory and explain the broadcast that makes it fast.
> - Size and load a shared-memory tile with a halo, and compute the reuse factor.
> - Recognize separable filters and the `K²` to `2K` arithmetic saving.

![A 5x5 input with a green 3x3 neighborhood, multiplied by a 3x3 filter, producing one output pixel.](diagrams/01-stencil.svg)
*Each output pixel is a weighted sum of a small window; adjacent windows overlap, which is the reuse the kernel must exploit.*

---

## 1. The motivation: seeing is a stencil

If matrix multiply is the *thinking* of deep learning (dense layers), convolution is the *seeing* (CNNs): it detects edges and textures by looking at local neighborhoods. Each output pixel reads a `K×K` window of the input and multiplies it by a filter:

$$\text{out}[y][x] = \sum_{i=-r}^{r}\sum_{j=-r}^{r} \text{in}[y+i][x+j]\cdot\text{mask}[i+r][j+r]$$

with radius `r` (1 for a 3×3 filter). Strictly, this sum without flipping the filter is *cross-correlation*; true convolution flips the kernel first, but the field (and every deep-learning framework) calls this operation convolution, so we will too. The problem is reuse: the pixel to the right shares two of three columns with this one, so a naive kernel re-reads most of the input. A CPU model counts it for a 16×16 output tile with a 3×3 filter: **2,304** naive global reads versus **324** if the input is loaded once, a **7.1× reuse factor**.

```text
convolution model
  tile=16x16, filter=3x3 (radius 1)
  tiled == reference   : True
  naive global reads   : 2,304  (re-reads each 3x3 neighborhood)
  tiled global reads    : 324  (load the 18x18 tile once)
  reuse factor         : 7.1x
  separable mul-adds    : 9 per pixel non-separable vs 6 separable
```

Build and run the kernel and the CPU model from the post directory:

```bash
nvcc -O3 -arch=sm_75 snippets/convolution.cu -o convolution && ./convolution
python snippets/convolution_model.py
```

Two distinct things are read redundantly: the **filter** (the same small array by every thread) and the **pixels** (overlapping windows). Each gets its own fix.

## 2. The filter: constant memory

Every thread reads the same filter weight at the same moment. That is the textbook case for **constant memory**: a 64 KB read-only space whose cache **broadcasts** one value to all 32 lanes of a warp in a single transaction, instead of 32 separate reads.

![Two panels: without constant memory 32 threads each issue a request for mask[0]; with constant memory one read is broadcast to all.](diagrams/03-constant-broadcast.svg)
*The same weight read by 32 threads becomes one broadcast, not 32 transactions.*

```cuda
__constant__ float c_mask[MASK * MASK];                 // file scope, read-only
// host: fill it with cudaMemcpyToSymbol, not cudaMemcpy
cudaMemcpyToSymbol(c_mask, h_mask, sizeof(h_mask));
```

It is declared at file scope, sized at compile time, and filled with `cudaMemcpyToSymbol`. On its own, moving the filter to constant memory takes the naive 2.1 ms kernel to 1.4 ms.

## 3. The pixels: a shared-memory tile with a halo

The bigger win is loading the overlapping pixels into shared memory once. But unlike matmul, the input tile is *larger* than the output tile: to compute a 16×16 output with a 3×3 filter, the edge threads need a one-pixel border from the neighbors, the **halo**. So the input tile is `16 + 2·radius = 18` on a side.

![An 18x18 input tile with a green inner 16x16 output region and a one-cell halo ring loaded but not computed.](diagrams/02-halo-tile.svg)
*Load `output + 2·radius` into shared memory; the outer ring only loads, the inner threads compute.*

```cuda
#define TILE 16
#define BLK  (TILE + MASK - 1)        // 18 for a 3x3 filter
__shared__ float s[BLK][BLK];
int col = blockIdx.x * TILE + threadIdx.x - R;   // shifted by the halo radius
int row = blockIdx.y * TILE + threadIdx.y - R;
s[ty][tx] = inBounds ? in[row * W + col] : 0.0f; // all 18x18 threads load
__syncthreads();
if (innerThread)                                  // only the 16x16 inner threads
    sum += s[ty + i - R][tx + j - R] * c_mask[i * MASK + j];
```

The block launches `18×18` threads: all of them load one pixel, but only the inner `16×16` compute. The grid is sized by the **output** tile (16), not the block (18), a frequent source of bugs. The reuse climbs with filter size, since a bigger window overlaps more: a 7×7 filter on a 16×16 tile reuses each pixel about 26×.

## 4. The numbers

Each fix removes one kind of redundant traffic, and they stack:

| Implementation | Time (1920×1080, 3×3) | Speedup |
|---|---|---|
| CPU (1 thread) | 14.7 ms | 1× (baseline) |
| CPU (8 threads) | 3.2 ms | 4.6× (reference) |
| GPU naive | 2.1 ms | 7× |
| + constant memory | 1.4 ms | 11× |
| + shared memory (halo) | 0.9 ms | 17× |
| cuDNN | 0.3 ms | 50× |

All speedups are versus the single CPU thread (14.7 ms); the 8-thread row is shown only for reference. These are RTX 3080 measurements (compute capability 8.6). The shipped `convolution.cu` launches only the final `convTiled` kernel to check correctness; the per-row times above were measured separately on that GPU.

![A time bar chart: CPU 8-thread 3.2, naive 2.1, constant 1.4, shared 0.9 in green, cuDNN 0.3 in blue.](diagrams/04-conv-perf.svg)
*Constant memory removes redundant filter reads; shared memory removes redundant pixel reads.*

The shared-memory kernel reaches **17×** the single-threaded CPU and lands within ~3× of cuDNN, which adds register blocking and algorithmic transforms (Winograd, FFT) on top.

## 5. Separable filters

A last, free win for many filters. A `K×K` filter that factors into a column vector times a row vector (a box blur, a Gaussian) can be applied as **two 1D passes** instead of one 2D pass. That turns `K²` multiply-adds per pixel into `2K`: 9 to 6 for a 3×3, and 49 to 14 for a 7×7. The cost is an intermediate buffer and a second kernel launch, almost always worth it for filters of 5×5 and up.

---

## Common pitfalls

- **Making `blockDim` equal to the output tile.** The block must be `TILE + 2·radius` on a side so the halo loads; a 16×16 block silently drops the border and corrupts edge outputs.
- **Sizing the grid by the block, not the tile.** The grid counts *output* tiles (`TILE`), so use `(width + TILE - 1) / TILE`; using the 18-wide block over-launches and misaligns.
- **Bank conflicts on power-of-two tiles.** An 18-wide tile happens to avoid them, but a 32-wide shared tile reintroduces the column conflict from [Post 06](../06-matrix-transpose/index.md); pad it `[BLK][BLK+1]`.
- **Overflowing constant memory.** It is 64 KB, which holds 16,384 floats, so a filter past ~127×127 will not fit and you need texture or shared memory instead. (In practice you switch far sooner, once the filter stops fitting the small constant cache.)
- **Half a boundary check.** `col < width` without `col >= 0` reads before the array; zero-pad both ends of every dimension.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — Constant Memory"* (current). The reference for `__constant__` and `cudaMemcpyToSymbol` (reference).
- Kirk, D. & Hwu, W., *"Programming Massively Parallel Processors"*, ch. 7 (Convolution). The canonical treatment of halos and tiling (technical, foundational).
- Podlozhnyuk, V., *"Image Convolution with CUDA"* (NVIDIA, 2007). The separable-convolution implementation in detail (technical).
- NVIDIA, *"cuDNN Developer Guide"* (current). What a production convolution adds (Winograd, FFT, implicit GEMM) (reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 06, Matrix transpose](../06-matrix-transpose/index.md)**: the coalescing and bank-conflict rules this post's tile loads depend on.
- **[Post 08, Parallel scan](../08-parallel-scan/index.md)**: the last core pattern, where each output depends on *all* previous inputs, not just a fixed neighborhood.
