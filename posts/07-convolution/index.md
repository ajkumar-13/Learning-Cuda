# 07 - Convolution: constant memory, shared-memory halos, and separable filters

> **TL;DR.** A convolution makes each output pixel a weighted sum of a `K x K` window of the input, and neighboring windows overlap, so a naive kernel reads every pixel up to $K^2$ times. The filter can go in **constant memory**, which serves a warp reading one weight with one broadcast request, and the pixels in a **shared-memory tile** that loads a 16 x 16 output tile plus a one-pixel **halo** once per block. On the author's GTX 1650 the caches already absorb most re-reads of a 3 x 3 filter, so only constant memory helps there (0.224 to 0.208 ms on a 1920 x 1080 image), while at 7 x 7 the tile beats constant memory by 17% and the **separable** form, two 1D passes, is fastest at 0.293 ms.
>
> **Prerequisites:** [post 06](../06-matrix-transpose/index.md).
> **Safe to skip?** Skip it if the reader can already write a tiled stencil kernel with its halo, put a filter in constant memory and say when its reads broadcast, and count what a separable filter saves.
>
> **After reading, you will be able to:**
>
> - Write a 2D stencil kernel from one thread's point of view and count the redundant reads that overlapping windows cause.
> - Put a filter in constant memory with `__constant__` and `cudaMemcpyToSymbol`, and say when a warp's read is one broadcast and when it splits.
> - Size and load a shared-memory tile with its halo, launch the grid by output tiles, and compute the tile's reuse factor.
> - Recognize a separable filter and count the multiply-adds per pixel of its two 1D passes against the 2D filter.
> - Read a measured convolution table and say which fix pays off for which filter size, and why the caches change the answer.

![A 5 by 5 input with a 3 by 3 window around pixel (2, 2) and a dashed window around (2, 3) sharing 6 of its 9 pixels; with a filter of ones times 1/9 they give 5 and 4.](diagrams/01-stencil.svg)

*Each output pixel sums a small window; the next pixel's window reuses six of its nine inputs.*

---

## 1. The question: every output pixel reads a window

A **convolution** over an image computes each output pixel from a small neighborhood of the input, weighted by a fixed filter (also called a mask); it blurs, sharpens and finds edges. With a `K x K` filter of radius $r = (K - 1) / 2$, the output at row $y$ and column $x$ is

$$\text{out}[y][x] = \sum_{i=-r}^{r}\sum_{j=-r}^{r} \text{in}[y+i][x+j] \cdot \text{mask}[i+r][j+r]$$

with a pixel outside the image counted as zero (**zero padding**). Strictly, this sum is a *cross-correlation*: a true convolution flips the filter first, reading `mask[r-i][r-j]`. The two agree for a symmetric filter, such as the box blur this post uses, and differ for an asymmetric one. Deep-learning libraries compute the unflipped sum and call it convolution (cuDNN offers both, as its `CUDNN_CROSS_CORRELATION` and `CUDNN_CONVOLUTION` modes), and this post does the same. The central processing unit (CPU) model of this post, `snippets/convolution_model.py`, checks both statements, and its own tiled kernel, on a random image:

```text
1. the tiled kernel against a direct reference (3x3 box filter, zero padding)
  image 70x40, tile 16x16: tiled == reference: True
  box filter: true convolution (filter flipped) == cross-correlation: True
  asymmetric filter: true convolution (filter flipped) == cross-correlation: False
```

The cost lies in the reads. In the figure at the top, two neighboring outputs share six of their nine input pixels; every interior pixel lies in nine windows, so a kernel that reads each window from global memory reads each pixel up to nine times, and up to 49 times for a 7 x 7 filter. The question is where the filter and the pixels should live so that each value leaves global memory as few times as possible.

## 2. The naive kernel, and its two redundant reads

On the graphics processing unit (GPU), one thread computes one output pixel. Its 2D global index is the pair `(row, col)` of [post 03](../03-matrix-multiplication/index.md), with `col = blockIdx.x * blockDim.x + threadIdx.x` and `row` the same form in `.y`; the image `in: [H, W]` is stored row by row, so pixel `(row, col)` is `in[row * W + col]`. Thread `(row, col)` of `convNaive` in `snippets/convolution.cu` sums its window:

```cuda
template <int K>
__global__ void convNaive(const float* in, const float* mask, float* out, int W, int H) {
    constexpr int R = K / 2;                     // the filter radius
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j) {
                int r = row + i - R, c = col + j - R;
                if (r >= 0 && r < H && c >= 0 && c < W)
                    sum += in[r * W + c] * mask[i * K + j];
            }
        out[row * W + col] = sum;
    }
}
```

The filter size `K` is a template parameter, fixed at compile time, and the program runs `K = 3` and `K = 7`. The guard `row < H && col < W` drops the surplus threads of the last blocks, and the four-sided test on `r` and `c` is the zero padding.

Each thread makes two kinds of read, and both repeat. The **filter**, `mask: [K, K]`, sits in global memory as `d_mask`, and at each tap `(i, j)` all the threads of a warp read the same weight. The **pixels** overlap: over one 16 x 16 block of outputs and a 3 x 3 filter, the threads make $16^2 \times 3^2 = 2{,}304$ pixel reads, but the block touches only $18^2 = 324$ distinct pixels. The next two sections give each kind of read its own memory.

## 3. The filter: constant memory

**Constant memory** is a 64 KB region of device memory, read-only for a kernel and cached on each streaming multiprocessor (SM); the CUDA Programming Guide gives an 8 KB constant-cache working set per SM on compute capability 7.5, the GTX 1650's. A variable in it is declared at file scope with `__constant__` and a size fixed at compile time:

```cuda
__constant__ float c_mask[MAX_K * MAX_K];   // file scope, read-only, broadcast to a warp
__constant__ float c_row[MAX_K];            // the separable filter's two 1D factors
__constant__ float c_col[MAX_K];
```

`MAX_K` is 7, so `c_mask` holds up to 49 weights (`c_row` and `c_col` belong to section 5). The host fills it with `cudaMemcpyToSymbol`, which names the symbol, not with `cudaMemcpy`, which wants a device address:

```cuda
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, box.data(), K * K * sizeof(float)));   // not cudaMemcpy
```

`convConst` is `convNaive` with the tap reading `c_mask[i * K + j]` and the `mask` parameter gone. The Programming Guide's rule for how constant memory serves a warp: a request splits into as many separate requests as there are different addresses in it. The index `i * K + j` depends only on the loop counters, so all 32 lanes of a warp read one address with one request, **broadcast** to every lane. A pixel is the opposite case: `row * W + col` differs in every lane, so the same read from constant memory would split into 32 requests served one after another.

![Top: the 32 lanes of a warp all read c_mask[4] from the constant cache with one broadcast request. Bottom: 32 lanes reading 32 different pixel addresses, which constant memory would serve as 32 requests in turn.](diagrams/02-constant-broadcast.svg)

*Constant memory suits a value that every lane reads at once, and only such a value.*

In the compiled code (`cuobjdump -sass`), `convNaive<3>` holds 18 global load instructions, nine for pixels and nine for weights, and `convConst<3>` 9, its weights read straight from the constant bank. The 64 KB hold 16,384 floats, so the largest odd square filter that fits is 127 x 127:

```text
4. constant memory
  64 KB holds 16,384 floats; the largest odd filter is 127x127 (16,129 weights)
```

## 4. The pixels: a shared-memory tile with a halo

The overlapping pixels go into shared memory, as post 03's tiles did, with one difference: a stencil's input tile is *larger* than its output tile. The threads on the edge of a 16 x 16 output tile need pixels one step beyond it, which belong to the neighboring tiles. That border, as wide as the filter radius, is the **halo**, and the input tile is $16 + 2r = 18$ pixels on a side. With `TILE` set to 16, `convTiled` launches one thread per input pixel, $18 \times 18 = 324$ threads; each loads one pixel into `s: [18, 18]`, and after the barrier only the inner $16 \times 16 = 256$ compute:

```cuda
template <int K>
__global__ void convTiled(const float* in, float* out, int W, int H) {
    constexpr int R = K / 2, BLK = TILE + K - 1;     // BLK = 18 for a 3 x 3 filter
    __shared__ float s[BLK][BLK];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx - R;            // shifted back by the halo radius
    int row = blockIdx.y * TILE + ty - R;
    s[ty][tx] = (row >= 0 && row < H && col >= 0 && col < W) ? in[row * W + col] : 0.0f;
    __syncthreads();                                 // the whole tile is loaded before any thread reads it

    if (tx >= R && tx < BLK - R && ty >= R && ty < BLK - R) {   // inner threads only
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < K; ++i)
            #pragma unroll
            for (int j = 0; j < K; ++j)
                sum += s[ty + i - R][tx + j - R] * c_mask[i * K + j];
        int oc = blockIdx.x * TILE + (tx - R), orow = blockIdx.y * TILE + (ty - R);
        if (orow < H && oc < W) out[orow * W + oc] = sum;
    }
}
```

Three details carry the halo. The load index steps between blocks by `TILE`, not `blockDim.x`, and is shifted back by the radius, so neighboring blocks' input tiles overlap by the halo. A thread whose pixel lies outside the image stores 0, the zero padding, so the compute loop needs no bounds test. And the `__syncthreads()` makes the whole tile present before any thread reads neighbors that other threads loaded. The grid therefore counts *output* tiles, while the tiled kernel's block is the input tile, `dim3(TILE + K - 1, TILE + K - 1)`; this fragment of the program's launch function sizes the grid for every kernel:

```cuda
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);   // grid sized by OUTPUT tiles
```

For the 1920 x 1080 image that is $120 \times 68 = 8{,}160$ blocks, of 256 threads, or of 324 for `convTiled<3>`.

![An 18 by 18 shared tile: 256 inner threads load and compute, a one-pixel halo of 68 threads only loads, a barrier between; reuse 7.1, 16.0 and 25.9 for 3 by 3, 5 by 5 and 7 by 7 filters.](diagrams/03-halo-tile.svg)

*The tile replaces a pixel's many reads from global memory with one load per tile; the halo is the price.*

The model counts what a tile saves and what it costs:

```text
2. one 16x16 output tile: global reads, and the threads of its block (one per tile pixel)
  filter  naive reads  tile reads     reuse  threads computing
  3x3         2,304   324 (18x18)    7.1x  256 of 324 (79%)
  5x5         6,400   400 (20x20)   16.0x  256 of 400 (64%)
  7x7        12,544   484 (22x22)   25.9x  256 of 484 (53%)
```

The **reuse factor**, naive reads over tile reads, is $16^2 K^2 / (15 + K)^2$, and it grows with the window: 7.1 for 3 x 3, 25.9 for 7 x 7. So does the cost: the halo threads load and then idle, 21% of the block at 3 x 3 and 47% at 7 x 7, and a 324-thread block is 11 warps, the last holding 4 threads. These counts are derived by the model; section 6 measures what they are worth.

## 5. Separable filters

Some filters are an outer product of a column vector and a row vector: the 3 x 3 box blur is $1/3$ down a column times $1/3$ along a row, and a Gaussian blur factors the same way. Such a **separable** filter runs as two 1D passes, along every row into an intermediate image `tmp`, then along every column of it, each costing $K$ multiply-adds per pixel: $2K$ in all instead of $K^2$. In `convolution.cu`, thread `(row, col)` of `convRows` sums `in[row * W + c] * c_row[j]` over the $K$ columns `c` around `col`, and `convCols` does the same down the column of `tmp` with `c_col`. The model counts the multiply-adds and checks that the two passes reproduce the 2D blur, padding included:

```text
3. multiply-adds per output pixel
  3x3: 9 as one 2D filter, 6 as a row pass and a column pass
  5x5: 25 as one 2D filter, 10 as a row pass and a column pass
  7x7: 49 as one 2D filter, 14 as a row pass and a column pass
  3x3 box = (1/3 row) then (1/3 column): True
```

The saving has a price. `tmp` is as large as the output, written by the first pass and read by the second, so a separable filter moves 16 bytes per pixel through global memory instead of 8, and pays a second launch. Podlozhnyuk's NVIDIA whitepaper (2007) stages each pass through shared memory; this post's passes read global memory directly.

## 6. The numbers

`snippets/convolution.cu` runs every kernel on a 1920 x 1080 image with the 3 x 3 and the 7 x 7 box filter. It first runs a copy kernel untimed for half a second, because this laptop GPU idles at a low clock and needs a few hundred milliseconds of work to reach its 1,740 MHz boost clock, and it times the GPU rows before the CPU rows, which heat the laptop. Each GPU row is the median of 7 rounds of 20 back-to-back launches timed with CUDA events, and the whole set of GPU rows runs five times, each row printing its fastest pass: now and then the laptop cuts the GPU's power budget for a second or two (`nvidia-smi` then reads about 20 W and an SM clock of 825 to 990 MHz), and a row timed inside such an episode runs 2 to 8 times slower. The copy kernel, one read and one write per pixel, is the practical ceiling. The CPU rows time the same loop on the host, on one thread and with OpenMP on the 12 hardware threads of the Intel Core i7-9750H. Every output is checked against the one-thread result, and GB/s counts the 8 bytes per pixel that any filter must move. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5)
zero padding; GPU rows: fastest of 5 passes, each the median of 7 rounds of 20 launches; CPU rows: best of 3
3 x 3 box filter, image 1920 x 1080, 16588800 bytes moved per pass
method                   time (ms)    GB/s   vs 1 CPU thread   max abs err
CPU, 1 thread               15.371     1.1              1.0x   reference
CPU, OpenMP 12 threads       5.431     3.1              2.8x   0.0e+00 ok
copy (ceiling)               0.127   130.8                 -   exact copy
naive, global filter         0.224    74.2             68.7x   4.6e-05 ok
constant filter              0.208    79.7             73.9x   4.6e-05 ok
tiled + halo                 0.291    56.9             52.8x   4.6e-05 ok
separable, 2 passes          0.266    62.4             57.8x   4.6e-05 ok
7 x 7 box filter, image 1920 x 1080, 16588800 bytes moved per pass
method                   time (ms)    GB/s   vs 1 CPU thread   max abs err
CPU, 1 thread               97.519     0.2              1.0x   reference
CPU, OpenMP 12 threads      24.663     0.7              4.0x   0.0e+00 ok
copy (ceiling)               0.128   129.3                 -   exact copy
naive, global filter         1.019    16.3             95.7x   4.6e-05 ok
constant filter              0.740    22.4            131.8x   4.6e-05 ok
tiled + halo                 0.617    26.9            158.0x   4.6e-05 ok
separable, 2 passes          0.293    56.6            332.6x   9.2e-05 ok
```

Every number in this output is measured on one machine, and the reader's will give other digits. Fourteen runs of this program, each started on an idle GPU at 74 to 81 degrees C, agreed to within 5% in every GPU row. None is a full-clock run: under sustained load `nvidia-smi` reports the card at its software power cap of 35 W, with the SM clock between about 1,200 and 1,740 MHz at 70 degrees C as at 85, so a cooler start does not change the table. An earlier draft quoted a table from an RTX 3080 that the author can no longer rerun; this one is the shipped harness's output.

![Effective bandwidth, 0 to 160 GB/s, copy line at about 130. 3 by 3: naive 74.2, constant 79.7, tiled 56.9, separable 62.4. 7 by 7: naive 16.3, constant 22.4, tiled 26.9, separable 56.6.](diagrams/04-conv-perf.svg)

*At 3 x 3 constant memory is the one fix that helps; at 7 x 7 the tile helps and the separable form wins.*

Four readings follow.

First, the caches already do much of the tile's work. If every read of `convNaive<3>` came from dynamic random-access memory (DRAM), a pass would move $(9 + 1) \times 4$ bytes per pixel, 82.9 MB, and 0.224 ms would mean 370 GB/s, more than twice the card's 160 GB/s theoretical peak; the 7 x 7 version would mean 407 GB/s. Most re-reads of a neighbor's pixel therefore hit in the SM's L1 cache or the 1,024 KB L2.

Second, constant memory helps at both sizes, by 7% at 3 x 3 (0.224 to 0.208 ms; 5 to 9% over repeated runs) and by 27% at 7 x 7 (1.019 to 0.740 ms): it halves the loads the kernel issues, and at 49 taps a pixel that half is large.

Third, the tile pays only for a large window. At 3 x 3 it is the slowest GPU kernel, 1.4 times the constant-memory time: the caches leave it little to save, and it pays for idle halo threads, a barrier and a partly empty warp per block (the measurement does not separate these costs). At 7 x 7 it beats constant memory by 17% (0.740 to 0.617 ms; 16 to 19% over repeated runs). The separable form follows its traffic and its multiply-adds: at 3 x 3 its two passes move 33.2 MB in 0.266 ms, 125 GB/s, close to the copy kernel's 130.8, so the doubled traffic bounds it; at 7 x 7, with 14 multiply-adds instead of 49, it is 2.5 times faster than constant memory and 2.1 times faster than the tile.

Fourth, every GPU kernel is 53 to 333 times faster than one CPU thread running the plain loop, and OpenMP gains only 2.8 to 4.0 times on this laptop; neither CPU row is tuned. A library such as cuDNN goes further, choosing per problem among algorithms that include implicit matrix multiplication, the fast Fourier transform and Winograd's minimal filtering; it is not installed on the author's machine, so this post gives it no number.

## 7. Make it run

The CPU model needs NumPy and no GPU. From the series root:

```bash
uv run python posts/07-convolution/snippets/convolution_model.py
```

It prints the four blocks quoted in sections 1, 3, 4 and 5. The kernel needs an NVIDIA GPU and the CUDA Toolkit, and builds from the series root with `-arch` set to the card's compute capability; `-Xcompiler /openmp` turns on the OpenMP row with Microsoft's compiler (`-Xcompiler -fopenmp` with g++), and without it that row is skipped:

```bash
nvcc -O3 -arch=sm_75 -Xcompiler /openmp posts/07-convolution/snippets/convolution.cu -o convolution && ./convolution
```

On Windows the program is `convolution.exe`, and `nvcc` may need `-ccbin` to find `cl.exe`; [RUNNING.md](../../RUNNING.md) covers both and the `-arch` of other cards. It prints section 6's tables in about five seconds and exits with status 1 if any output disagrees with the CPU reference.

## 8. What can go wrong?

Each mistake below was reproduced with a one-line change to `convolution.cu`, and the program's check caught every one.

- **A block the size of the output tile.** Launching `convTiled` with 16 x 16 threads leaves 14 x 14 inner threads, so two rows and two columns of every output tile are never written; they keep the harness's not-a-number (NaN) fill, and both tiled rows print `nan MISMATCH`.
- **A grid sized by the block.** `(W + 17) / 18` blocks instead of `(W + 15) / 16` leave the last rows and columns of every output unwritten, and those rows print *faster* times: a benchmark needs its correctness check.
- **No barrier after the load.** Without the `__syncthreads()`, threads read neighbors not yet loaded; the maximum error was 99 at 3 x 3 and 130 at 7 x 7, on pixel values up to 255.
- **Half a boundary check.** `r < H && c < W` without `r >= 0 && c >= 0` reads before the array or wraps into the previous row; the edges were off by up to 83, with no error reported.
- **`cudaMemcpy` into a `__constant__` array.** `cudaMemcpy(c_mask, ...)` fails with `invalid argument`; unchecked, the kernels would run with a filter that was never copied.

---

## Common pitfalls

1. **Assuming the tile always wins.** It cut the 7 x 7 time by 17% and left the 3 x 3 kernel at 1.4 times the time of constant memory alone on this card; measure at the filter sizes that matter.
2. **Putting per-thread data in constant memory.** A warp's constant read splits into one request per distinct address, so an array indexed by `row` or `col` is read one lane at a time.
3. **Treating a separable filter as free.** Its intermediate image doubles the traffic; at 3 x 3 the two passes lost to one 2D pass.
4. **Overflowing constant memory.** Its 64 KB hold 16,384 floats, so no odd filter beyond 127 x 127 fits.
5. **Mixing up convolution and cross-correlation.** An asymmetric filter gives different results under the two, so a reference must flip, or not, exactly as the kernel does.

---

## Further reading

- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, sections "Constant Memory" and "Technical Specifications per Compute Capability", consulted October 2026. `__constant__`, `cudaMemcpyToSymbol`, one request per distinct address, and the 64 KB and 8 KB figures.
- Hwu, W. W., Kirk, D. B. and El Hajj, I., *Programming Massively Parallel Processors: A Hands-on Approach*, 4th ed. (Morgan Kaufmann, 2022), chapter 7, "Convolution: an introduction to constant memory and caching". The naive kernel, constant memory, tiles with halos, and tiles that leave the halo to the caches.
- Podlozhnyuk, V., "Image Convolution with CUDA" (NVIDIA whitepaper, June 2007). Separable convolution in two passes, each staged through shared memory.
- NVIDIA, *cuDNN Documentation*, consulted October 2026. The convolution modes and the forward algorithms a production library chooses among.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 08 - Parallel scan: breaking the dependency chain of a prefix sum](../08-parallel-scan/index.md):** the last pattern of Part II, where each output depends on every earlier input instead of a fixed window.
- **[Post 14 - Kernel fusion: the fastest kernel is the one never launched](../14-kernel-fusion/index.md):** the traffic argument of the separable filter's intermediate image, generalized: an intermediate written to global memory only to be read back is wasted bandwidth, and fusing the kernels removes it.
