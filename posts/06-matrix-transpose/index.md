# 06 - Matrix transpose: coalescing, shared memory, and bank conflicts

> **TL;DR.** A transpose does no arithmetic, yet on the author's GTX 1650 the naive kernel moves data at 30% of a plain copy's bandwidth, because a warp's 32 writes down a column land in 32 different 128-byte lines where a row's land in one. Staging a 32 x 32 tile in **shared memory** makes both global accesses rows and runs 1.9 times faster, but its column read hits one shared-memory **bank** 32 times. One column of **padding**, `[TILE][TILE + 1]`, removes that conflict and brings the transpose to 89.5% of the copy, 3.0 times the naive kernel.
>
> **Prerequisites:** [post 03](../03-matrix-multiplication/index.md).
> **Safe to skip?** Skip it if the reader can already count the lines and sectors a warp's access touches, write a tiled transpose with both index swaps and its barrier, and find and pad away a shared-memory bank conflict.
>
> **After reading, you will be able to:**
>
> - Explain coalesced against strided access by counting the 128-byte lines and 32-byte sectors one warp's access touches.
> - Diagnose why the naive transpose's strided writes waste most of the bytes each sector moves.
> - Stage a 32 x 32 tile through shared memory so that both global accesses of a transpose are coalesced, with the block and tile indices swapped.
> - Place the barrier between the tile's load and its transposed read, and say what it protects.
> - Compute the bank of a shared-memory word, recognize a 32-way bank conflict, and remove it by padding the tile to `[TILE][TILE + 1]`.
> - Measure a transpose against a copy kernel and read each fix as a share of the copy's bandwidth.

![One warp reads a row of A into one 128-byte line of four fully used sectors, and writes a column of B into 32 lines and 32 sectors, using 128 of the 1,024 bytes it moves; bars compare the two byte counts.](diagrams/01-coalesced-vs-strided.svg)

*A row costs the warp one line; a column costs it 32 lines, and seven eighths of every sector it moves are unused.*

---

## 1. The question: a kernel with no arithmetic

The transpose of `A: [M, N]` is `B: [N, M]` with `B[col][row] = A[row][col]`. There is nothing to compute: every element is read once and written once, so a `4096 x 4096` matrix of floats costs 128 MiB of traffic and nothing else. A plain copy moves the same bytes, and on the author's GTX 1650 it reaches 139.1 GB/s, 87% of the card's 160 GB/s theoretical peak. The naive transpose reaches 41.5 GB/s.

The difference is the order of the bytes: the copy writes `B` along its rows, the transpose down its columns. This post explains why the order matters to the graphics processing unit (GPU), fixes it with post 03's shared-memory tile, used here to change the order of global accesses rather than their number, and fixes the on-chip problem the tile creates with one extra column.

## 2. How a warp's accesses become memory transactions

The 32 threads of a warp issue each load or store together, and the hardware gathers their 32 addresses into as few memory transactions as it can. The *CUDA C++ Programming Guide* ("Device Memory Accesses") says device memory is accessed in transactions of 32, 64 or 128 bytes, and the *CUDA C++ Best Practices Guide* ("Coalesced Access to Global Memory") states the rule for compute capability 6.0 and later, the GTX 1650's 7.5 included: the warp's accesses "coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp". A 32-byte transaction is a **sector**; four consecutive sectors make a 128-byte line.

When the 32 lanes read 32 consecutive floats, the access is **coalesced**: one line, four sectors, every byte asked for. When they read 32 floats a long stride apart, such as a column of a row-major matrix, whose neighbors are `N` floats apart, every lane's 4 bytes sit in their own line and their own sector. The central processing unit (CPU) model of this post, `snippets/transpose_model.py`, counts both for one warp:

```text
1. one warp, 32 lanes of 4-byte floats, A: [4096, 4096] row-major
                       128-byte lines  32-byte sectors  sector bytes    used
  a row (coalesced)                 1                4           128 100.0%
  a column (strided)               32               32         1,024  12.5%
```

These counts are derived by the model. A column costs the warp 8 times the bytes of a row for the same 128 useful ones, and 32 lines instead of one; the figure at the top draws both. The Best Practices Guide gives the same arithmetic for a stride of 2, "50% of load/store efficiency". A transpose cannot escape by choosing its loop order: a row of `A` is a column of `B`, so one side is always strided.

## 3. The naive kernel, and where it leaks

Each kernel in this post gives one block a `32 x 32` tile of the matrix, `TILE` being 32. Following Harris ("An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 18 February 2013), the block has `32 x 8` threads, `BLOCK_ROWS` being 8, and each thread moves four elements, rows `row`, `row + 8`, `row + 16` and `row + 24`, so that its index arithmetic is paid once for four elements. The column `col` is the global index `blockIdx.x * blockDim.x + threadIdx.x`, with `blockDim.x` equal to `TILE`; the first row is `blockIdx.y * TILE + threadIdx.y`, because 8 rows of threads cover 32 rows of the matrix. Each row of threads is one warp. The naive kernel from `snippets/transpose.cu`:

```cuda
__global__ void transposeNaive(float* B, const float* A, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < M && col < N)
            B[col * M + (row + j)] = A[(row + j) * N + col];   // strided write
}
```

Thread `(row, col)` reads `A[row + j][col]`, so a warp reads 32 neighbors in a row of `A`: coalesced. It writes `B[col][row + j]`, so the warp's 32 writes go down a column of `B`, `M` floats apart: 32 sectors per store where 4 would do. Section 6 measures 41.5 GB/s, 30% of the copy kernel, which is the same loop writing `B[(row + j) * N + col]`.

The measured penalty is smaller than the model's counts suggest. If every sector went to memory, a warp's load and store would move $128 + 1{,}024 = 1{,}152$ bytes in the naive kernel against $128 + 128 = 256$ in the copy, 4.5 times as many; the copy is 3.4 times faster ($3.235 / 0.965$). The model counts one warp's store alone, but the block's other warps write rows `row + 1`, `row + 2` and so on into the same sectors of `B`, and together the block fills each 128-byte line of its tile of `B`. The likely reading, which this post does not measure, is that the 1,024 KB level-2 (L2) cache of the GTX 1650 merges those partial writes before they reach memory, and that the naive kernel pays mostly for the 32 sector writes each store sends to the L2 cache; a profiler's sector counts (post 11) would settle it.

The block shape follows Volkov ("Better Performance at Lower Occupancy", GPU Technology Conference, 22 September 2010): bandwidth needs enough bytes in flight, and a thread that issues several independent loads before it uses any supplies them as several threads would. The tiled kernels of section 4 do that, four loads of `A` back to back in the compiled code (`cuobjdump -sass`); in the copy and the naive kernel, where `A` and `B` might overlap as far as the compiler knows, each load waits for the previous element's store, and the threads alone hide the latency: four blocks of 256 threads fill an SM's 1,024 thread slots.

## 4. The fix: turn the tile in shared memory

The fix decouples the read pattern from the write pattern. A block reads its tile of `A` by rows into a shared-memory array `tile: [TILE, TILE]`, then writes the tile of `B` by rows, reading `tile` by columns. Shared memory sits on the streaming multiprocessor (SM), and its cost depends on banks (section 5), not on lines and sectors, so the strided access moves to the memory that can afford it:

```cuda
__global__ void transposeShared(float* B, const float* A, int M, int N) {
    __shared__ float tile[TILE][TILE];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < M && col < N)
            tile[threadIdx.y + j][threadIdx.x] = A[(row + j) * N + col];
    __syncthreads();
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < N && col < M)
            B[(row + j) * M + col] = tile[threadIdx.x][threadIdx.y + j];
}
```

Two swaps make it a transpose, and the figure below draws them on a `4 x 4` tile. The **block indices swap**: the block that read the tile at block row `blockIdx.y`, block column `blockIdx.x` of `A` writes the tile at block row `blockIdx.x`, block column `blockIdx.y` of `B`, so after the barrier `col` comes from `blockIdx.y` and `row` from `blockIdx.x`. The **tile indices swap**: the load writes `tile[threadIdx.y + j][threadIdx.x]` and the store reads `tile[threadIdx.x][threadIdx.y + j]`, so a warp that stores a row of `B` reads a column of `tile`. The store's guard compares `row` with `N` and `col` with `M`, because `B` has `N` rows of `M` elements.

![A 4 by 4 tile of A holding 0 to 15; a warp loads row 0 into row 0 of the shared tile; after a purple __syncthreads() bar it reads column 0 of the tile, 0 4 8 12, and writes it as row 0 of B's tile.](diagrams/02-tiled-transpose.svg)

*Both global accesses are rows; the transpose happens inside shared memory, behind the barrier.*

The `__syncthreads()` protects the tile's hand-over between warps. A warp that stores a row of `B` reads a column of `tile`, whose 32 values were written by all eight warps of the block, four rows each; the barrier makes every thread finish its loads before any thread reads. Without it a warp can read a slot that its owner has not yet written. The tile is filled and read once, so the kernel needs one barrier where post 03's tiled multiply, which refills its tiles in a loop, needs two. The model runs the same indexing thread by thread on a matrix whose sides are not multiples of 32:

```text
3. the tiled transpose on the CPU, A: [70, 45], blocks of 32 x 8 threads
  grid 2 x 3 blocks, 4 elements per thread
  every element of B written exactly once: True
  B == A.T: True
```

## 5. The new problem: bank conflicts

Shared memory is split into 32 **banks**. On compute capability 7.5, as on 5.0 and later, successive 4-byte words go to successive banks, so word `w` lives in bank `w % 32`, and each bank delivers 32 bits per clock (the Best Practices Guide, "Shared Memory and Memory Banks"). A warp's access takes one pass when its 32 words lie in 32 different banks. When several lanes ask for *different* words of one bank, "the accesses are serialized": $k$ distinct words in one bank take $k$ passes, a $k$-way **bank conflict**. Lanes that read the same word share it at no cost.

In `tile[TILE][TILE]`, element `tile[r][c]` is word `32 * r + c`, in bank `c`. The load writes a row, `c` from 0 to 31: 32 banks, one pass. The store reads a column, `r` from 0 to 31 with `c` fixed: 32 words in bank `c`, a 32-way conflict that turns one read into 32 passes. The fix makes each row one word longer:

```cuda
__shared__ float tile[TILE][TILE + 1];                 // one column of padding
```

Now `tile[r][c]` is word `33 * r + c`, in bank `(r + c) % 32`, and a column's 32 elements fall in 32 different banks. The pad column is never read or written; it only shifts each row one bank along. The model computes both accesses for both widths:

```text
2. shared-memory banks for one warp (32 banks of 4-byte words)
  tile[32][32] write a row: 1-way (conflict-free); read a column: 32-way conflict
  tile[32][33] write a row: 1-way (conflict-free); read a column: 1-way (conflict-free)
  shared memory per block: 4,096 bytes unpadded, 4,224 padded (3.1% more)
```

![The top-left 8 by 8 corner of the tile, each cell labelled with its bank: unpadded, column 0 is bank 0 in every row, in red; padded to 33 words, column 0 runs through banks 0 to 7, in green.](diagrams/03-bank-conflict.svg)

*One unused column changes the row stride from 32 words to 33, and a column of the tile then touches every bank once.*

`transposePadded` in `snippets/transpose.cu` is `transposeShared` with that one line changed. It costs 128 bytes per block, 3.1% more shared memory, far inside the 48 KB a block may use on the GTX 1650. Post 08 applies the same bank model to the addressing of a scan.

## 6. The numbers

`snippets/transpose.cu` times the four kernels on `A: [4096, 4096]` with CUDA events. The laptop card idles at 300 MHz and needs a few hundred milliseconds of work to leave that clock, so the program first runs the copy kernel untimed for one second of GPU time, then times the kernels back to back: five rounds of 20 launches of each, a row's time being the median round's average. Bandwidth is the bytes read plus written, $2 \times 4{,}096^2 \times 4 = 134{,}217{,}728$, over that time, in decimal GB/s. Afterwards each kernel's `B` is compared with a host transpose, on this matrix and on a `1000 x 700` one. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), theoretical peak 160 GB/s
A: [4096, 4096] floats, 64 MiB; blocks of 32 x 8 threads, 4 elements per thread
time: median of 5 rounds, each the average of 20 launches after warm-up
kernel    time (ms)    GB/s  % of copy  check
copy          0.965   139.1     100.0%  ok
naive         3.235    41.5      29.8%  ok
shared        1.727    77.7      55.9%  ok
padded        1.079   124.4      89.5%  ok
A: [1000, 700], sides not multiples of 32: copy ok, naive ok, shared ok, padded ok
```

Every number in the output is measured on one card. It is one of five runs, each started on an idle GPU at no more than 80 degrees C; the five agreed to within 0.04 ms in every row, and during this run's timed rounds `nvidia-smi` read an SM clock of 1,545 to 1,740 MHz at 83 degrees C. Sampling its throttle reasons in a later run showed why the clock dips below 1,740 MHz: a sustained kernel holds the laptop card at its software power cap, about 35 W, at any temperature. The reader's card will give other digits; the order of the rows is the lesson. An earlier draft of this post quoted an RTX 3080 table that the shipped program did not produce; this is the shipped program's output.

![Bars of bandwidth on the GTX 1650: naive 41.5 and shared 77.7 GB/s in red, padded 124.4 in ink, with dotted lines at the copy's 139.1 and the 160 GB/s peak.](diagrams/04-transpose-perf.svg)

*Each fix removes one hazard: the tile removes the strided global writes, the pad removes the bank conflict.*

The tile is worth $3.235 / 1.727 = 1.9$ times and the pad $1.727 / 1.079 = 1.6$ times more, $3.0$ times together; the padded transpose reaches 89.5% of the copy and 78% of the theoretical peak. Its global accesses are as coalesced as the copy's, so what remains between them comes from work the copy does not do, most likely the barrier, which holds each block's stores until all its loads have arrived, and the round trip through shared memory; this post does not measure the two apart. The unpadded tile has the same global accesses, so the whole gap between its 77.7 and the padded 124.4 GB/s is the 32-way conflict: moving a strided access into shared memory is not finished until the banks are checked.

The order matches Harris's published figures. At `1024 x 1024` on a Tesla M2050 and a Tesla K20c, his naive transpose ran at 18.8 and 55.3 GB/s, the tiled one at 51.3 and 97.6, and the padded one at 99.5 and 144.3, about 95% of his fastest copy on each card.

## 7. Make it run

The CPU model needs NumPy and no GPU. From the series root:

```bash
uv run python posts/06-matrix-transpose/snippets/transpose_model.py
```

It prints the three blocks quoted in sections 2, 4 and 5. The kernel needs an NVIDIA GPU and the CUDA Toolkit, and builds from the series root with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 posts/06-matrix-transpose/snippets/transpose.cu -o transpose && ./transpose
```

On Windows the program is `transpose.exe`, and `nvcc` may need `-ccbin` to find the host compiler; [RUNNING.md](../../RUNNING.md) covers both and the `-arch` value for other cards. It prints section 6's output in about three seconds on the GTX 1650 and exits with status 1 if any kernel's `B` disagrees with the host. On a laptop, run it twice.

## 8. What can go wrong?

Each mistake below was reproduced on the author's GTX 1650 with a one-line change to `transposePadded` in `transpose.cu`, unless stated.

- **Not swapping the block indices.** A store whose `col` and `row` come from `blockIdx.x` and `blockIdx.y`, as in the load, writes each transposed tile back to its own block position. Both checks printed `MISMATCH`, at 141 GB/s, faster than the copy: a fast wrong kernel looks like an optimization.
- **Not swapping the tile indices.** Storing `tile[threadIdx.y + j][threadIdx.x]` copies each tile instead of transposing it. Both checks printed `MISMATCH`.
- **The wrong stride for `B`.** Writing `B[(row + j) * N + col]` instead of `* M` passes on the square matrix, where `M` equals `N`, and fails on the `1000 x 700` one, which is why the harness checks a rectangular matrix.
- **Dropping the barrier.** Without `__syncthreads()`, three runs out of three printed `MISMATCH` on both matrices: warps read slots that other warps had not yet written.
- **A block shape that does not match the kernel.** The loops assume `BLOCK_ROWS` rows of threads. A launch with `32 x 32` threads, the shape of a one-element-per-thread transpose, sends `threadIdx.y + j` up to 55, past the 32 rows of `tile`; the program stopped with `an illegal memory access was encountered`.
- **Dropping the store's guard.** On the `1000 x 700` matrix the threads past the edge wrote into the next row of `B`; no error, and the check printed `MISMATCH`.
- **Timing a card whose clock is falling.** An earlier version of the harness checked each kernel on the host between timed rows; the GPU idled meanwhile, its clock fell to 990 MHz, and in one run the copy timed at 54.4 GB/s and the padded transpose at 227.5% of copy. The harness now times every row back to back and checks afterwards.

## 9. Summary

A row of 32 floats costs a warp four 32-byte sectors in one line, a column 32 sectors in 32 lines. A shared-memory tile with both index swaps and one barrier makes both global accesses rows; padding it to `[TILE][TILE + 1]` removes the 32-way bank conflict of its column read. On the GTX 1650 the three transposes run at 30%, 56% and 89.5% of a copy, measured.

---

## Common pitfalls

1. **Counting a strided access as 32 times the traffic.** On compute capability 6.0 and later a warp moves 32-byte sectors, so a column of floats moves 8 times the useful bytes; it is the line count that grows 32 times.
2. **Stopping at the tile.** Moving the strided access into shared memory creates a column read of the tile; check its banks, as `transposeShared`'s 56% of copy shows.
3. **Padding the wrong dimension.** `[TILE + 1][TILE]` adds a row and leaves the row stride at 32 words; only a longer row, `[TILE][TILE + 1]`, shifts each row by one bank.
4. **Testing only square matrices.** A swapped `M` and `N` is invisible when they are equal; check a matrix whose sides differ and are not multiples of `TILE`.
5. **Judging a transpose against the theoretical peak alone.** A copy kernel on the same card, at 87% of the 160 GB/s peak here, is the ceiling for a kernel that only moves data.

---

## Further reading

- Harris, M., "An Efficient Matrix Transpose in CUDA C/C++" (NVIDIA Technical Blog, 18 February 2013). The copy, naive, tiled and padded kernels with `32 x 8` blocks, measured on a Tesla M2050 and a Tesla K20c.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Device Memory Accesses", consulted October 2026. Transaction sizes and the conditions for coalescing.
- NVIDIA, *CUDA C++ Best Practices Guide*, sections "Coalesced Access to Global Memory" and "Shared Memory and Memory Banks", consulted October 2026. The 32-byte transaction rule, strided access, the 32 banks and their serialization.
- Volkov, V., "Better Performance at Lower Occupancy" (GPU Technology Conference, 22 September 2010). Why fewer threads, each with more independent work in flight, can reach the same bandwidth.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 07 - Convolution: constant memory, shared-memory halos, and separable filters](../07-convolution/index.md):** a shared-memory tile again, now with a halo of neighbors around it and the filter in constant memory.
- **[Post 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy](../10-roofline/index.md):** places the transpose, a kernel with no arithmetic at all, on the bandwidth roof this post measured against.
