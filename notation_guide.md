# Notation and vocabulary guide

One notation, shared by every post in the series. The goal is that a name means the same thing in every post: a reader who learns in post 03 that `s_A` is the shared-memory tile of `A` never meets it as `tileA` in post 15 and has to wonder whether that is something else. The [glossary](GLOSSARY.md) gives a one-paragraph definition of every term; this page fixes the names, the symbols, the labels on numbers and the typography.

Two rules govern everything below:

1. **One quantity, one name, everywhere.** The same quantity is never written two ways, in one post or across posts; prose, math and code use the same identifier.
2. **Each name is reintroduced on first use in every post, with its shape**, and every acronym is expanded on first use in every post, because readers arrive mid-series from a search engine: a tile is `tile: [TILE, TILE]`, the operands of a matrix multiplication are `A: [M, K]`, `B: [K, N]`, `C: [M, N]`.

## Contents

1. Operands and typography
2. The launch and the global index
3. Tiles and shared memory
4. The performance model
5. The three kinds of number
6. The three cards and the feature floors
7. House conventions
8. Sibling series
9. Figure roles
10. Checklist before publishing a post

---

## 1. Operands and typography

The concrete arrays a kernel reads and writes are code spans, never typeset: `A`, `B`, `C` for the operands (for vector addition `C = A + B`, for matrix multiplication `C = A * B`), and `d_A`, `d_B`, `d_C` for their device copies. The scalar problem size is `N` in 1D and `M`, `N`, `K` for a matrix multiplication.

When a post typesets the abstract linear-algebra framing, it uses the family's typography:

| Kind | Style | Example |
|---|---|---|
| matrix | bold upper case | $\mathbf{X}$, $\mathbf{W}$ |
| vector | bold lower case | $\mathbf{x}$ |
| scalar | italic lower case | $i$, $t$ |
| index range | italic, stated with its bounds | $i \in \{0, \dots, N-1\}$ |

Typeset bold B is banned as a matrix name, because across the family it is the reserved alias of a bias vector (written $\mathbf{b}$, lower case). A post that must typeset the second operand of a matrix product writes $\mathbf{X}$, or keeps the code span `B`. The matrix product as a sum is written $C_{ij} = \sum_k A_{ik} B_{kj}$, with italic, unbolded entries.

## 2. The launch and the global index

| Concept | Identifier | Type |
|---|---|---|
| thread index in its block | `threadIdx.x` (`.y`, `.z`) | scalar |
| block index in the grid | `blockIdx.x` (`.y`, `.z`) | scalar |
| threads per block | `blockDim.x`, or `blockSize` on the host | scalar, a multiple of 32 |
| blocks per grid | `gridDim.x`, or `blocksPerGrid` on the host | scalar |
| warp size | 32 | constant |
| global index, 1D | `i` | scalar |
| global index, 2D | `(row, col)` | pair of scalars |

The global index is always written exactly `blockIdx.x * blockDim.x + threadIdx.x`, in that order, with `*` and `+`, in a code span, and introduced once per post. In 2D, `row` is the same form in `.y` and `col` the same form in `.x`. It is never reordered (with the thread index first, for instance) and never typeset with `\text{}` or `\cdot`.

A kernel is described from one thread's point of view: "thread `i` writes `C[i]`", never "the loop writes": on the GPU the grid is the loop. The launch is written `kernel<<<blocksPerGrid, blockSize>>>(args)`.

## 3. Tiles and shared memory

| Concept | Identifier | Shape |
|---|---|---|
| tile width | `TILE` | scalar |
| shared-memory tile of `A` | `s_A` | `[TILE, TILE]` |
| shared-memory tile of `B` | `s_B` | `[TILE, TILE]` |
| double-buffered tiles | `s_A`, `s_B` with a leading buffer index | `[2, TILE, TILE]` |
| padded tile (no bank conflicts) | `tile` | `[TILE, TILE + 1]` |

A shared-memory array that is not a tile of an operand takes a short descriptive name with an `s_` prefix or the name the post gives it on first use (`s` for a reduction buffer, `warpSums` for per-warp partials).

## 4. The performance model

| Quantity | Symbol | Unit |
|---|---|---|
| arithmetic intensity | $I$ | FLOP per byte moved to and from DRAM |
| peak DRAM bandwidth | $\beta$ | GB/s |
| peak compute | $\pi$ | GFLOP/s |
| attainable performance | $P(I) = \min(\pi, I \cdot \beta)$ | GFLOP/s |
| achieved bandwidth | a fraction of $\beta$ | percent |
| theoretical occupancy | resident warps over the most an SM can hold | percent |

A kernel is memory bound when $I < \pi / \beta$, the ridge point, and compute bound above it. Arithmetic intensity is stated in FLOP per byte and compared with the card's ridge point, never with another card's.

## 5. The three kinds of number

Every number in a post is one of three kinds, and the post says which:

| Kind | What it is | How the post shows it |
|---|---|---|
| derived by the model | printed by the post's CPU model in `snippets/` | the true output is pasted into the post, and the sentence names the script |
| measured on a named card | timed on one of the three cards | the card and its compute capability are named with the table or in the sentence |
| published figure | a representative number from a published source | the source is cited, with its full entry in [REFERENCES.md](REFERENCES.md) |

The sentence after every table of numbers says which kind the table holds. Where the shipped `.cu` checks correctness only and the table was timed separately, that sentence says so. A GPU timing belongs to its card; on another card the digits differ, and the shape of the table is the lesson. An illustrative value (a latency drawn to show a ratio) carries the word "illustrative".

## 6. The three cards and the feature floors

| Card | Architecture | Compute capability | `-arch` |
|---|---|---|---|
| GeForce GTX 1650 | Turing | 7.5 | `sm_75` |
| GeForce RTX 3080 | Ampere | 8.6 | `sm_86` |
| A100 | Ampere | 8.0 | `sm_80` |

The snippets default to `-arch=sm_75`, the GTX 1650 the early posts were measured on, and the header comment of every `.cu` states its build line and `-arch`. Compute capability is written `major.minor` (7.5) and the architecture flag `sm_XX` (`sm_75`), the same number with the dot dropped. The feature floors: Tensor Cores need compute capability 7.0 or above (post 17), where independent thread scheduling also starts and makes `__syncwarp()` necessary in warp-level code (post 09); `cp.async`, BF16 arithmetic and `flash_attention.cu` need `sm_80` (posts 15 and 18); Triton needs Linux (post 19).

## 7. House conventions

- **Post numbers** are two digits with a lower-case "post" in prose: "post 06", "posts 09 and 11".
- **Spelling** is American: *optimize*, *synchronize*, *color*, *behavior*, *catalog*, *modeling*. A quoted title and a code identifier keep their own spelling.
- **Acronyms** are expanded on first use in every post: graphics processing unit (GPU), streaming multiprocessor (SM), single instruction, multiple threads (SIMT), dynamic random-access memory (DRAM), Peripheral Component Interconnect Express (PCIe), floating-point operation (FLOP), warp matrix multiply-accumulate (WMMA), and so on.
- **Voice** is third person and neutral: "the reader", "a thread", "the kernel", never "you" in body prose; a build or run recipe may use the imperative.
- **Punctuation:** no em or en dashes; a range is written "posts 04 to 09"; a title's subtitle follows a colon.
- **Math** is inline `$...$` or display `$$...$$` only. Code identifiers, CUDA calls, compiler flags and file names are code spans with their exact casing: `__syncthreads()`, `cudaMemcpy`, `-arch=sm_75`, `device_query.cu`.
- **Dimensions** in prose and figures are written with a plain `x`: a `16 x 16 x 16` tile, a `4096 x 4096` matrix.

## 8. Sibling series

Another series is named by its key and its posts by id in code font, never by a path, a URL, or a phrase such as "the Transformer series' post 3":

| Series | Key | How this series cites it |
|---|---|---|
| Compute | `compute` | by id: how a CPU executes a program (`cmp-002`, recommended by post 01), floating-point arithmetic as a first pass (`cmp-035`, recommended by post 16) |
| Transformer | `transformer` | by id: the attention equation and why it is quadratic (`tf-003`, recommended by post 18) |

The series `cnn-from-scratch`, `nnfs`, `machine-learning`, `numerical-linear-algebra` and `discrete-math-and-algorithms` are named by key only, where a post says what it leaves to them.

## 9. Figure roles

Every figure in the series uses one color per job a thing does in the machine, so that color carries the job and the written name carries the identity. The design tokens carry the series prefix `--cuda-`.

- **Blue (host):** the CPU side and control: the host program, a launch, a stream, a graph node, the copy the host issues, the profiler reading the device.
- **Orange (compute):** work done on the device: a thread, a lane, a warp, a block, a streaming multiprocessor, an arithmetic unit, a Tensor Core, a kernel body. An active lane is orange.
- **Green (memory):** data at rest and the tiers that hold it: registers, shared memory, L1, L2, global memory, host memory, an array, a tile in shared memory, a histogram bin, a prefix-sum buffer.
- **Purple (sync):** anything that orders or fences: `__syncthreads`, `__syncwarp`, an atomic, a memory fence, a stream event, a graph dependency, the host-device boundary when a figure is about crossing it, a `cp.async` wait. A barrier is a purple bar across the lanes it holds.
- **Red (hazard):** waste or a wrong result: an idle lane in a divergent warp, a serialized pass, a strided access that wastes most of each 32-byte sector it moves, a bank conflict, a contended atomic, a lost update in a race, an unguarded out-of-bounds thread. Red never means a component.

A library baseline (cuBLAS, cuDNN, CUB, PyTorch, CUTLASS, Triton) and every neutral quantity are drawn in ink and neutral fills, named as the post names them. There is no series accent color, a chart with one series is ink, and no figure relies on color alone: every shape carries its name ("shared memory", "block 1", "stream 2"). A measured value in a figure names its card, compute capability, toolkit and date in the subtitle; a cited value is drawn hollow or dashed with the word "cited".

## 10. Checklist before publishing a post

- [ ] Operands as code spans `A`, `B`, `C`, `d_A`; no typeset bold B (Section 1).
- [ ] The global index written exactly `blockIdx.x * blockDim.x + threadIdx.x`, introduced once; the kernel described from one thread's point of view (Section 2).
- [ ] `TILE`, `s_A`, `s_B` for tiles; every shape stated on first use (Section 3).
- [ ] $I$, $\beta$ and $\pi$ as Section 4 defines them.
- [ ] Every number derived, measured on a named card, or a cited published figure, and the sentence after each table says which (Section 5).
- [ ] Every `.cu` header states its build line and `-arch` (Section 6).
- [ ] Every acronym expanded on first use in this post; American spelling; third person; no em or en dashes (Section 7).
- [ ] Sibling series by key and id only (Section 8).
- [ ] Every relative link resolves, with forward slashes only.
