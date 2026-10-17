# Learning CUDA: From First Kernel to FlashAttention

A free, from-scratch series of 19 posts on GPU programming with CUDA. It starts with why a GPU exists and what a kernel is, writes and benchmarks the first kernels (vector addition, tiled matrix multiplication), builds the parallel patterns almost every GPU algorithm is made from (reduction, histogram, transpose, convolution, scan) and the synchronization rules that make them correct, then draws the performance model that bounds a kernel (roofline, occupancy) and the tools that measure it (Nsight, compute-sanitizer). The last two parts move from one kernel to the whole machine (streams, graphs, fusion, asynchronous copy and software pipelining) and to the numerics, hardware and algorithms that run today's models (floating-point formats, Tensor Cores, FlashAttention), and finish at the production tools (CUTLASS, Triton) that most code actually uses.

Every post ships a CPU model in Python and NumPy that reproduces the kernel's logic and prints the numbers the post quotes, and the real `.cu` kernel beside it. A reader without a GPU follows every post through the model.

## What a reader builds

- A kernel written from one thread's point of view: the global index `blockIdx.x * blockDim.x + threadIdx.x`, the guard on the surplus threads, the copies across the host and device boundary, and a shared-memory tiled matrix multiplication with its barriers in the right places.
- The five parallel patterns (reduce, histogram, transpose, stencil, scan) as working kernels, with coalescing, bank conflicts, atomic contention and warp divergence read off a kernel's indexing, and the guarantees of `__syncthreads`, `__syncwarp`, atomics, fences and scopes stated as rules.
- A performance model built by counting: arithmetic intensity, the roofline of a named card, theoretical occupancy, then the same values measured with Nsight Compute and correctness proved with compute-sanitizer.
- The systems half: copies and compute overlapped across streams, a captured and replayed CUDA Graph, an elementwise chain fused into its producer kernel, and a double-buffered `cp.async` tile loop.
- The modern half: a float read as sign, exponent and mantissa, mixed precision and a stable softmax, a WMMA matrix multiplication on Tensor Cores, the online-softmax update that makes FlashAttention memory-linear, and a reasoned choice between raw CUDA, CUTLASS, Triton and a library call.

## The posts

### Part I - First Kernels

After this part a reader can write a kernel from one thread's point of view, compute the global index, guard the surplus threads, move data across the host and device boundary, explain why vector addition is memory bound, and write a shared-memory tiled matrix multiplication.

1. [01 - Why GPUs exist: the CUDA execution model from the ground up](posts/01-introduction-to-cuda/index.md)
2. [02 - Your first CUDA kernel: vector addition and the memory wall](posts/02-vector-addition/index.md)
3. [03 - Matrix multiplication: from a memory-bound dot product to shared-memory tiling](posts/03-matrix-multiplication/index.md)

### Part II - Parallel Patterns

After this part a reader owns reduce, histogram, transpose, stencil and scan as working kernels, can diagnose coalescing, bank conflicts, atomic contention and warp divergence from a kernel's indexing, and can state what `__syncthreads`, `__syncwarp`, atomics, fences and scopes each guarantee.

4. [04 - Reduction: summing millions of numbers without a serial bottleneck](posts/04-reduction/index.md)
5. [05 - Histogram: beating atomic contention with privatization](posts/05-histogram/index.md)
6. [06 - Matrix transpose: coalescing, shared memory, and bank conflicts](posts/06-matrix-transpose/index.md)
7. [07 - Convolution: constant memory, shared-memory halos, and separable filters](posts/07-convolution/index.md)
8. [08 - Parallel scan: breaking the dependency chain of a prefix sum](posts/08-parallel-scan/index.md)
9. [09 - When threads share data: barriers, atomics, and the memory model](posts/09-synchronization/index.md)

### Part III - Understanding Performance

After this part a reader can count a kernel's arithmetic intensity, place it on the roofline of a named card, derive its theoretical occupancy, measure the achieved values with Nsight Compute, and prove its correctness with compute-sanitizer.

10. [10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy](posts/10-roofline/index.md)
11. [11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy](posts/11-profiling-debugging/index.md)

### Part IV - Systems & Execution

After this part a reader can pipeline copies and compute across streams, capture and replay a graph, fuse an elementwise chain into a producer kernel, and write a double-buffered `cp.async` tile loop.

12. [12 - CUDA streams: overlapping copy and compute for free throughput](posts/12-cuda-streams/index.md)
13. [13 - Cutting launch overhead: CUDA Graphs and stream-ordered allocation](posts/13-cuda-graphs/index.md)
14. [14 - Kernel fusion: the fastest kernel is the one never launched](posts/14-kernel-fusion/index.md)
15. [15 - Asynchronous copy and software pipelining: cp.async, double buffering, and hiding memory latency](posts/15-async-copy-pipelining/index.md)

### Part V - Tensor & Attention

After this part a reader can read a float as sign, exponent and mantissa, apply mixed precision and a stable softmax, write a WMMA matrix multiplication, derive the online-softmax update that makes FlashAttention memory-linear, and place raw CUDA, CUTLASS, Triton, cuBLAS and PyTorch on one ladder.

16. [16 - Numbers on a GPU: FP32, FP16, BF16, and numerical stability](posts/16-numerical-precision/index.md)
17. [17 - Tensor cores: matrix-at-once hardware and mixed precision](posts/17-tensor-cores/index.md)
18. [18 - FlashAttention: making attention memory-linear with tiling and online softmax](posts/18-flash-attention/index.md)
19. [19 - CUTLASS and Triton: why nobody writes raw CUDA for matmul anymore](posts/19-cutlass-triton/index.md)

## Where to start

A programmer new to GPUs reads from post 01 in order. A reader who already calls `torch.matmul` and wants to know what runs underneath can start at post 03, and should not skip Part II, because every later post leans on shared memory, coalescing and the barrier rules. A reader who already writes kernels can start at post 10, after a skim of posts 03 and 06 for the series' notation.

## Before starting

The series assumes C or C++ at the level of pointers, arrays and `for` loops; Python at the level of running a script and reading NumPy; and the matrix product written as a sum, $C_{ij} = \sum_k A_{ik} B_{kj}$, which post 03 restates. An NVIDIA GPU and the CUDA Toolkit 12.x are needed only for the `.cu` kernels, and a few posts set a floor on the card: post 17 needs compute capability 7.0 or above, posts 15 and 18 need an Ampere card or newer (`sm_80`), and post 19's Triton example needs Linux. No other series is required; the `compute` series' `cmp-002` is recommended for the picture of a CPU that the GPU is contrasted with.

## How a post is organized

Each post moves from a picture to the intuition, then the math, then the code. It opens with a short summary and what the reader will be able to do afterwards, works through numbered sections, and closes with the common pitfalls, further reading and what to read next. A kernel is always described from one thread's point of view, because on the GPU the grid is the loop. Each kernel post writes the obvious kernel first, shows where it leaks (strided writes, contended atomics, re-read tiles), applies one fix per section, and compares against the library to show the remaining gap. Every post ships two companions in its `snippets/` folder: a `.py` model that reproduces the kernel's logic and prints the numbers, and the `.cu` kernel, whose header comment states its one-line `nvcc` build and `-arch`. Every post uses the notation fixed in the [notation guide](notation_guide.md). The figures read correctly in light and dark mode. Where a post has review questions, the series website shows them, with hints and explanations, beside the post.

## What is derived, measured, and published

Every number in a post is one of three kinds, and the post says which in the sentence that uses it or in the sentence after its table: derived by the post's CPU model, whose true output is pasted into the post; measured on a named card, the GeForce GTX 1650 (compute capability 7.5), the GeForce RTX 3080 (8.6) or the A100 (8.0); or a representative published figure, cited to its source in [REFERENCES.md](REFERENCES.md). Where the shipped `.cu` checks correctness only and its table was timed separately, the post says so. A GPU timing belongs to the card it was measured on; on another card the digits differ, and the shape of the table is the lesson.

## Running the code

The [running guide](RUNNING.md) is the one recipe for every post: what runs where, the feature floors, the host-compiler setting Windows needs, and the build failures that still occur with CUDA 12.x. In short, from the series root, a CPU model runs with `uv run python posts/04-reduction/snippets/reduction_model.py` (or any Python with NumPy), and a kernel builds with `nvcc -O3 -arch=sm_XX posts/04-reduction/snippets/reduction.cu -o reduction`, where `sm_XX` is the card's compute capability with the dot dropped.

## Examples and challenges

[examples/](examples/) holds standalone example kernels (vector addition, an elementwise multiply, a CPU reference), the benchmark harness behind post 02's vector-addition table, a CMake build that compiles every runnable post kernel straight from its post's `snippets/` folder, and practice challenges (vector addition, ReLU, matrix addition, matrix transpose, reversing an array), each a problem statement, a kernel to fill in and a test harness. The challenges are practice beside the posts, not part of the series. Its [README](examples/README.md) gives the build commands.

## Reference pages

- [Running guide](RUNNING.md): how to run every model and build every kernel, and what to do when a build fails.
- [Cheatsheet](CHEATSHEET.md): one printable page with the launch syntax, the index formula, the memory tiers, the synchronization calls and the build line.
- [Glossary](GLOSSARY.md): every term the series uses, one entry each.
- [Notation guide](notation_guide.md): the operand names, the index and tile names, the three kinds of number, the three cards, the spelling and the figure colors.
- [References](REFERENCES.md): every guide, paper and piece of documentation the posts cite, grouped by post.

## Corrections and feedback

This repository is the published source of the series and is maintained by its author. It does not take pull requests or issues. Toolkit versions, card specifications and library behavior change, so a dead link or a stale figure is a real bug: corrections and suggestions go through the series website.

## License

Prose, questions, and figures are licensed under CC BY 4.0. Code is licensed under MIT. See [LICENSE](LICENSE).
