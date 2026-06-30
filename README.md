<div align="center">

# Learning CUDA: From First Kernel to FlashAttention

**A 15-part, from-scratch course in GPU programming — from adding two arrays to writing a memory-linear attention kernel, every idea derived, benchmarked, and drawn.**

![Posts](https://img.shields.io/badge/posts-15-blue?style=flat-square) ![Parts](https://img.shields.io/badge/parts-4-green?style=flat-square)

</div>

---

## Why this series?

Most CUDA tutorials stop at "here is a kernel." This one builds the whole stack: the execution model, the memory hierarchy, the classic parallel patterns, the system-level tricks, and the modern hardware (Tensor Cores) and algorithms (FlashAttention) that run today's models — finishing at the production tools (CUTLASS, Triton) that most code actually uses.

Every concept is derived from first principles, illustrated with original diagrams, and grounded in a runnable companion. Each post ships a faithful **CPU model** (NumPy or plain Python) that reproduces the kernel's logic and prints real, non-invented numbers, so you can follow every result even without a GPU; the real `.cu` kernels are committed alongside and build straight from [examples/](examples/).

**Who it's for:** programmers who know a little C++ and want to understand the GPU at every layer, not just call a library. The only prerequisites are basic programming and high-school algebra.

---

## How to read it

The series runs in four parts. Each post follows the same rhythm — **picture → intuition → math → code** — and ends with common pitfalls, further reading, and a pointer to what comes next. Every post lives in its own folder:

```
posts/NN-slug/
├── index.md          ← the post
├── frontmatter.yaml  ← metadata (title, date, tags, hero, reading time)
├── diagrams/         ← original SVG illustrations (light + dark mode)
└── snippets/         ← runnable CPU models and the real .cu kernels
```

Read sequentially for the full arc, or jump to any post if you have the prerequisites.

**Setup & reference:** [SETUP.md](SETUP.md) (get a GPU machine running, cross-OS) · [RUNNING.md](RUNNING.md) (run any post's code) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) (when a build fails) · [CHEATSHEET.md](CHEATSHEET.md) (one-page reference) · [notation_guide.md](notation_guide.md) · [GLOSSARY.md](GLOSSARY.md) · [REFERENCES.md](REFERENCES.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [LICENSE](LICENSE)

**Hands-on code:** [examples/](examples/) holds standalone example kernels, a CMake build, and practice challenges (vector add, ReLU, matrix ops) that go alongside the posts.

---

## Part I — First Kernels

*The execution model and your first working, benchmarked kernels.*

| # | Post | What it teaches |
|--:|------|-----------------|
| 01 | [Introduction to CUDA](posts/01-introduction-to-cuda/index.md) | Why GPUs exist; grids, blocks, threads, warps; the memory hierarchy |
| 02 | [Vector addition](posts/02-vector-addition/index.md) | Your first kernel, memory management, honest benchmarking, the memory wall |
| 03 | [Matrix multiplication](posts/03-matrix-multiplication/index.md) | 2D indexing, shared-memory tiling, from memory-bound to compute-bound |

## Part II — Parallel Patterns

*The handful of patterns that almost every GPU algorithm is built from.*

| # | Post | What it teaches |
|--:|------|-----------------|
| 04 | [Reduction](posts/04-reduction/index.md) | Tree reduction, sequential addressing, warp shuffle |
| 05 | [Histogram](posts/05-histogram/index.md) | Atomic contention and privatization |
| 06 | [Matrix transpose](posts/06-matrix-transpose/index.md) | Memory coalescing and shared-memory bank conflicts |
| 07 | [Convolution](posts/07-convolution/index.md) | Constant memory, shared-memory halos, separable filters |
| 08 | [Parallel scan](posts/08-parallel-scan/index.md) | Breaking a dependency chain; Blelloch; stream compaction |

## Part III — Systems & Performance

*Moving from one kernel to the whole machine.*

| # | Post | What it teaches |
|--:|------|-----------------|
| 09 | [Profiling and debugging](posts/09-profiling-debugging/index.md) | Nsight, compute-sanitizer, and the occupancy you can measure |
| 10 | [CUDA streams](posts/10-cuda-streams/index.md) | Overlapping copy and compute; pinned memory; the pipeline |
| 11 | [Kernel fusion](posts/11-kernel-fusion/index.md) | Removing intermediate memory traffic; the free-op insight |
| 12 | [Async copy & pipelining](posts/12-async-copy-pipelining/index.md) | cp.async, double buffering, hiding memory latency |

## Part IV — Tensor & Attention

*The hardware and algorithms behind modern AI, and the tools that ship them.*

| # | Post | What it teaches |
|--:|------|-----------------|
| 13 | [Tensor cores](posts/13-tensor-cores/index.md) | Matrix-at-once hardware (WMMA) and mixed precision |
| 14 | [FlashAttention](posts/14-flash-attention/index.md) | Tiling and online softmax for memory-linear attention |
| 15 | [CUTLASS and Triton](posts/15-cutlass-triton/index.md) | The production tools, and why you learned raw CUDA first |

---

## Running the companions

Each post's CPU model runs with only Python and NumPy — no GPU needed:

```bash
python posts/04-reduction/snippets/reduction_model.py
```

The real CUDA kernels compile with `nvcc` on a machine with a GPU (match `-arch` to your card; see the [arch table](SETUP.md#step-3--pick-your--arch-flag)):

```bash
nvcc -O3 -arch=sm_75 posts/02-vector-addition/snippets/vector_add.cu -o vector_add && ./vector_add
```

Or build **every** runnable kernel at once: [examples/](examples/) wires each post's `.cu` into one CMake build, producing an executable per kernel — exactly like the example kernels there:

```bash
cd examples && cmake -S . -B build && cmake --build build --config Release
./build/reduction        # one executable per post kernel (Windows: .\build\reduction.exe)
```

See [RUNNING.md](RUNNING.md) for the per-post recipe, [SETUP.md](SETUP.md) to install a GPU toolchain, and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) when a build fails.
