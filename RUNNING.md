# Running the code

Every post ships two companions in its `snippets/` folder: a **CPU model** (`.py`) that reproduces the kernel's logic and prints the numbers the post quotes, and the **real CUDA kernel** (`.cu`). This page is the shared run recipe; [SETUP.md](SETUP.md) covers installing a GPU toolchain and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) covers failures.

> **No GPU?** Run the `.py` models anywhere Python does — they reproduce every number the posts quote. The `.cu` kernels are written to be read without running them.

## Prerequisites by part

| Part | Posts | You need |
|---|---|---|
| I — First Kernels | 01–03 | Python 3.9+, NumPy; an NVIDIA GPU + CUDA Toolkit 12+ for the `.cu` files |
| II — Parallel Patterns | 04–09 | as above |
| III — Understanding Performance | 10–11 | as above; the profiling post uses Nsight and `compute-sanitizer`, which ship with the CUDA 12.x toolkit |
| IV — Systems & Execution | 12–15 | as above; CUDA Graphs (post 13) need CUDA 11+; the async-copy kernel (post 15) needs an Ampere GPU (`sm_80`+) |
| V — Tensor & Attention | 16–19 | a Volta-or-newer GPU (compute capability ≥ 7.0); `flash_attention.cu` (post 18) targets `sm_80`; post 19 also uses Triton (`pip install triton`, Linux) |

## The two commands

Run any post's CPU model with Python and NumPy:

```bash
python posts/04-reduction/snippets/reduction_model.py
```

Build and run its real kernel with one `nvcc` line — use the `-arch` for your GPU (each `.cu` header states the one it was measured with; see the [arch table](SETUP.md#step-3--pick-your--arch-flag)):

```bash
nvcc -O3 -arch=sm_75 posts/04-reduction/snippets/reduction.cu -o reduction && ./reduction
```

On Windows the executable is `reduction.exe`; if `nvcc` can't find `cl.exe`, add `-ccbin` (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#cannot-find-compiler-clexe)).

## Posts whose companion is Python-only or an excerpt

- **Post 10 (Roofline & occupancy)** and **Post 16 (Floating-point precision)** are modeling posts: their `.py` model *is* the verified artifact, printing derived numbers that need no GPU. Post 16 also ships a small `.cu` contrast for reading.
- **Post 18 (FlashAttention)** ships an *illustrative* kernel excerpt that shows the tile loop and online-softmax update; its CPU model (`flash_attention.py`) is the runnable, verified artifact. The post says so where the code appears.
- **Post 19 (CUTLASS and Triton)** is a tooling survey and ships Python companions (`triton_matmul.py`, `tools_model.py`) rather than a `.cu`; the post points you to posts 03 and 17 for the hand-written CUDA baselines it compares against.

## Reproducing the numbers

Every empirical number in a post comes from a committed companion: the CPU models print exact, verifiable counts, and each benchmark table states the GPU and compute capability it was measured on. If a CPU-model number does not reproduce on your machine, that is a bug — see [CONTRIBUTING.md](CONTRIBUTING.md).
