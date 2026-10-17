# Examples and exercises

Hands-on CUDA code that accompanies the **[Learning CUDA series](../README.md)**. Where the
series posts each ship a focused snippet, this folder holds standalone example kernels, a
CMake build, and a set of practice challenges to solve.

> **Setup.** For installing a GPU toolchain on any OS, see the series'
> [RUNNING.md](../RUNNING.md) and its [build failures](../RUNNING.md#when-a-build-fails). The
> Windows-with-Visual-Studio specifics (the `-ccbin` dance, CMake generators) live here in
> [WINDOWS-SETUP.md](WINDOWS-SETUP.md) and [WINDOWS-TROUBLESHOOTING.md](WINDOWS-TROUBLESHOOTING.md).

## What's here

```
examples/
├── src/                       # example kernels + challenge folders
│   ├── vec_add.cu             # vector addition
│   ├── multiply.cu            # element-wise multiply
│   ├── vec_add_cpu.cpp        # CPU-only reference
│   ├── benchmark_vector_add.cu
│   ├── vector_addition/       # challenge: question.md + solution.cu + test_solution.cu
│   ├── relu/                  # challenge: ReLU activation
│   ├── matrix_addition/       # challenge: matrix addition
│   ├── matrix_transpose/      # challenge: transpose kernel
│   └── reverse_array/         # challenge: reverse an array
├── CMakeLists.txt             # builds the example kernels with nvcc
├── test_cuda.cu               # minimal hello-from-GPU sanity check
└── NewCudaScripts.md          # how to add and compile a new .cu
```

## Build and run

With CMake (from this folder):

```bash
cmake -S . -B build && cmake --build build --config Release
./build/vecadd_nvcc        # Windows: .\build\vecadd_nvcc.exe
```

Or compile any single file directly with `nvcc` (match `-arch` to your GPU; see the
[arch table](../RUNNING.md#pick-the--arch-flag)):

```bash
nvcc -O2 src/vec_add.cu -o vec_add && ./vec_add
```

On Windows, if `nvcc` cannot find `cl.exe`, add `-ccbin` with your MSVC path; see
[WINDOWS-SETUP.md](WINDOWS-SETUP.md) and [CMakeLists_explained.md](CMakeLists_explained.md).

## Run the series kernels

The build also compiles every runnable kernel from the [series posts](../README.md) — one
executable per post, built straight from the post's own `snippets/*.cu` (no copies, so they
never drift from what the posts teach). After `cmake --build build`:

```bash
./build/reduction      # post 04   (Windows: .\build\reduction.exe)
./build/transpose      # post 06
./build/streams        # post 12
# ...also device_query, vector_add, matmul, histogram, convolution, scan,
#    synchronization, occupancy_demo, cuda_graphs, fusion, async_pipeline,
#    precision, tensor_core
```

Two need specific hardware to **run** (they still compile on a CUDA 11+ toolkit):
`async_pipeline` (post 15) needs an Ampere GPU (`sm_80`+), and `tensor_core` (post 17) needs a
Tensor-Core GPU (Volta+, i.e. RTX not GTX). Post 18's FlashAttention ships an illustrative
excerpt (its Python model is the runnable artifact) and post 19 is Python/Triton-only, so
neither is a build target here.

## Challenges

Each challenge folder has a `question.md` (the problem), a `solution.cu` (your kernel goes
here), and a `test_solution.cu` (a harness that checks it):

```bash
nvcc -O2 src/vector_addition/solution.cu src/vector_addition/test_solution.cu -o test && ./test
```

Available: `vector_addition`, `relu`, `matrix_addition`, `matrix_transpose`, `reverse_array`.

## Where to learn the concepts

The hands-on code here mirrors the series posts. Start with
[Post 01 - Why GPUs exist: the CUDA execution model from the ground up](../posts/01-introduction-to-cuda/index.md) and
[Post 02 - Your first CUDA kernel: vector addition and the memory wall](../posts/02-vector-addition/index.md), then work the challenges.
