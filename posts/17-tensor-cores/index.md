# 17 · Tensor cores — matrix-at-once hardware and mixed precision

> **TL;DR.** Even a tuned FP32 matmul on this post's RTX 3080 reaches only ~10.7 TFLOPS, while the card's FP16 **Tensor Core** peak is roughly 119 TFLOPS: hardware where one warp instruction computes a whole `16×16×16` matrix multiply-accumulate, `D = A·B + C`, which is **4096 fused multiply-adds at once**, 128× a normal warp. The price of admission is **FP16** inputs, fast but limited (max 65,504, ~3 digits), so the trick that makes them usable is **mixed precision**: multiply in FP16 but **accumulate in FP32**, keeping the speed with near-FP32 accuracy. Training mostly uses **bf16** (and TF32) instead, which trade FP16's precision for the same wide FP32 exponent and so avoid the 65,504 overflow. You reach the hardware through the **WMMA** (Warp Matrix Multiply-Accumulate) API: load tiles into opaque *fragments*, call `mma_sync`, store the result.
>
> **After reading this you will be able to:**
> - Explain how a Tensor Core differs from a CUDA core and why it is so much denser.
> - Use mixed precision (FP16 multiply, FP32 accumulate) and say why it is accurate.
> - Write a WMMA matmul with fragments, `load_matrix_sync`, `mma_sync`, `store_matrix_sync`.
> - Avoid the alignment, dimension, and warp-synchrony traps Tensor Cores impose.

![A CUDA core does one FMA per thread (32 per warp); a Tensor Core computes a 16x16x16 matrix multiply-accumulate (4096 FMAs) per warp instruction, in green.](diagrams/01-cuda-vs-tensor.svg)
*A standard core does scalar math; a Tensor Core does a whole tile of multiply-accumulate in one instruction.*

---

## 1. The motivation: the missing TFLOPS

A note on hardware: the earlier matmul posts measured a GTX 1650 (compute capability 7.5, ~0.3 TFLOPS FP32 peak); from here on the numbers are from an **RTX 3080** (compute capability 8.6), whose tiled FP32 matmul reaches ~10.7 TFLOPS and cuBLAS SGEMM ~33. Yet that same RTX 3080 has an FP16 Tensor Core peak of roughly **119 TFLOPS** — over 10× the tuned FP32 kernel. The gap is not a better loop; it is *different hardware*. Alongside the CUDA cores sit **Tensor Cores**, specialized for exactly one thing: matrix multiply-accumulate. To use them you leave the comfortable world of `float` and enter FP16 and the WMMA API.

```bash
# Tensor Cores need compute capability 7.0 (Volta) or newer.
# Benchmarks here are on an RTX 3080 (sm_86).
nvcc -O3 -arch=sm_75 snippets/tensor_core.cu -o tensor_core && ./tensor_core
python snippets/tensor_core_model.py   # CPU model: density + mixed-precision accuracy
```

## 2. The hardware: a matrix per instruction

A CUDA core is scalar: each thread does one `c = a*b + c`, so a warp manages 32 fused multiply-adds per cycle. A Tensor Core is a *tile* machine: the whole warp cooperates on `D = A·B + C` where `A`, `B`, `C`, `D` are `16×16` matrices. That single warp instruction is `16·16·16 = 4096` FMAs, **128× denser**. A CPU model makes the density and the precision story concrete:

```text
tensor core model
  FMAs per warp instruction: CUDA core 32, Tensor Core 4096 (128x)
  FP16 range: max 65504, ~3 decimal digits
  dot products of length K=4096, max abs error vs FP64 reference:
    pure FP16 accumulate   : 0.03131
    FP16 mul + FP32 accum   : 0.000606  (52x more accurate)
  -> mixed precision keeps Tensor Core speed with near-FP32 accuracy
```

## 3. Mixed precision: fast multiply, accurate accumulate

The catch is precision. `A` and `B` are FP16, which has only ~3 decimal digits and overflows past 65,504. Summing a long dot product entirely in FP16 loses the small bits, and the error grows. The fix is to do the **multiply in FP16** (fast, on the Tensor Core) but the **accumulate in FP32** (accurate), which the hardware supports natively.

![A pipeline: FP32 inputs to FP16, multiply in FP16 on the Tensor Core, accumulate in FP32 (green), output FP32; with pure-FP16 error 0.031 vs mixed 0.0006.](diagrams/02-mixed-precision.svg)
*Multiply in FP16 for speed; accumulate in FP32 for accuracy. The model shows mixed precision is ~52× more accurate than pure FP16.*

The model above shows the payoff: pure FP16 drifts to a 0.031 error on a length-4096 dot product, while FP16-multiply with FP32-accumulate stays at 0.0006. This is exactly the recipe behind **automatic mixed precision (AMP)** training in every modern framework.

## 4. The programming model: WMMA

You cannot write `c = a*b` and get Tensor Cores; the hardware needs data in specific layouts, operated on as whole tiles, by the whole warp. NVIDIA exposes this through **WMMA** (Warp Matrix Multiply-Accumulate). You work with **fragments**: opaque containers whose elements are distributed across the warp's 32 lanes, never indexed directly.

![The WMMA lifecycle: load tiles from global memory into fragments, run mma_sync (green), store the result back; all warp-synchronous.](diagrams/03-wmma-lifecycle.svg)
*Load tiles into fragments, one `mma_sync`, store: every call is warp-synchronous.*

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;   // FP32 accumulate
wmma::fill_fragment(c_frag, 0.0f);
for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, A + ..., K);
    wmma::load_matrix_sync(b_frag, B + ..., N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // one instruction, 4096 FMAs
}
wmma::store_matrix_sync(C + ..., c_frag, N, wmma::mem_row_major);
```

Every call ends in `_sync`: all 32 lanes must execute it together, so a WMMA call inside a divergent branch is a bug. A `K`-loop of these tile multiplies builds a full matrix product entirely on the Tensor Cores.

## 5. The numbers

On a 4096³ multiply, Tensor Cores plus mixed precision are an order of magnitude beyond a tuned FP32 kernel:

| Implementation | TFLOPS | Speedup vs naive FP32 |
|---|---|---|
| naive FP32 | 3.0 | 1.0× |
| tiled FP32 (shared memory) | 10.7 | 3.5× |
| cuBLAS SGEMM (FP32) | 32.7 | 10.8× |
| Tensor Core (basic WMMA) | 65.4 | 21.5× |
| Tensor Core (optimized) | 98.1 | 32.3× |
| cuBLAS HGEMM (FP16 TC) | 124.9 | 41.1× |

![A TFLOPS bar chart rising from naive FP32 (3) through the Tensor Core kernels (65, 98 in green) to cuBLAS HGEMM (125 in blue).](diagrams/04-tensor-perf.svg)
*The same matmul spans 3 to 125 TFLOPS; the Tensor Core kernels are the green jump.*

A basic WMMA kernel already reaches ~65 TFLOPS, and an optimized one ~98, about **30× the naive FP32**. cuBLAS HGEMM, with software pipelining on top, reaches ~125. (That edges past the ~119 figure quoted earlier because 119 is the FP32-accumulate peak this post targets; cuBLAS HGEMM can accumulate in FP16, whose peak is roughly double.) The cost is real, though: the inputs are FP16, so this is the right tool for deep learning (where AMP is standard) and the wrong one for double-precision scientific work.

---

## Common pitfalls

- **Non-multiple-of-16 dimensions.** WMMA tiles are `16×16×16`; matrices whose `M`, `N`, or `K` are not multiples of 16 must be padded, or the kernel crashes or returns wrong results.
- **Calling WMMA in a divergent branch.** The `_sync` operations require all 32 lanes; guarding a load or `mma_sync` with `if (threadIdx.x < 16)` deadlocks or corrupts the fragment.
- **Mismatched layouts.** Fragments are declared `row_major` or `col_major`; if the data in memory does not actually match, the product is garbage. For matmul, `A` row-major and `B` col-major is the usual pairing.
- **Accumulating in FP16 to save space.** It reintroduces the precision loss mixed precision exists to avoid; keep the accumulator FP32 unless you have measured that FP16 is safe.
- **Ignoring alignment.** Tensor Core loads want 256-byte-aligned data for full speed; unaligned pointers can quietly halve throughput.

---

## Further reading

- NVIDIA, *"CUDA C++ Programming Guide — Warp Matrix Functions (WMMA)"* (current). The reference for fragments and the `_sync` operations (reference).
- Markidis, S. et al., *"NVIDIA Tensor Core Programmability, Performance & Precision"* (2018). A study of WMMA performance and the FP16 accuracy trade-off (technical).
- Micikevicius, P. et al., *"Mixed Precision Training"* (2017). The paper behind FP16-compute, FP32-accumulate training (technical, foundational).
- NVIDIA, *"Programming Tensor Cores in CUDA 9"* (Developer Blog, 2017). The original WMMA walkthrough (technical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 14, Kernel fusion](../14-kernel-fusion/index.md)**: the IO-awareness that the next post combines with these Tensor Cores.
- **[Post 18, Flash attention](../18-flash-attention/index.md)**: fusing tiled Tensor Core matmuls with an online softmax to make attention memory-linear.
