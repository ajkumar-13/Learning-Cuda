# 19 · CUTLASS and Triton — why nobody writes raw CUDA for matmul anymore

> **TL;DR.** You can now write a tiled, Tensor Core matmul by hand, but a *production* GEMM also needs multi-level tiling, double buffering, async copies, epilogue fusion, split-K, and a separate tuning for every GPU generation — thousands of lines for a kernel that still tends to reach only ~60% of peak, so almost nobody does it. **CUTLASS**, NVIDIA's C++ template library, exposes the exact device → threadblock → warp → MMA tiling hierarchy you built as composable, architecture-tuned pieces (near-peak, but a steep climb). **Triton** is a Python DSL where you program at the **block** level and the compiler generates the threads, coalescing, shared memory, and Tensor Core calls, reaching ~95% of cuBLAS for an afternoon's work. The point of learning raw CUDA first was to *understand* these tools, not to compete with them.
>
> **After reading this you will be able to:**
> - Place raw CUDA, CUTLASS, Triton, cuBLAS, and PyTorch on the abstraction ladder.
> - Read a Triton kernel and explain how block-level programming differs from thread-level.
> - Describe the CUTLASS tiling hierarchy and what each template level controls.
> - Choose the right tool for a given kernel, and say why hand-written CUDA is rarely it.

![Five rungs from raw CUDA up to PyTorch, trading control for ease; Triton highlighted green as the sweet spot.](diagrams/01-abstraction-ladder.svg)
*Each rung up trades control for ease, usually at little or no loss of speed. You learned the bottom rung to trust the rest.*

---

## 1. The motivation: the maintenance nightmare

The tiled and Tensor Core kernels from [Post 03](../03-matrix-multiplication/index.md) and [Post 17](../17-tensor-cores/index.md) were already involved, and they were the *simple* versions. A library-grade GEMM adds register tiling, software pipelining with `cp.async`, double or triple buffering, split-K and stream-K for awkward shapes, fused epilogues, every layout combination, and a different tuning for Volta, Ampere, Hopper, and Blackwell. That is 5,000 to 50,000 lines that must be re-tuned each GPU generation. Writing it from scratch is months; matching cuBLAS is rare. The sensible response is to move up the **abstraction ladder**.

## 2. CUTLASS: the hierarchy as building blocks

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source C++ template library for exactly this. It exposes the same nested tiling you built by hand, but as configurable, pre-tuned components: the **device**-level GEMM contains a **threadblock tile** (which stages `A` and `B` in shared memory), which contains a **warp tile**, which issues the **MMA** Tensor Core instruction.

![Nested boxes: device GEMM, then threadblock tile, then warp tile, then the green MMA instruction.](diagrams/03-cutlass-hierarchy.svg)
*The device → threadblock → warp → instruction nesting from the earlier posts, packaged as template parameters.*

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, RowMajor, cutlass::half_t, RowMajor, cutlass::half_t, RowMajor,
    float, OpClassTensorOp, Sm80,                    // FP32 accumulate, Tensor Cores, Ampere
    GemmShape<128,128,32>, GemmShape<64,64,32>, GemmShape<16,8,16>, ... >;
```

Every template argument is a tiling decision you now recognize. CUTLASS reaches ~98% of cuBLAS and supports fused epilogues and custom layouts, at the cost of a steep learning curve and days of work.

## 3. Triton: program in blocks, not threads

**Triton** takes the opposite trade. It is a Python DSL where you write a kernel for one *block* of data: `tl.program_id` picks the block, `tl.arange` makes a vector of offsets, `tl.load`/`tl.store` move tiles, and `tl.dot` does the tile multiply on Tensor Cores. There is no `threadIdx`, no `__syncthreads`, no manual shared memory; the compiler generates all of it.

![Raw CUDA vector add (thread-level, with the global-index math) beside the Triton version (block-level, in green).](diagrams/02-cuda-vs-triton.svg)
*Raw CUDA is one thread; Triton is one block. The compiler fills in the threads, coalescing, and synchronization.*

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # register accumulator
    for k in range(0, K, BLOCK_K):
        a = tl.load(...); b = tl.load(...)
        acc += tl.dot(a, b)                                # Tensor Cores if available
    tl.store(..., acc.to(tl.float16))
```

The full kernel, with strides and boundary masking, is in [snippets/triton_matmul.py](snippets/triton_matmul.py).

This capstone ships Python companions ([triton_matmul.py](snippets/triton_matmul.py) and [tools_model.py](snippets/tools_model.py)) and no `.cu` by design — the hand-written CUDA baselines live in [Post 03](../03-matrix-multiplication/index.md) and [Post 17](../17-tensor-cores/index.md). A CPU model confirms this block-tiled accumulation is the same algorithm as the hand-written tiled matmul (`python snippets/tools_model.py`):

```text
tools model
  block-tiled matmul == reference : True
  4096^3 FP16 GEMM on A100 (% of cuBLAS peak):
    tool                   dev time   %peak  code
    cuBLAS                 minutes     100%  ~1
    CUTLASS (tuned)        days         98%  ~50 (templates)
    Triton (autotuned)     hours        95%  ~30 (Python)
    Triton (naive)         hours        90%  ~30 (Python)
    raw CUDA (ours)        weeks        60%  thousands
  -> Triton reaches ~95% of cuBLAS for hours of work, not weeks
```

Triton's other lever is **autotuning**: decorate the kernel with a list of block-size configs and it benchmarks them and keeps the fastest for each problem shape.

## 4. The numbers: performance versus effort

On a 4096³ FP16 GEMM on an A100, the high-level tools dominate hand-written code on *both* axes that matter, speed and effort. The absolute throughputs below are representative published A100 figures (cuBLAS ~275 TFLOPS on a 4096³ FP16 GEMM), not numbers measured by this post's runnable companion:

| Tool | Throughput | Effort |
|---|---|---|
| cuBLAS | 275 TFLOPS (100%) | minutes |
| CUTLASS (tuned) | 270 (98%) | days |
| Triton (autotuned) | 263 (95%) | hours |
| Triton (naive) | 248 (90%) | hours |
| raw CUDA (hand-written) | 165 (60%) | weeks |

The percentages are of cuBLAS, not the hardware roofline: the A100's FP16 Tensor Core peak is ~312 TFLOPS, so cuBLAS itself runs at roughly 88% of the theoretical maximum, and "% of cuBLAS peak" measures how close each tool gets to that strong baseline rather than to the silicon's ceiling. The last few percent that CUTLASS recovers over Triton comes from techniques the libraries automate: **split-K** and **stream-K** partition the K dimension across blocks (or work-units) to keep every streaming multiprocessor busy on awkward shapes, **cp.async** streams the next tile from global memory while the current one computes, and **epilogue fusion** folds the bias, activation, or scaling into the GEMM's write-out so the result is touched only once.

![A TFLOPS bar chart: cuBLAS 275 and CUTLASS 270 in blue, Triton 263/248 in green, raw CUDA 165 neutral, with effort noted.](diagrams/04-tool-perf.svg)
*The hand-written kernel is both the slowest and the most work. Triton is ~95% of cuBLAS for a fraction of the effort.*

## 5. When to use what, and the meta-lesson

The rule is short. For a **standard** op (GEMM, conv, attention), call **cuBLAS / cuDNN** or `torch.matmul`, done. For a **custom** fused op where you need most of the speed quickly, write **Triton**. For the last few percent or an exotic layout, reach for **CUTLASS**. Hand-written CUDA is for **learning the hardware** and the rare case nothing else covers.

That is the whole arc of this series. You started with one thread adding one element and finished able to read a FlashAttention kernel and a CUTLASS template. The reason to learn the bottom of the ladder was never to live there; it was so that every tool above it is legible: you know what `tl.dot` compiles to, why an FP32 accumulator matters, what a bank conflict costs, and when fusion is free. That understanding is what makes you effective at the top of the ladder, where the real work happens.

---

## Common pitfalls

- **Hand-writing a GEMM for production.** Unless you are learning or covering a gap, a hand-rolled kernel will be slower and far more work than cuBLAS, CUTLASS, or Triton; reach for those first.
- **Assuming Triton is always "good enough."** It usually reaches 90–95%; when the last few percent on a hot kernel pays for itself, CUTLASS or cuBLAS is the right call.
- **Skipping autotuning.** A naive Triton kernel can leave 5–10% on the table; `@triton.autotune` over a few block configs recovers most of it for free.
- **Fighting CUTLASS's template errors blind.** The errors are notoriously dense; start from the `examples/` and change one template argument at a time.
- **Forgetting the foundations transfer.** Coalescing, shared-memory tiling, occupancy, and fusion still govern performance inside Triton and CUTLASS; the tools automate the mechanics, not the thinking.

---

## Further reading

- Tillet, P., Kung, H. T., & Cox, D., *"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"* (MAPL, 2019). The language and compiler design (technical, foundational).
- NVIDIA, *"CUTLASS"* (GitHub and documentation, current). The template library, the CuTe abstraction, and worked examples (reference).
- OpenAI, *"Triton Tutorials"* (current). The official matmul, softmax, and fused-attention walkthroughs (technical).
- Thakkar, V. et al., *"CUTLASS 3.x and CuTe"* (NVIDIA GTC, current). The modern layout-algebra API (technical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 18, FlashAttention](../18-flash-attention/index.md)**: the algorithm whose official implementations are written in exactly these tools.
- **[Post 01, Introduction to CUDA](../01-introduction-to-cuda/index.md)**: the beginning of the ladder — worth a second read now that you can see the whole climb.
