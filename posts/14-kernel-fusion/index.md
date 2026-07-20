# 14 · Kernel fusion — the fastest kernel is the one you don't launch

> **TL;DR.** One clean line like `relu(conv(x))` is two GPU kernels, and the intermediate tensor between them is written to global memory by the first and immediately read back by the second, pure wasted bandwidth. **Fusion** does both operations in one kernel: the intermediate stays in a **register** and only the final result is written, halving the memory traffic for two ops and saving `(N-1)/N` for a chain of `N`. The reason it is nearly free is **arithmetic intensity**: an elementwise op like ReLU does one operation per eight bytes, so on its own the cores just wait on memory, but stacked onto a compute-bound kernel its single instruction hides in the existing pipeline. The one limit is **register pressure**: fuse too much and occupancy collapses or registers spill to slow memory.
>
> **After reading this you will be able to:**
> - Explain why separate elementwise kernels waste memory bandwidth on intermediates.
> - Fuse a chain of operations to read once, compute in registers, and write once.
> - Use arithmetic intensity to say which ops are "free" to fuse and which are not.
> - Recognize the register-pressure limit and when *not* to fuse.

![Separate kernels write the intermediate Y to memory and read it back (red); a fused kernel keeps Y in registers and writes only Z (green).](diagrams/01-fusion-traffic.svg)
*The intermediate exists only to be written and immediately re-read. Fusion deletes that round-trip.*

---

## 1. The motivation: the hidden cost of clean code

`x = relu(conv(x))` reads beautifully and runs badly. On the GPU it is a conv kernel that writes its output `Y` to global memory, then a ReLU kernel that reads `Y` straight back, applies a one-instruction `max`, and writes `Z`. `Y` is touched twice by global memory for no reason: it is a scratch value that should never have left the chip. Two kernels also mean two launches and two synchronizations.

The cost is bandwidth. For a chain of `N` operations done separately, every step reads the whole tensor and writes the whole tensor, so the traffic is `N` times read-plus-write. Done fused, the tensor is read once and written once.

## 2. The fix: compute in registers, write once

The conv above is the illustrative compute-bound *producer* of the intermediate, the kind of heavy kernel you fuse cheap elementwise work onto. The runnable kernel below demonstrates the same idea on a pure elementwise chain (scale, then bias, then ReLU), and the CPU model in `snippets/fusion_model.py` extends it to four ops by adding a GELU (Gaussian Error Linear Unit). A fused kernel keeps the result in a register, applies the bias and the ReLU right there, and writes the final value:

```cuda
__global__ void scaleBiasReluFused(const float* x, float* z, float s, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] * s + b;     // intermediate lives in a register
        z[i] = fmaxf(v, 0.0f);      // one read of x, one write of z, no global write of Y
    }
}
```

Build and run it from the post directory (the sources live in `snippets/`):

```bash
nvcc -O3 -arch=sm_75 snippets/fusion.cu -o fusion && ./fusion
python snippets/fusion_model.py
```

A CPU model of a four-op chain shows the saving exactly: separate kernels move 134 MB, the fused kernel moves 34 MB, a **75% reduction** (`(N-1)/N` for `N = 4` ops), and the result is identical:

```text
fusion model
  tensor              = 4,194,304 floats = 16.8 MB
  chain               = + bias -> * scale -> relu -> gelu  (4 ops)
  fused == separate   : True
  separate traffic    : 134 MB  (4 ops x read+write)
  fused traffic       : 34 MB  (read once, write once)
  saving              : 75%  (= (N-1)/N)
```

## 3. Why memory-bound ops fuse for free

The deeper reason fusion wins is **arithmetic intensity** (FLOPs per byte). ReLU does one comparison per element and moves eight bytes: an intensity of `0.125`, far below any GPU's compute-to-bandwidth balance point. Run alone, it is hopelessly memory-bound; the cores spend almost all their time waiting on the load and store.

![Run alone, ReLU is mostly memory stalls; fused onto a compute-bound conv, its one instruction hides in the conv's existing stalls, in green.](diagrams/02-free-piggyback.svg)
*A memory-bound op stacked onto a compute-bound one costs almost nothing: one extra instruction in an already-busy pipeline.*

When you fuse that ReLU onto a compute-bound conv or matmul, its single instruction executes while the bigger kernel is already running (or already stalled waiting for memory). No launch, no extra traffic, no intermediate. This is exactly why every serious framework fuses elementwise ops (bias, activation, dropout, residual add) onto the matmul or conv that produces them.

## 4. The trade-off: register pressure

Fusion is not unconditionally free. Each fused operation needs registers to hold its intermediates, and an SM (streaming multiprocessor) has a fixed register file of 65,536 32-bit registers. The more registers a thread uses, the fewer threads fit at once, which lowers **occupancy** (the ratio of active warps to the maximum warps per SM) and leaves fewer warps to hide memory latency.

![Two register files: a 32-register kernel fits 2048 threads (100% occupancy); a 128-register over-fused kernel fits only 512 (25%), and may spill.](diagrams/03-register-pressure.svg)
*Over-fusion trades memory traffic for occupancy, and if registers spill to local memory (DRAM), it is a net loss.*

A simple kernel at 32 registers per thread fills an Ampere-class SM's 2048 threads (full occupancy; Turing `sm_75` caps at 1024, so its ceiling is lower); an over-fused kernel at 128 registers fits only 512 (25%). Worse, if a kernel needs more registers than exist, the compiler **spills** them to "local memory," which is actually DRAM and ~100× slower than a register. Fuse elementwise ops freely, but check occupancy (`nvcc -Xptxas=-v`, Nsight) before fusing anything heavy: if occupancy falls below roughly 50%, reconsider the fusion.

## 5. The numbers, and the real world

The win grows with the number of chained operations removed:

| Fusion | Speedup |
|---|---|
| MatMul + ReLU | 1.07× |
| MatMul + Bias + ReLU | 1.24× |
| MatMul + Bias + GELU | 1.35× |
| Conv + BatchNorm + ReLU | 1.41× |
| Residual + LayerNorm | 1.94× |
| Attention (3 ops) | 3.33× |

*These speedups are representative published figures for fused versus unfused kernels, not measurements from the shipped `snippets/fusion.cu` (which checks correctness only). Treat them as illustrative orders of magnitude; the exact numbers depend on the GPU, shapes, and library versions.*

![A speedup bar chart rising from MatMul+ReLU at 1.07 to Residual+LayerNorm 1.94 and Attention 3.33, the last two in green.](diagrams/04-fusion-perf.svg)
*More chained ops fused means more round-trips removed, so a bigger speedup.*

This is not a niche trick; it is how modern ML runs. `torch.compile` analyzes the graph and emits fused Triton kernels; cuDNN ships fused conv-bias-activation; and **FlashAttention** is fusion taken to its limit, computing `QKᵀ`, the softmax, and `·V` in one kernel so the `S×S` score matrix (for sequence length `S`) never touches global memory, turning attention's traffic from `O(S²)` to `O(S)`. The golden rule of GPU performance fits in one line: *eliminate intermediate tensors, keep values in registers, write to memory once.*

---

## Common pitfalls

- **Over-fusing until occupancy collapses.** Each fused op adds registers; past a point you have too few warps to hide latency, and the kernel slows despite moving less data. Profile occupancy, not just traffic.
- **Triggering register spills.** If `nvcc -Xptxas=-v` reports spills, the "fused" kernel is hitting DRAM through local memory; back off or use `__launch_bounds__`.
- **Fusing independent work that should run in parallel.** Two independent convolutions belong in two kernels (or two streams); fusing them serializes work that could overlap.
- **Reaching for a hand-fused kernel first.** `torch.compile`, cuDNN, and CUTLASS already fuse the common patterns well; write a custom fused kernel only when profiling shows the library leaves traffic on the table.
- **Fusing across a needed synchronization.** If the second op depends on a *block-wide* or *grid-wide* result of the first (a reduction across the whole tensor), it may not fuse into the same kernel without a barrier you cannot express within a block.

---

## Further reading

- Dao, T. et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"* (2022). Fusion taken to the limit, and the IO-aware framing (technical, foundational).
- NVIDIA, *"cuDNN Developer Guide — Fused Operations"* (current). The library's automatic fusion of conv, bias, and activation (reference).
- PyTorch, *"Introduction to `torch.compile`"* (current). How a framework fuses graphs into Triton kernels automatically (technical).
- Williams, S., Waterman, A., & Patterson, D., *"Roofline"* (2009). The arithmetic-intensity model that says which ops are memory-bound (technical, historical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 12, CUDA streams](../12-cuda-streams/index.md)**: the other half of system optimization — overlap what you must transfer, after fusion removes what you needn't.
- **[Post 15, Async copy and pipelining](../15-async-copy-pipelining/index.md)**: the next traffic-hiding trick. Overlap the loads you cannot remove, with `cp.async` and double buffering, the way cuBLAS and CUTLASS do.
