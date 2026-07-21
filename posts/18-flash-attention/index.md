# 18 · FlashAttention — making attention memory-linear with tiling and online softmax

> **TL;DR.** Transformer attention, `softmax(QKᵀ/√d)V`, has a quadratic problem: the score matrix is `N×N`, so doubling the sequence length quadruples the memory, and a 32K-token sequence needs a 2 GB matrix per head. The standard implementation makes it worse by running three kernels and **writing that `N×N` matrix to global memory twice**, so attention is **memory-bound**, not compute-bound. **FlashAttention** computes the exact same result without ever materializing the matrix: it **tiles** `Q`, `K`, `V` into blocks that fit in on-chip SRAM, and uses an **online softmax** (a running max and sum, rescaled per block) to fold each block in incrementally. The matrix lives only in SRAM, traffic drops from `O(N²)` to `O(N)`, and the same idea makes 100K-token context windows possible.
>
> **After reading this you will be able to:**
> - Explain why standard attention is memory-bound and scales quadratically.
> - Derive the online-softmax update (running max, running sum, rescale).
> - Describe the FlashAttention tiling: outer over `Q` blocks, inner over `K`/`V` blocks.
> - Say why recomputation in the backward pass is a net win.

![Standard attention writes the N x N score matrix to HBM twice (red); FlashAttention fuses everything in SRAM and writes only O (green).](diagrams/01-quadratic-wall.svg)
*The `N×N` matrix is the wall. Standard attention materializes it; FlashAttention never does.*

---

## 1. The motivation: the quadratic wall

Attention compares every token with every other token, so the score matrix `S = QKᵀ` is `N×N`. Double the sequence and the matrix quadruples: 8 MB at 2K tokens, 134 MB at 8K, **2.1 GB at 32K** (per head, FP16). The standard implementation, three kernels, makes the *traffic* quadratic too: kernel 1 writes `S` to HBM, kernel 2 reads it and writes the softmaxed `P`, kernel 3 reads `P` to multiply by `V`. The `N×N` matrix crosses slow global memory four times, written twice and read twice. The bottleneck is not the matmuls; it is **bandwidth**.

## 2. The math trick: online softmax

The obstacle to tiling is that softmax needs the **whole** row: it divides each `exp(xᵢ)` by the sum over all `j`. The fix is to compute it **incrementally**, keeping two running statistics per row: the **max** `m` seen so far (for numerical stability) and the **sum** `ℓ` of `exp(xⱼ - m)`. When a new block arrives with its own local max, you take the new global max and **rescale** the old sum to it before adding the block's contribution.

![Two blocks folded into a running max m and sum l; when the max grows, the old sum is rescaled by exp(m_old - m_new), in green.](diagrams/02-online-softmax.svg)
*A running max and sum are `O(1)` state per row, so the full row of scores never has to exist.*

$$m_{\text{new}} = \max(m_{\text{old}}, \tilde{m}), \qquad \ell_{\text{new}} = \ell_{\text{old}}\,e^{m_{\text{old}} - m_{\text{new}}} + \tilde{\ell}\,e^{\tilde{m} - m_{\text{new}}}$$

The same rescale `e^{m_{\text{old}} - m_{\text{new}}}` is applied to the running **output** accumulator, so the attention output is built up block by block as well. A CPU model confirms the incremental version is exact, matching standard attention to floating-point noise. **Run it yourself:** `python snippets/flash_attention.py` (needs NumPy).

```text
flash attention model
  N=256, d=32, block=64
  flash == standard   : True  (max err 8.33e-17)
  the N-by-N score matrix FlashAttention never materializes:
    seq   2048: standard      8.4 MB   vs   flash O(N) running stats
    seq   8192: standard    134.2 MB   vs   flash O(N) running stats
    seq  32768: standard   2147.5 MB   vs   flash O(N) running stats
  double the sequence length -> 4x standard memory, but only 2x for flash
```

## 3. The algorithm: tiling Q, K, V

With online softmax, the whole operation tiles. An **outer loop** walks blocks of `Q` rows; for each, an **inner loop** streams blocks of `K` and `V` through on-chip SRAM, computing the block scores, the online-softmax update, and the running output, accumulating entirely on-chip. Only when a `Q` block is finished is one `O` block written to HBM.

![Outer loop over Q blocks, inner over K/V blocks, all inside a green SRAM box; O written once per Q block.](diagrams/03-tiling.svg)
*Only block-sized tiles ever live in SRAM; the `N×N` matrix never touches HBM.*

```python
for i in range(0, N, B):              # outer: blocks of Q rows
    O_i, m_i, l_i = 0, -inf, 0
    for j in range(0, N, B):          # inner: blocks of K, V
        S = (Q_i @ K_j.T) * scale     # block scores, in SRAM
        m_new = maximum(m_i, S.max(-1))
        P = exp(S - m_new); corr = exp(m_i - m_new)
        l_i = corr * l_i + P.sum(-1)
        O_i = corr * O_i + P @ V_j     # accumulate, rescaled
        m_i = m_new
    O[i] = O_i / l_i                   # normalize, write once
```

Mapping the math of Section 2 onto these identifiers: `m_old` is `m_i` going into the step and `m_new` is the updated `m_new`; the block-local stats `m̃`/`ℓ̃` are `S.max(-1)` and `P.sum(-1)`; and the rescale factor `e^{m_old - m_new}` is `corr`.

A CUDA version of this loop is in [`snippets/flash_attention.cu`](snippets/flash_attention.cu). It is an **excerpt**, not a runnable program: it has no `main()` and the cooperative shared-memory load of `K`/`V` is stubbed, so it is for reading and a compile-check (`nvcc -arch=sm_80 -c snippets/flash_attention.cu -o /dev/null`), not execution. The runnable, verified artifact is the CPU model above.

This is the [kernel-fusion](../14-kernel-fusion/index.md) idea applied to the whole attention block: do all the work in fast memory, touch slow memory the minimum number of times. The two matmuls inside (`QKᵀ` and `P·V`) are where the arithmetic lives: in a production kernel they run on the [Tensor Cores](../17-tensor-cores/index.md), while the excerpt here uses plain scalar FMA (fused multiply-add) loops for clarity. **Causal masking** drops out naturally: decoder attention is causal (a token may only attend to earlier positions), so FlashAttention masks future keys by skipping inner `K`/`V` blocks that lie entirely past the current `Q` block and masking within the diagonal block, which also saves the upper-triangular work.

## 4. The numbers

Because attention is memory-bound, removing the `N×N` traffic is a large, *growing* win, and it lifts the memory ceiling entirely:

| Sequence | Standard | FlashAttention | Speedup |
|---|---|---|---|
| 512 | 0.5 ms | 0.3 ms | 1.7× |
| 2,048 | 4.2 ms | 1.1 ms | 3.8× |
| 8,192 | 68 ms | 8 ms | 8.5× |
| 16,384 | 280 ms | 18 ms | 15.6× |
| 32,768 | out of memory | 42 ms | ∞ |

These times were measured separately on an NVIDIA A100 (80 GB, compute capability 8.0), FP16, one head; the shipped CPU model checks correctness and the memory scaling, not these wall-clock times.

![A speedup bar chart rising 1.7, 3.8, 8.5, 15.6 with sequence length, and an infinity bar at 32K where standard attention runs out of memory.](diagrams/04-flash-perf.svg)
*The speedup grows with length because the quadratic traffic dominates more and more; past ~16K, standard attention simply cannot fit.*

The memory reduction is even more dramatic: 32× at 2K tokens, 512× at 32K. For **training**, FlashAttention does not store the `N×N` matrix for the backward pass either; it **recomputes** it block by block. That sounds wasteful, but attention is memory-bound, so the spare compute is effectively free, and the memory saved is what lets the sequence be long at all. This single algorithm is why context windows went from GPT-2's 1K to today's 100K+.

---

## Common pitfalls

- **Forgetting the rescale on the output.** The running `O` must be scaled by `exp(m_old - m_new)` whenever the max grows, exactly like `ℓ`; skipping it weights early blocks wrong.
- **Dropping the safe-softmax max subtraction.** Without subtracting the running max, `exp` of a large score overflows. The online max is both the tiling mechanism and the stability trick.
- **Block sizes that overflow SRAM.** The tiles for `Q`, `K`, `V`, and the scores must all fit in shared memory; 64×64 suits an A100, but a smaller GPU needs 32×32 or it spills.
- **Recomputing the softmax from scratch in the backward pass.** Save the per-row logsumexp (`m + log ℓ`) in the forward pass so the backward recomputation is cheap and numerically consistent.
- **Writing your own when a library exists.** The official FlashAttention and the fused-attention kernels in cuDNN/CUTLASS are extensively tuned; hand-roll only to learn or to cover a case they miss.

---

## Further reading

- Dao, T. et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"* (2022). The original algorithm, IO analysis, and kernel (technical, foundational).
- Dao, T., *"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"* (2023). The parallelism and work-partitioning improvements (technical).
- Milakov, M. & Gimelshein, N., *"Online normalizer calculation for softmax"* (2018). The online-softmax recurrence at the heart of the method (technical).
- Rabe, M. & Staats, C., *"Self-attention Does Not Need O(n²) Memory"* (2021). The memory-linear attention idea, in parallel (technical).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 19, CUTLASS and Triton](../19-cutlass-triton/index.md)**: the production tools that let you write kernels like this without hand-managing fragments and shared memory.
- **[Post 17, Tensor cores](../17-tensor-cores/index.md)**: look back at the matrix hardware that runs the two matmuls inside the FlashAttention loop.
