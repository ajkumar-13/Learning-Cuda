# 16 · Numbers on a GPU — FP32, FP16, BF16, and numerical stability

> **TL;DR.** A GPU number is a fixed budget of bits split between **range** (set by the exponent) and **precision** (set by the mantissa), and every format spends that budget differently: FP32 balances both, FP16 keeps precision but overflows past 65,504, and **bf16** spends bits the other way to hold FP32's full range at lower precision. That budget has three consequences you must design around. Floating-point addition is **non-associative**, so a parallel reduction (Post 04) and a sequential one give slightly different answers, and even two block counts disagree. FP16 overflows so readily that long sums need **mixed precision**: multiply in FP16 but accumulate in FP32, the recipe behind the Tensor Cores of Post 17. And `exp()` in a softmax overflows unless you subtract the row max first, the **stable-softmax** trick that Post 18 turns into a streaming computation.
>
> **After reading this you will be able to:**
> - Read a float as sign, exponent, and mantissa, and say which formats trade range for precision.
> - Explain why a parallel reduction is not bit-for-bit reproducible, and when that matters.
> - Apply mixed precision (FP16 multiply, FP32 accumulate) and the stable-softmax max subtraction.
> - Choose a test tolerance against an FP64 reference instead of demanding exact equality.

![Four horizontal bit strips aligned on the left: FP32 with 1 sign, 8 exponent, 23 mantissa bits; FP16 with 1, 5, 10; BF16 with 1, 8, 7; TF32 with 1, 8, 10. Exponent fields are gray and mantissa fields a lighter gray, with BF16's wide 8-bit exponent highlighted in green to show it matches FP32's range while its short 7-bit mantissa gives up precision.](diagrams/01-float-formats.svg)
*Same 16-bit budget, opposite choices: FP16 spends its bits on mantissa (precision), bf16 on exponent (range, in green).*

---

## 1. The bits: range against precision

A floating-point number is scientific notation in binary. It stores a sign `s`, an exponent `e`, and a mantissa (fraction) `m`, and its value is

$$x = (-1)^{s} \times 1.m \times 2^{\,e - b}$$

where `b` is a fixed bias. Two design knobs fall out of this. The **exponent** width sets the *range*: how large and how small a number can be before it overflows to infinity or underflows to zero. The **mantissa** width sets the *precision*: how finely spaced two representable numbers are, and therefore how many decimal digits you can trust. Widen one field and, at a fixed total size, you must narrow the other. That single tension explains every format in the table.

| Format | Bits | Exponent | Mantissa | ~Decimal digits | Max finite |
|---|---|---|---|---|---|
| FP64 | 64 | 11 | 52 | ~16 | ~1.8e308 |
| FP32 | 32 | 8 | 23 | ~7 | ~3.4e38 |
| TF32 | 19 | 8 | 10 | ~3 | ~3.4e38 |
| FP16 | 16 | 5 | 10 | ~3 | 65,504 |
| BF16 | 16 | 8 | 7 | ~2-3 | ~3.4e38 |

Read the two 16-bit rows against each other, because that is the whole story of modern deep-learning math. **FP16** gives 5 exponent bits and 10 mantissa bits: decent precision (~3 digits), but its range tops out at 65,504, so it overflows easily. **BF16** is FP16's bits reallocated: it keeps FP32's 8 exponent bits, so it has FP32's *range* (max ~3.4e38, no easy overflow), and pays for it with only 7 mantissa bits (~2-3 digits). **TF32** is a compute mode, not a storage type: it has FP32's 8-bit exponent with a 10-bit mantissa, so it occupies an FP32 slot but multiplies as if it were narrower. The choice is never "which is more accurate" in the abstract; it is which of range and precision your kernel cannot afford to lose.

## 2. Addition is not associative

Here is the fact that trips up everyone who parallelizes a sum. In exact arithmetic `(a + b) + c = a + (b + c)`. In floating point it is not, because every add rounds its result to the nearest representable value, and rounding a big-plus-small can throw the small part away entirely. Add `1.0` to `1e8` in FP32 and the answer is exactly `1e8`: the gap between representable numbers near `1e8` is 8, so a `1.0` lands below the rounding threshold and vanishes. Do that a million times and you have lost a million.

![Two columns summing the same values, one large 1e8 and many small 1.0s, in two orders. The left column adds the big value first, so each small 1.0 is rounded away (the lost values struck out in red) and it lands on 100000000. The right column adds the small values first (green), building them into 1000000 before the big value absorbs them, and reaches the correct 101000000.](diagrams/02-non-associative.svg)
*Order decides the answer: big-first (red) rounds every small term away; small-first (green) keeps them. Same numbers, different sums.*

The model sums exactly this list, one `1e8` followed by a million `1.0`s, four ways. It is a standard-library-plus-NumPy script with a fixed construction, so it needs no GPU and reproduces identically. Build the kernel and run the model with:

```bash
nvcc -O3 -arch=sm_75 snippets/precision.cu -o precision && ./precision
python snippets/precision_model.py
```

```text
numbers on a GPU: floating-point behavior (deterministic, seed=0)

format facts (numpy finfo)
  FP16 : max = 65504       eps = 9.766e-04  (~3 decimal digits)
  FP32 : max = 3.4028e+38  eps = 1.192e-07  (~7 decimal digits)
  bf16 : max = 3.3895e+38  eps ~ 7.8e-03  (FP32's exponent, ~2-3 digits)

(a) floating-point addition is NOT associative
    sum of one big value 1e8 followed by 1,000,000 ones; exact = 101000000.0
    order / method                 fp32 result        abs error
    --------------------------------------------------------
    left-to-right (big first)       100000000.0      1000000.0
    sorted ascending (small 1st)    101000000.0            0.0
    pairwise (numpy sum)            100999992.0            8.0
    Kahan (compensated)             101000000.0            0.0
    -> same numbers, four orders, four different fp32 answers

(b) FP16 overflows easily (max 65504)
    x = 100000.0
      fp16 : inf          (overflow: 100000 > 65504)
      fp32 : 100000.0     (fine)
      bf16 : 99840.0      (finite: bf16 keeps FP32's exponent range)

(c) softmax stability: subtract the row max before exp()
    x = [   2.    3. 1000. 1001.]
    naive  softmax = [ 0.  0. nan nan]   (exp overflowed to inf)
    stable softmax = [0.     0.     0.2689 0.7311]
    stable vs FP64 reference: max err 1.89e-08

(d) mixed-precision dot product, length K=4096 (fp16 inputs)
    accumulate in    result        abs error vs FP64
    ------------------------------------------------
    fp16             -0.18591      4.757e-03
    fp32             -0.18116      2.940e-07  (16180x more accurate)
    -> multiply in FP16 for speed, accumulate in FP32 for accuracy
```

Block (a) is the headline: left-to-right (big first) loses the whole million and returns `100000000`; summing small-first returns the exact `101000000`; NumPy's pairwise (tree) sum lands at `100999992`, off by 8. Three orders, three answers. Now connect it to [Post 04](../04-reduction/index.md): a parallel reduction *is* a summation order, the tree order, and it is a different order from the CPU's sequential loop, so the two do not match bit-for-bit. Worse for reproducibility, changing the launch (a different block count, a grid-stride loop, a different warp-shuffle schedule) changes the tree shape and therefore the last bits of the result. This is not a bug; it is arithmetic. The practical rule is in Section 6: never compare a reduction to a reference with `==`.

## 3. Overflow and underflow: why FP16 is fragile

Range is the other half of the bit budget, and FP16 has very little of it. Block (b) makes the cliff concrete: the value `100000` is perfectly ordinary, but it is larger than FP16's maximum of 65,504, so storing it in FP16 gives `inf`. FP32 holds it exactly, and bf16 holds it too (as `99840`, rounded to its coarse mantissa but finite), precisely because bf16 kept FP32's 8 exponent bits. The symmetric hazard is **underflow**: a value smaller than the format's minimum normal number flushes toward zero, and in a chain of multiplies that zero then poisons everything downstream.

This fragility is why you cannot just cast a network to FP16 and hope. A single large activation, a growing sum of squares, or an `exp()` of a large logit overflows to `inf`, and `inf` minus `inf` or `inf` divided by `inf` is `nan`, which then spreads through every later operation. Two responses exist, and this post covers both: change the *format* so the range is wider (bf16, the reason training largely moved to it), or change the *algorithm* so the dangerous magnitudes never arise (mixed-precision accumulation in Section 4, stable softmax in Section 5).

## 4. Mixed precision: fast multiply, accurate accumulate

You want FP16's speed and memory savings for the *multiplies* in a matmul or dot product, but Section 2 warned that a long FP16 sum loses its small terms and Section 3 warned it can overflow. The resolution is to split the precision by role: keep the inputs and the multiply in FP16, but hold the running **accumulator in FP32**. The products are cheap and half-width; the sum, where the rounding damage compounds, is done in a format wide enough to absorb it.

Block (d) measures the payoff on a length-4096 dot product of FP16 vectors. Accumulating entirely in FP16 gives an absolute error of `4.757e-03` against the FP64 reference; switching only the accumulator to FP32 drops the error to `2.940e-07`, about **16,000 times more accurate**, for the same FP16 inputs and the same number of operations. Nothing about the multiply changed. The single decision of where the sum lives is the difference between a trustworthy answer and a drifting one.

The companion kernel [`snippets/precision.cu`](snippets/precision.cu) is this experiment on the GPU. It runs two versions of the same block-reduced dot product, one accumulating in `__half` and one in `float`, and checks both against a host FP64 reference computed on the identical FP16 operands, so the only variable is the accumulator. It is self-contained (own `main()`, a `CUDA_CHECK` macro, a tolerance check) and builds with the `nvcc` line in Section 2. This is the exact contract the [Tensor Cores](../17-tensor-cores/index.md) of Post 17 implement in hardware: they multiply `16×16` FP16 tiles and accumulate the products into an FP32 fragment, which is why the recipe is fast *and* accurate at once.

## 5. Stable softmax: subtract the max

Softmax turns a vector of scores into probabilities, `softmax(x)_i = exp(x_i) / sum_j exp(x_j)`. Written that way it is a numerical trap: attention scores can be 100 or 1000, and $e^{1000}$ overflows even FP32's huge range to `inf`, after which `inf / inf` is `nan` and the whole row is destroyed. Block (c) shows it: the naive softmax of `[2, 3, 1000, 1001]` returns `[0, 0, nan, nan]`.

![Two softmax pipelines on the same score vector with a huge entry. The naive path applies exp() directly, overflows the large scores to inf, and the normalization produces nan (both drawn in red). The max-subtracted path first subtracts the row maximum so the largest exponent becomes exp(0)=1, every intermediate value stays finite (green), and it reaches the correct probabilities.](diagrams/03-stable-softmax.svg)
*Naive `exp()` overflows to inf then nan (red); subtracting the row max makes the largest exponent `exp(0)=1`, so every value stays finite (green). Both give the same probabilities.*

The fix is one line of algebra that is *exactly* equal, not an approximation. Subtract the row maximum $M = \max_j x_j$ from every element before exponentiating:

$$\text{softmax}(x)_i = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}}$$

Multiplying top and bottom by $e^{-M}$ leaves the ratio unchanged, but now the largest exponent is $e^{0} = 1$ and every other is smaller, so nothing overflows and the sum is well-conditioned. The model's stable path returns `[0, 0, 0.2689, 0.7311]` and matches the FP64 reference to `1.89e-08`. This is the numerical foundation of [Post 18](../18-flash-attention/index.md): FlashAttention cannot see a whole row at once, so it carries a *running* max and rescales the partial sums as new blocks arrive, which is the same max subtraction promoted to an online, streaming computation.

## 6. Better sums, and how to test a kernel

If summation order costs accuracy, two techniques buy it back. **Pairwise (tree) summation** adds in a balanced binary tree, so the rounding error grows like $\log N$ instead of $N$; it is what NumPy's `sum` does and roughly what a GPU reduction does, and block (a) shows it landing far closer than the naive loop. **Kahan (compensated) summation** goes further: it keeps a second variable tracking the low-order bits each add throws away and feeds them back in, recovering near-exact results (block (a) shows Kahan hitting the exact `101000000`) at the cost of a few extra FLOPs per element. Reach for pairwise by default and Kahan when a long sum genuinely needs it.

Two cautions close the loop. First, **fast math**. Compiler flags like `-use_fast_math` and the contraction of a multiply-and-add into a single fused `FMA` change the rounding, and reassociation flags let the compiler reorder sums; all of them trade last-bit accuracy and reproducibility for speed, so enable them deliberately, not by habit. Second, and most important for daily work, **testing**. Because none of the results above are bit-exact, a kernel test that asserts `output == reference` will fail on correct code. Compute the reference in FP64, then assert the GPU result is within a tolerance: an absolute tolerance for values near zero, a relative one otherwise, sized to the format and the reduction length. The companion kernel demonstrates the pattern, passing when its FP32-accumulated error falls under `1e-3`. Testing floating-point kernels is the art of choosing that tolerance, never demanding equality.

---

## Common pitfalls

- **Comparing GPU output to a reference with `==`.** Parallel reductions sum in a different order than the CPU, so the last bits differ by construction. Assert closeness within an FP64-referenced tolerance, not exact equality.
- **Casting to FP16 and hoping.** FP16 overflows past 65,504 and underflows near zero, and one `inf` becomes a `nan` that spreads. Use bf16 for the wider range, or keep the accumulator in FP32.
- **Accumulating a long sum in FP16.** The running total outgrows FP16's ~3 digits and small terms vanish. Multiply in FP16 if you like, but accumulate in FP32; the model shows a 16,000× accuracy gap on one dot product.
- **Softmax without the max subtraction.** `exp()` of a large score overflows to `inf` and the row becomes `nan`. Subtract the row max first: it is algebraically exact and costs one pass.
- **Assuming a result is reproducible across launches.** A different block count or reduction schedule changes the summation tree and therefore the low bits. If you need bit-exact reproducibility, fix the algorithm and the launch, or accept a tolerance.
- **Turning on fast math to "go faster" without checking.** `-use_fast_math` and FMA contraction change rounding and can break a tolerance you were relying on. Measure the accuracy cost before shipping it.

---

## Further reading

- Goldberg, D., *"What Every Computer Scientist Should Know About Floating-Point Arithmetic"* (1991). The canonical explanation of rounding, non-associativity, and cancellation (technical, foundational).
- IEEE, *"IEEE Standard for Floating-Point Arithmetic (IEEE 754)"* (current). The definition of the formats, rounding modes, and special values `inf`/`nan` (reference).
- Micikevicius, P. et al., *"Mixed Precision Training"* (2018). The FP16-multiply, FP32-accumulate recipe and loss scaling, in a training context (technical).
- Higham, N. J., *"Accuracy and Stability of Numerical Algorithms"* (2nd ed., 2002). The reference on pairwise and compensated (Kahan) summation and error bounds (technical, foundational).
- NVIDIA, *"Train With Mixed Precision"* and the *"CUDA C++ Programming Guide"* half-precision intrinsics (current). The practical bf16/FP16/TF32 guidance and the `__half` API (reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 17, Tensor cores](../17-tensor-cores/index.md)**: the hardware that multiplies in FP16 and accumulates in FP32, the mixed-precision recipe this post justifies.
- **[Post 18, FlashAttention](../18-flash-attention/index.md)**: where the stable-softmax max subtraction becomes an online, streaming computation.
- **[Post 04, Reduction](../04-reduction/index.md)**: the parallel sum whose answer this post shows is order-dependent.
