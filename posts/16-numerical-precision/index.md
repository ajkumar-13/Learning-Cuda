# 16 - Numbers on a GPU: FP32, FP16, BF16, and numerical stability

> **TL;DR.** A floating-point number spends a fixed budget of bits on range (the exponent) and precision (the mantissa): FP16 keeps 10 mantissa bits and overflows above 65,504, while BF16 keeps FP32's range with only 7. Every addition rounds, so a sum's last bits depend on its order: the same 1,048,576 floats summed on the author's GTX 1650 with 1, 14, 56 and 224 blocks gave three different FP32 sums, each predicted bit for bit by a model on the central processing unit (CPU). An FP16 dot product accumulated in FP32 was 5,325 times more accurate than one accumulated in FP16 (60 to 60,448 times over 20 other draws of the data). A softmax that subtracts the row maximum cannot overflow, and a kernel is tested against an FP64 reference with an error bound sized to its format and summation depth, never with `==`.
>
> **Prerequisites:** [post 04](../04-reduction/index.md).
> **Safe to skip?** Skip it if the reader can already derive a format's range from its bit widths, explain why two launches of one reduction disagree in the last bits, say what an FP32 accumulator and a fused multiply-add change, write a softmax that cannot overflow, and size a test tolerance from the unit roundoff.
>
> **After reading, you will be able to:**
>
> - Read a float as sign, exponent and mantissa, and derive from the bit widths which formats trade range for precision.
> - Explain why a parallel reduction is not bit-for-bit reproducible across block counts, and when one launch is.
> - Apply mixed precision, FP16 inputs with an FP32 accumulator, and say what the compiler's fused multiply-add changes.
> - Apply the stable-softmax max subtraction and explain why it changes no probability.
> - Choose a test tolerance against an FP64 reference from the format's unit roundoff and the summation depth.

![Four bit strips drawn to scale: FP32 with 1 sign, 8 exponent and 23 mantissa bits; FP16 with 1, 5 and 10, its largest value 65,504 in red; BF16 with 1, 8 and 7, its exponent outlined; TF32 with 1, 8 and 10. Beside each strip, its largest value and its spacing eps.](diagrams/01-float-formats.svg)

*Two 16-bit formats make opposite bets: FP16 spends its bits on the mantissa, BF16 on FP32's exponent.*

---

## 1. The question: how does a float spend its bits?

A floating-point number is scientific notation in binary. It stores a sign bit $s$, an exponent field $e$ and a mantissa (fraction) field $m$, and a normal number's value is

$$x = (-1)^{s} \times 1.m \times 2^{\,e - b},$$

where $b$ is a fixed bias and the leading 1 is implicit, so a mantissa of $m$ bits gives a precision of $p = m + 1$ significant bits. The exponent's width sets the range: how large a number can be before it overflows to infinity and how small before it falls below the smallest normal number. The mantissa's width sets the precision: the gap between neighboring numbers, which at 1 is $\varepsilon = 2^{-m}$ and grows with the number. At a fixed total width, widening one field narrows the other.

The companion model, [`snippets/precision_model.py`](snippets/precision_model.py), derives each format's limits from its widths alone (largest value $(2 - 2^{-m}) \times 2^{b}$ with $b = 2^{e-1} - 1$, smallest normal $2^{1-b}$, and $p \log_{10} 2$ decimal digits) and checks the FP64, FP32 and FP16 rows against NumPy's `finfo`:

```text
formats, derived from the bit widths (p = mantissa bits + 1, the implicit leading 1)
  format  bits  exp  mantissa  max finite  min normal  eps = 2^-(p-1)  decimal digits
  FP64      64   11        52  1.798e+308  2.225e-308       2.220e-16            16.0
  FP32      32    8        23   3.403e+38   1.175e-38       1.192e-07             7.2
  TF32      19    8        10   3.401e+38   1.175e-38       9.766e-04             3.3
  FP16      16    5        10       65504   6.104e-05       9.766e-04             3.3
  BF16      16    8         7   3.390e+38   1.175e-38       7.812e-03             2.4
  FP16 subnormals reach 5.960e-08; the FP64, FP32 and FP16 rows equal numpy.finfo
```

These are derived by the model, and they agree with the binary64, binary32 and binary16 formats of IEEE 754-2019 (the floating-point standard of the Institute of Electrical and Electronics Engineers). FP16, the standard's half precision, keeps 10 mantissa bits, about 3.3 decimal digits, but its 5 exponent bits stop at 65,504. BF16 (bfloat16, "brain floating point") is FP32 with the low 16 mantissa bits cut off: it keeps FP32's 8 exponent bits and range and pays with 7 mantissa bits, about 2.4 digits. TF32 (TensorFloat-32) is not a storage type but a Tensor Core math mode of compute capability 8.0 and later, which multiplies FP32 operands with a 10-bit mantissa. The author's GTX 1650 (compute capability 7.5) has no TF32 mode, and native BF16 instructions also start at 8.0: for `sm_75`, CUDA 12.3's `cuda_bf16.h` compiles a BF16 add or multiply into FP32 instructions and integer rounding (`cuobjdump -sass` of such a build shows `FFMA` and no BF16 instruction). The choice between formats is which of range and precision a kernel cannot afford to lose.

## 2. Addition is not associative

In exact arithmetic $(a + b) + c = a + (b + c)$. In floating point every addition rounds its exact result to the nearest representable number, and a large value plus a small one can round the small part away. FP32 has 24 significant bits, so between $2^{26}$ and $2^{27}$ its numbers are 8 apart: $10^8 + 1$ lies within 4 of $10^8$ and rounds back to it, as the left of the figure below draws. The model sums one $10^8$ and a million 1.0s in four orders:

```text
(a) addition is not associative: one 1e8, then 1,000,000 ones; exact sum 101000000
    order                          fp32 result    abs error
    left to right (big first)      100000000.0    1000000.0
    ascending (small first)        101000000.0          0.0
    pairwise (numpy sum)           100999992.0          8.0
    Kahan (compensated)            101000000.0          0.0
    -> four orders, 3 different fp32 answers; the spacing of fp32 near 1e8 is 8
```

Big first loses the whole million, one rounding at a time. Small first keeps it: a million 1.0s sum exactly to 1,000,000, a number FP32 holds, and adding $10^8$ once gives 101,000,000, which is a multiple of 8. NumPy's sum, a blocked pairwise sum, lands one spacing short at 100,999,992. Kahan's compensated sum (section 6) is exact. Three answers from four orders of the same numbers.

A parallel reduction is an order too. [Post 04](../04-reduction/index.md)'s kernels add within each block as a tree and then combine the block sums, so they add in an order a CPU loop never uses, and a different launch gives a different tree. `snippets/precision.cu` makes this concrete with `sumBlocks`, a grid-stride sum: thread `i`, with `i = blockIdx.x * blockDim.x + threadIdx.x`, adds `x[i]`, `x[i + T]`, `x[i + 2T]` and so on, $T$ being the grid's total thread count; each block of 256 threads then reduces its 256 partial sums with post 04's shared-memory tree, and the host adds the block sums in block order:

```cuda
__global__ void sumBlocks(const float* x, float* partial, int n) {
    __shared__ float s[BLOCK];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += x[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}
```

The operands are 1,048,576 floats in $[0, 1)$ from an integer hash that the program and the model compute identically, each with 24 random bits, so every value is exact in FP32 and the FP64 sum of all of them is exact. The model repeats the kernel's additions in the kernel's order for four block counts, 14 being one block per streaming multiprocessor (SM) of the GTX 1650:

```text
(b) the same 1,048,576 floats in [0, 1), summed as precision.cu's sumBlocks does
    exact sum 524781.0219
    blocks x 256 threads   fp32 sum         abs error
         1 x 256          524781.0625     0.0406
        14 x 256          524781.0000     0.0219
        56 x 256          524781.0625     0.0406
       224 x 256          524780.9375     0.0844
    one CPU loop           524778.0625     2.9594
    -> 4 block counts, 4 summation trees, 3 different fp32 sums; none is the exact one
```

The GPU, run on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5)
(b) 1048576 floats in [0, 1), exact sum 524781.0219
    blocks x 256 threads   fp32 sum         abs error
         1 x 256          524781.0625     0.0406
        14 x 256          524781.0000     0.0219
        56 x 256          524781.0625     0.0406
       224 x 256          524780.9375     0.0844
    56 blocks twice more: 524781.0625 and 524781.0625, the same bits
```

Every GPU sum equals the model's bit for bit: near 524,781 FP32 numbers are $2^{-4} = 0.0625$ apart, so four decimals print each value exactly. These are results of the arithmetic, not timings; each step is a correctly rounded IEEE addition in a fixed order, so another card running this build should print the same bits. Each block count is a different tree, and three distinct sums came out (that 1 and 56 blocks agree is chance). All four trees stay within 1.35 spacings of the exact sum, while one sequential CPU loop is 47 spacings off: its running total grows large and every later addition rounds against it.

![Two panels. Left: the FP32 number line near 10 to the 8, with neighbors 99,999,992, 100,000,000 and 100,000,008 eight apart; the exact 100,000,001 rounds back to 100,000,000. Right: bars of the absolute error of the same sum with 1, 14, 56 and 224 blocks, 0.0406, 0.0219, 0.0406 and 0.0844, against an FP32 spacing of 0.0625.](diagrams/02-sum-order.svg)

*A small addend below half a spacing vanishes; each block count rounds at different places and lands on its own neighbor of the exact sum.*

One launch is reproducible: 56 blocks run twice more gave the same bits, because the tree is fixed and the host adds the block sums in order. Post 04's kernels combine block sums with `atomicAdd`, whose additions land in an order that changes from run to run, so such a reduction can differ in its last bits between identical launches. This is arithmetic, not a bug, and it is why section 6 compares a reduction with a reference through a tolerance.

## 3. Overflow and underflow: why FP16 is fragile

Range is the other half of the budget. The model stores three values in each format, and the device's own conversions, `__float2half` and `__float2bfloat16` in a kernel, print the same digits:

```text
(c) range: FP16 overflows above 65504 and loses precision below 6.1e-05
    value       fp16          fp32          bf16
    100000      inf           100000        99840
    1e-05       1.00136e-05   1e-05         1.00136e-05
    1e-08       0             1e-08         1.00117e-08
    -> 1e-05 is an fp16 subnormal (relative error 1.4e-03); 1e-08 rounds to 0
```

```text
(c) value       fp16          bf16
    100000      inf           99840
    1e-05       1.00136e-05   1.00136e-05
    1e-08       0             1.00117e-08
```

The value 100,000 is ordinary, but it is above FP16's 65,504, so FP16 stores infinity. BF16 holds it as 99,840: near 100,000 its numbers are $2^{16-7} = 512$ apart and 99,840 is the nearest multiple of 512, coarse but finite. At the small end FP16's normal numbers stop at $2^{-14} \approx 6.1 \times 10^{-5}$; below that the standard's subnormal numbers trade precision for range, so $10^{-5}$ survives with a relative error of $1.4 \times 10^{-3}$, and below about $3 \times 10^{-8}$, half the smallest subnormal $2^{-24}$, a value rounds to zero. A zero in a chain of multiplications stays zero, and an infinity becomes a NaN (not a number) the moment it meets $\infty - \infty$, $\infty / \infty$ or $0 \times \infty$; a NaN then spreads through every operation that touches it.

![Three rows on a logarithmic axis from 10 to the minus 9 to 10 to the 6: FP16's normal range from 6.1 times 10 to the minus 5 to 65,504 and its subnormal range down to 6.0 times 10 to the minus 8; BF16 and FP32 covering the whole axis. The values 10 to the minus 8, 10 to the minus 5 and 100,000 are marked on each row: on the FP16 row, 100,000 becomes inf and 10 to the minus 8 becomes 0, in red.](diagrams/03-fp16-range.svg)

*FP16's range spans about 12 decades; one large activation or one tiny gradient falls off an edge that BF16 and FP32 do not have.*

So a network cannot simply be cast to FP16: a large activation, a growing sum of squares or the `exp()` of a large score overflows, and a small gradient underflows. The remedies are a format with more range (BF16, at the cost of precision) or an algorithm that keeps the dangerous magnitudes from arising: an FP32 accumulator (section 4), the max subtraction of a softmax (section 5), and in training the loss scaling of Micikevicius and others (2017), which scales the loss up so that small FP16 gradients stay above the underflow edge.

## 4. Mixed precision: FP16 inputs, FP32 accumulator

FP16 halves the bytes a kernel moves and doubles the values per register, but a long FP16 sum rounds at every step to 11 significant bits. Mixed precision splits the work by role: the inputs stay FP16, and the running sum is FP32. `precision.cu` computes one dot product of 4,096 FP16 values (operands in $[-0.25, 0.25)$, from the same hash) three ways, each in one block of 256 threads: thread `t` adds terms `t`, `t + 256` and so on, then the shared-memory tree finishes the sum. The FP16 kernel's loop is written as a multiply and an add:

```cuda
    for (int i = threadIdx.x; i < n; i += BLOCK)
        acc = __hadd(acc, __hmul(a[i], b[i]));           // contracted: a * b + acc, rounded once
```

The compiler does not keep them apart. `nvcc` contracts a multiply followed by an add into a fused multiply-add (FMA), one instruction that rounds once, unless told not to, and the machine code (SASS, read with `cuobjdump -sass`, encodings removed) of the FP16 kernel, two lines, and the FP32 kernel, one, shows it:

```text
        /*0240*/                   HFMA2 R6, R3.H0_H0, R4.H0_H0, R6.H0_H0 ;
        /*05d0*/                   HFMA2 R8, R9.H0_H0, R8.H0_H0, R6.H0_H0 ;
        /*0280*/                   FFMA R7, R10, R9, R7 ;
```

`cuda_fp16.h` documents `__hadd_rn` and `__hmul_rn` as the forms that prevent "floating-point contractions of mul+add into fma", so the program's second kernel, `dotFP16rn`, uses them and rounds the product and the sum apart (its SASS has `HMUL2` and `HADD2` and no `HFMA2`). The third, `dotMixed`, converts each FP16 value to FP32: a product of two 11-bit significands needs at most 22 bits, so it is exact in FP32, and only the additions round. The model repeats all three with the same rounding and prints the bound section 6 derives:

```text
(e) dot product of K = 4,096 FP16 values, 256 threads and a tree, as precision.cu
    FP64 reference -2.755099521; sum of |a_i b_i| = 62.6963
    accumulator                  result         abs error   bound gamma_d * sum|a_i b_i|
    fp16, fused (as compiled)    -2.753906250  1.193e-03   7.434e-01  within
    fp16, rounded apart (_rn)    -2.755859375  7.599e-04   7.748e-01  within
    fp32 (mixed precision)       -2.755099297  2.241e-07   8.969e-05  within
    d = 24 additions on the longest path (16 per thread, 8 in the tree; +1 when products round)
    fp32 is 5325x more accurate than fused fp16 on these operands
    over 20 other operand draws the ratio ran from 60x to 60448x (median 11857x)
    -> multiply FP16 values, accumulate in FP32; test against a bound, not with ==
```

The GPU, on the same card and toolkit:

```text
(e) dot product, K = 4096, FP64 reference -2.755099521
    accumulator                  result         abs error   bound
    fp16, fused (as compiled)    -2.753906250  1.193e-03   7.434e-01  within
    fp16, rounded apart (_rn)    -2.755859375  7.599e-04   7.748e-01  within
    fp32 (mixed precision)       -2.755099297  2.241e-07   8.969e-05  within
every result within its bound: PASS
```

The three GPU results equal the model's in every printed digit, which at nine decimals separates neighboring FP16 and FP32 values here. With the same FP16 inputs and the same tree, moving the sum into an FP32 register cut the error from $1.2 \times 10^{-3}$ to $2.2 \times 10^{-7}$, 5,325 times. That ratio is one draw: over 20 other pairs of operand vectors it ran from 60 to 60,448 times, because the FP32 error of one sum can land near zero by luck, so one ratio, this one or [post 17](../17-tensor-cores/index.md)'s on its own operands, describes its data only. On this draw the fused FP16 kernel was no more accurate than the one that rounds twice, since the FP16 additions dominate the error; the fusion matters for tests, because a reference that rounds twice will not reproduce an FMA's bits.

Mixed precision is the contract post 17's Tensor Core instructions implement: FP16 matrix inputs, products formed exactly and summed into an FP32 accumulator fragment.

## 5. Stable softmax: subtract the maximum

A softmax turns a row of scores into probabilities, $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$. Written that way it is a trap: $e^{89}$ already exceeds FP32's largest value, attention scores can be in the hundreds, and once the numerator and the sum are both infinity the ratio is a NaN. The fix is exact algebra. With the row maximum $M = \max_j x_j$,

$$\text{softmax}(x)_i = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}},$$

which multiplies numerator and denominator by the same $e^{-M}$ and so changes no probability. Now the largest exponent is $e^{0} = 1$, the sum is at least 1, and nothing overflows; a very negative $x_i - M$ underflows to 0, which is the right probability to the printed precision.

```text
(d) softmax of x = [2, 3, 1000, 1001] in fp32
    naive  exp(x) / sum      = [ 0.  0. nan nan]   (exp(1000) is inf, inf / inf is nan)
    stable exp(x - max) / sum = [0.     0.     0.2689 0.7311]
    stable against an FP64 reference: largest error 1.89e-08
```

The GPU's `softmax` kernel, one thread computing the row both ways with `expf`, prints the same:

```text
(d) naive  softmax: 0.0000 0.0000 nan nan
    stable softmax: 0.0000 0.0000 0.2689 0.7311
```

![Two pipelines on the row 2, 3, 1000, 1001. Naive: exp gives 7.389, 20.09, inf, inf, and dividing by the infinite sum gives 0, 0, nan, nan, in red. Stable: subtracting the maximum 1001 gives minus 999, minus 998, minus 1, 0; exp gives 0, 0, 0.3679, 1; dividing by 1.3679 gives 0, 0, 0.2689, 0.7311.](diagrams/04-stable-softmax.svg)

*Subtracting the row maximum caps every exponent at 0, so the row stays finite and the probabilities do not change.*

The figure follows both pipelines value by value. [Post 18](../18-flash-attention/index.md) needs this in a harder form: FlashAttention never holds a whole row, so it keeps a running maximum and rescales its partial sums each time the maximum grows, the same subtraction made online.

## 6. Better sums, and how to test a kernel

Two summation algorithms buy accuracy back. Pairwise summation adds in a balanced binary tree, so no term passes through more than about $\log_2 N$ additions and the error bound grows with $\log_2 N$ instead of $N$; NumPy's `sum` uses a blocked form of it, and a GPU reduction's tree is the same idea, which is why every tree of section 2 beat the CPU loop. Kahan's compensated summation (Kahan, 1965) keeps a second variable holding the low-order part each addition rounds away and feeds it into the next one, for a few extra operations per element; section 2's Kahan row is exact. The model's version is four lines:

```python
    for v in arr:
        y = F32(v) - c
        t = F32(s + y)
        c = F32((t - s) - y)
        s = t
```

Two compiler settings change the rounding a test sees. Contraction into FMA is on by default: `nvcc`'s documentation says `--fmad` "enables (disables) the contraction of floating-point multiplies and adds/subtracts" and that its default is true, which is why section 4's kernel became `HFMA2`. `--use_fast_math` goes further: it "implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true", flushing FP32 subnormals to zero and using approximate division and square root (it also maps functions such as `expf` to faster, less accurate intrinsics). Whitehead and Fit-Florea's NVIDIA white paper on IEEE 754 compliance (2011) explains the FMA's single rounding and notes that operations written as rounding intrinsics such as `__fmul_rn` and `__fadd_rn` "will not be merged into fused multiply-add instructions". Each setting trades last-bit agreement with a reference for speed, so it is a decision, made once and tested.

The test itself follows from the error analysis of summation (Higham, 2002, chapter 4). If every term of a sum passes through at most $d$ roundings, each with relative error at most the unit roundoff $u$ ($2^{-24}$ for FP32, $2^{-11}$ for FP16), the computed sum $\hat{s}$ obeys

$$|\hat{s} - s| \le \gamma_d \sum_i |x_i|, \qquad \gamma_d = \frac{d\,u}{1 - d\,u}.$$

In `precision.cu` each thread adds 16 terms and the tree adds 8 levels, so $d = 24$ (25 when FP16 products round too), and $\sum |a_i b_i| = 62.70$. That gives the FP32 kernel a tolerance of $8.97 \times 10^{-5}$ and the FP16 kernels about $0.75$, the bounds section 4's output prints; the program fails if a result leaves its bound. The bound is a worst case (the actual FP32 error was 400 times smaller), but it holds for any data and is sized by the two things that matter: the format, through $u$, and the order, through $d$. It is relative to $\sum |x_i|$ rather than to the result, since a sum that cancels to near zero has a tiny result and a large rounding error. Demanding `==` fails on correct code.

## 7. Make it run

From the series root, the model needs NumPy, which the series environment carries, and the program needs an NVIDIA GPU and the CUDA Toolkit:

```bash
uv run python posts/16-numerical-precision/snippets/precision_model.py
nvcc -O3 -arch=sm_75 posts/16-numerical-precision/snippets/precision.cu -o precision
./precision
```

The model prints every model block quoted above in about 3 seconds. On Windows the program is `precision.exe` and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both. The program builds in about 7 seconds, runs in about a second, and exits with status 1 if a result leaves its bound. Another card should print the same sums and dot products, since each is a fixed sequence of correctly rounded operations; the `expf` digits of the softmax could differ in the last place between toolkit versions.

## 8. What can go wrong?

- **A reference that does not match the kernel's rounding.** The first model of this post multiplied in FP16, rounding each product, while `nvcc` had fused the FP16 multiply and add into `HFMA2`; the two FP16 results then differ, and a bit-exact comparison of model and kernel fails although both are right. Reading the SASS, and writing `_rn` intrinsics where separate rounding is meant, is how to know which arithmetic ran.
- **A tolerance that ignores the format.** The FP16 kernels' errors of about $10^{-3}$ fail the FP32 bound of $9 \times 10^{-5}$ and pass their own bounds of about 0.75; a single tolerance shared by kernels of different formats is either too loose for one or too tight for the other.
- **A relative tolerance on a sum that cancels.** A dot product of operands of both signs can be near zero while its rounding error is not; the bound is relative to $\sum |x_i|$, which the test computes alongside the reference.

---

## Common pitfalls

1. **Comparing GPU output with a reference using `==`.** A parallel reduction adds in a different order from the CPU, and a different launch in a different order again; compare with an FP64 reference and a bound sized to the format and the summation depth.
2. **Casting to FP16 and hoping.** FP16 overflows above 65,504, loses precision below about $6 \times 10^{-5}$ and rounds values below about $3 \times 10^{-8}$ to zero; keep the accumulator in FP32, use BF16 where range matters more than digits, or scale values into FP16's range.
3. **Accumulating a long sum in FP16.** Every addition rounds to 11 significant bits; the FP32 accumulator was 5,325 times more accurate on one draw.
4. **Writing a softmax without the max subtraction.** `exp()` of a score near 89 already overflows FP32 and the row becomes NaN; subtracting the row maximum changes no probability.
5. **Treating FMA contraction as opt-in.** `nvcc` fuses a multiply and an add by default; `--fmad=false` or the `_rn` intrinsics keep them apart where a test needs separate rounding.
6. **Turning on `--use_fast_math` without measuring the accuracy.** It flushes subnormals to zero and approximates division, square root and the math functions; check a kernel against its bound after enabling it.

---

## Further reading

- Goldberg, D., "What Every Computer Scientist Should Know About Floating-Point Arithmetic", *ACM Computing Surveys* 23(1), 1991. Rounding, non-associativity and cancellation.
- IEEE, *IEEE Standard for Floating-Point Arithmetic* (IEEE 754-2019). The binary16, binary32 and binary64 formats, subnormal numbers, infinities and NaN.
- Higham, N. J., *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, chapter 4. Error bounds for recursive, pairwise and compensated summation.
- Kahan, W., "Pracniques: Further Remarks on Reducing Truncation Errors", *Communications of the ACM* 8(1), January 1965, page 40. Compensated summation.
- Whitehead, N. and Fit-Florea, A., "Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs", NVIDIA, 2011. The FMA, rounding intrinsics and the fast-math flags on NVIDIA GPUs.
- NVIDIA, *CUDA Compiler Driver NVCC*, options `--fmad` and `--use_fast_math`, consulted October 2026.
- Micikevicius, P. and others, "Mixed Precision Training", arXiv 1710.03740, October 2017; ICLR 2018. FP16 storage with FP32 accumulation and loss scaling in training.
- NVIDIA, "Train With Mixed Precision", NVIDIA Deep Learning Performance documentation, consulted October 2026. Practical FP16, BF16 and TF32 guidance.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 17 - Tensor cores: matrix-at-once hardware and mixed precision](../17-tensor-cores/index.md):** the matrix instructions that take FP16 inputs and accumulate in FP32, the contract this post justifies.
- **[Post 04 - Reduction: summing millions of numbers without a serial bottleneck](../04-reduction/index.md):** the parallel sum whose order, and so whose last bits, this post traced to the tree.
