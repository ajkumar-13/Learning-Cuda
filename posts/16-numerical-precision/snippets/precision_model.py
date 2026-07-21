"""precision_model.py — how floating-point numbers behave on a GPU.

CUDA cannot run in this blog's test environment, so this NumPy model uses the
SAME arithmetic a kernel would and prints REAL computed numbers for four facts:

  (a) Non-associativity. The same values summed in different orders give
      different answers, which is why a parallel reduction (Post 04) and a
      sequential one disagree, and why two block counts disagree.
  (b) FP16 overflow. FP16 tops out at 65,504; a larger value becomes inf,
      while FP32 and bf16 (FP32's exponent range) hold it.
  (c) Stable softmax. Naive softmax of a vector with a large max overflows
      exp() to inf and produces nan; subtracting the row max first is finite
      and correct.
  (d) Mixed precision. A length-K dot product accumulated in FP16 loses the
      small bits; multiplying in FP16 but accumulating in FP32 stays close to
      the FP64 reference.

Deterministic: fixed construction and seed=0, so it reproduces identically.
Run: python precision_model.py   (needs NumPy)
"""
import numpy as np

np.seterr(over="ignore", invalid="ignore")   # we intentionally trigger inf/nan


def to_bf16(x):
    """Round FP32 to bfloat16 (8 exponent bits, 7 mantissa bits), round-to-even.

    NumPy has no native bf16, so we truncate the low 16 bits of the FP32 bit
    pattern with a round-to-nearest-even bias. bf16 keeps FP32's exponent, so
    its range is FP32's range; only precision (mantissa) is cut.
    """
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    bias = ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)
    u = (u + bias) & np.uint64(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32)


def kahan_sum(arr):
    """Compensated (Kahan) summation in FP32: track the lost low-order bits."""
    s = np.float32(0.0)
    c = np.float32(0.0)                       # running compensation
    for v in arr:
        y = np.float32(v) - c
        t = np.float32(s + y)
        c = np.float32((t - s) - y)           # what the add just dropped
        s = t
    return s


print("numbers on a GPU: floating-point behavior (deterministic, seed=0)")
print()

# ---- format facts, straight from numpy's finfo ----
f16, f32 = np.finfo(np.float16), np.finfo(np.float32)
bf16_max = np.uint32(0x7F7F0000).view(np.float32)     # largest finite bf16 bit pattern
print("format facts (numpy finfo)")
print("  FP16 : max = %-10g  eps = %.3e  (~3 decimal digits)" % (f16.max, f16.eps))
print("  FP32 : max = %-10.4e  eps = %.3e  (~7 decimal digits)" % (f32.max, f32.eps))
print("  bf16 : max = %-10.4e  eps ~ 7.8e-03  (FP32's exponent, ~2-3 digits)"
      % bf16_max)
print()

# ---- (a) floating-point addition is NOT associative ----
BIG, ONES = 1e8, 1_000_000
arr = np.concatenate(([np.float32(BIG)], np.ones(ONES, dtype=np.float32)))
exact = np.float64(BIG) + np.float64(ONES)                 # 101,000,000 exactly

lr     = arr.astype(np.float32).cumsum(dtype=np.float32)[-1]     # left-to-right
asc    = np.sort(arr).astype(np.float32).cumsum(dtype=np.float32)[-1]  # small first
pair   = np.sum(arr, dtype=np.float32)                     # numpy pairwise/tree
kahan  = kahan_sum(arr)

print("(a) floating-point addition is NOT associative")
print("    sum of one big value 1e8 followed by %s ones; exact = %.1f"
      % (f"{ONES:,}", exact))
print("    order / method                 fp32 result        abs error")
print("    " + "-" * 56)
for name, val in [("left-to-right (big first)", lr),
                  ("sorted ascending (small 1st)", asc),
                  ("pairwise (numpy sum)", pair),
                  ("Kahan (compensated)", kahan)]:
    print("    %-30s %12.1f   %12.1f" % (name, val, abs(np.float64(val) - exact)))
print("    -> same numbers, four orders, four different fp32 answers")
print()

# ---- (b) FP16 overflows easily ----
v = 100000.0
print("(b) FP16 overflows easily (max 65504)")
print("    x = %.1f" % v)
print("      fp16 : %-12s (overflow: %.0f > 65504)" % (str(np.float16(v)), v))
print("      fp32 : %-12.1f (fine)" % np.float32(v))
print("      bf16 : %-12.1f (finite: bf16 keeps FP32's exponent range)"
      % to_bf16(np.float32(v)))
print()

# ---- (c) stable softmax: subtract the row max before exp() ----
x = np.array([2.0, 3.0, 1000.0, 1001.0], dtype=np.float32)


def softmax_naive(x):
    e = np.exp(x)                             # exp(1000) -> inf in fp32
    return e / e.sum()


def softmax_stable(x):
    e = np.exp(x - x.max())                   # largest exponent is exp(0)=1
    return e / e.sum()


ref = np.exp(x.astype(np.float64) - x.astype(np.float64).max())
ref = ref / ref.sum()
naive, stable = softmax_naive(x), softmax_stable(x)
print("(c) softmax stability: subtract the row max before exp()")
print("    x = %s" % np.array2string(x, precision=1))
print("    naive  softmax = %s   (exp overflowed to inf)"
      % np.array2string(naive, precision=4))
print("    stable softmax = %s"
      % np.array2string(stable, precision=4))
print("    stable vs FP64 reference: max err %.2e"
      % np.max(np.abs(stable.astype(np.float64) - ref)))
print()

# ---- (d) mixed-precision dot product ----
K = 4096
rng = np.random.default_rng(0)
a = (rng.standard_normal(K) * 0.1).astype(np.float32)
b = (rng.standard_normal(K) * 0.1).astype(np.float32)
a16, b16 = a.astype(np.float16), b.astype(np.float16)
ref_dot = float(a16.astype(np.float64) @ b16.astype(np.float64))   # exact on the fp16 operands

acc16 = np.float16(0.0)
for i in range(K):
    acc16 = np.float16(acc16 + np.float16(a16[i] * b16[i]))         # multiply AND accumulate in fp16
acc32 = np.float32(0.0)
for i in range(K):
    acc32 = np.float32(acc32 + np.float32(a16[i]) * np.float32(b16[i]))  # fp16 in, fp32 accumulate

err16 = abs(float(acc16) - ref_dot)
err32 = abs(float(acc32) - ref_dot)
print("(d) mixed-precision dot product, length K=%d (fp16 inputs)" % K)
print("    accumulate in    result        abs error vs FP64")
print("    " + "-" * 48)
print("    fp16             %-12.5f  %.3e" % (float(acc16), err16))
print("    fp32             %-12.5f  %.3e  (%.0fx more accurate)"
      % (float(acc32), err32, err16 / err32))
print("    -> multiply in FP16 for speed, accumulate in FP32 for accuracy")
