"""precision_model.py: how floating-point numbers behave, and what precision.cu will print.

A NumPy model of post 16. It derives the format table from the bit widths, then runs five
experiments with the same arithmetic, the same operands and the same order of operations as
the kernels of precision.cu, so the GPU's results can be checked against it bit for bit:

  (a) Order.      One 1e8 and a million 1.0s, summed in four orders.
  (b) Blocks.     The same 2**20 floats summed the way precision.cu's sumBlocks kernel sums
                  them, for four block counts: each count gives its own last bits.
  (c) Range.      FP16 overflows past 65,504 and loses precision below 6.1e-05.
  (d) Softmax.    The naive softmax of a row with large scores is nan; the stable one is not.
  (e) Mixed.      A length-4,096 dot product of FP16 inputs, accumulated three ways, against
                  an error bound sized to the format and the summation order.

Deterministic: the operands come from an integer hash that precision.cu computes identically.
Run from the series root: uv run python posts/16-numerical-precision/snippets/precision_model.py
"""
import math

import numpy as np

np.seterr(over="ignore", invalid="ignore", under="ignore")   # inf, nan and underflow on purpose

F16, F32, F64 = np.float16, np.float32, np.float64
BLOCK = 256                                       # threads per block in precision.cu
N_SUM = 1 << 20                                   # floats in experiment (b)
GRIDS = (1, 14, 56, 224)                          # block counts of experiment (b); 14 = one per SM
K = 4096                                          # dot-product length of experiment (e)


def hash01(n, seed):
    """n floats in [0, 1) with 24 random bits each (exact in FP32), as precision.cu's hash01()."""
    h = (np.arange(n, dtype=np.uint64) * 2654435761 + seed * 0x9E3779B9) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x45D9F3B) & 0xFFFFFFFF
    h ^= h >> 16
    return ((h >> 8).astype(F64) / 2.0**24).astype(F32)


def to_bf16(x):
    """Round FP32 to BF16 (8 exponent bits, 7 mantissa bits), round to nearest even."""
    u = np.asarray(x, dtype=F32).view(np.uint32).astype(np.uint64)
    u = (u + ((u >> 16) & 1) + 0x7FFF) & 0xFFFF0000
    return u.astype(np.uint32).view(F32)


def kahan_sum(arr):
    """Compensated (Kahan) summation in FP32: c carries what each add rounded away."""
    s, c = F32(0.0), F32(0.0)
    for v in arr:
        y = F32(v) - c
        t = F32(s + y)
        c = F32((t - s) - y)
        s = t
    return s


def tree(s):
    """precision.cu's shared-memory tree: s[t] += s[t + stride] for stride 128, 64, ..., 1."""
    s = s.copy()
    stride = len(s) // 2
    while stride > 0:
        s[:stride] = s[:stride] + s[stride:2 * stride]
        stride //= 2
    return s[0]


def sum_blocks(x, grid):
    """sumBlocks<<<grid, 256>>>: thread i adds x[i], x[i + T], ... (T = grid * 256 threads),
    each block reduces its 256 sums with the tree, and the host adds the block sums in order."""
    T = grid * BLOCK
    rows = -(-len(x) // T)
    pad = np.zeros(rows * T, dtype=F32)           # adding 0.0 is exact, so padding changes nothing
    pad[:len(x)] = x
    acc = np.zeros(T, dtype=F32)
    for r in pad.reshape(rows, T):
        acc = acc + r                             # FP32 adds, one per element, in index order
    total = F32(0.0)
    for b in range(grid):
        total = F32(total + tree(acc[b * BLOCK:(b + 1) * BLOCK]))
    return total


def dot_kernel(a16, b16, mode):
    """dotFP16 / dotFP16rn / dotMixed<<<1, 256>>>: thread t adds terms t, t + 256, ... then the tree.
    'fma'  FP16 accumulator, a * b + acc rounded once (the HFMA2 nvcc emits for __hadd(acc, __hmul(a, b)))
    'rn'   FP16 accumulator, product and sum rounded apart (__hmul_rn, __hadd_rn)
    'fp32' FP32 accumulator; the product of two FP16 values is exact in FP32"""
    a, b = a16.astype(F64).reshape(-1, BLOCK), b16.astype(F64).reshape(-1, BLOCK)
    if mode == "fp32":
        acc = np.zeros(BLOCK, dtype=F32)
        for ra, rb in zip(a, b):
            acc = acc + (ra * rb).astype(F32)
        return F64(tree(acc))
    acc = np.zeros(BLOCK, dtype=F16)
    for ra, rb in zip(a, b):
        if mode == "fma":                         # exact in FP64 for these operands, then one rounding
            acc = (acc.astype(F64) + ra * rb).astype(F16)
        else:
            acc = acc + (ra * rb).astype(F16)     # FP16 + FP16 is rounded once to FP16
    return F64(tree(acc))


def dot_operands(seed_a, seed_b):
    a = ((hash01(K, seed_a) - F32(0.5)) * F32(0.5)).astype(F16)   # in [-0.25, 0.25), then FP16
    b = ((hash01(K, seed_b) - F32(0.5)) * F32(0.5)).astype(F16)
    return a, b


print("numbers on a GPU: floating-point behavior (deterministic)")
print()

# ---- the format table, derived from the bit widths ----
print("formats, derived from the bit widths (p = mantissa bits + 1, the implicit leading 1)")
print("  format  bits  exp  mantissa  max finite  min normal  eps = 2^-(p-1)  decimal digits")
FORMATS = [("FP64", 11, 52), ("FP32", 8, 23), ("TF32", 8, 10), ("FP16", 5, 10), ("BF16", 8, 7)]
facts = {}
for name, e, m in FORMATS:
    bias = 2 ** (e - 1) - 1
    mx = (2 - 2.0 ** -m) * 2.0 ** bias
    mn = 2.0 ** (1 - bias)
    eps = 2.0 ** -m
    digits = (m + 1) * math.log10(2)
    facts[name] = (mx, mn, eps)
    big = "%.0f" % mx if mx < 1e6 else "%.3e" % mx
    print("  %-6s %5d %4d %9d  %10s  %10.3e  %14.3e  %14.1f" % (name, 1 + e + m, e, m, big, mn, eps, digits))
for name, dt in (("FP64", F64), ("FP32", F32), ("FP16", F16)):
    fi = np.finfo(dt)
    assert facts[name] == (float(fi.max), float(fi.smallest_normal), float(fi.eps)), name
assert facts["BF16"][0] == float(np.uint32(0x7F7F0000).view(F32))
print("  FP16 subnormals reach %.3e; the FP64, FP32 and FP16 rows equal numpy.finfo" % np.finfo(F16).smallest_subnormal)
print()

# ---- (a) order ----
BIG, ONES = 1e8, 1_000_000
arr = np.concatenate(([F32(BIG)], np.ones(ONES, dtype=F32)))
exact = F64(BIG) + F64(ONES)
results = [("left to right (big first)", arr.cumsum(dtype=F32)[-1]),
           ("ascending (small first)", np.sort(arr).cumsum(dtype=F32)[-1]),
           ("pairwise (numpy sum)", np.sum(arr, dtype=F32)),
           ("Kahan (compensated)", kahan_sum(arr))]
print("(a) addition is not associative: one 1e8, then %s ones; exact sum 101000000" % f"{ONES:,}")
print("    order                          fp32 result    abs error")
for name, val in results:
    print("    %-28s %13.1f %12.1f" % (name, val, abs(F64(val) - exact)))
print("    -> four orders, %d different fp32 answers; the spacing of fp32 near 1e8 is %g"
      % (len({float(v) for _, v in results}), np.spacing(F32(BIG))))
print()

# ---- (b) block count ----
x = hash01(N_SUM, 7)
ref = float(np.sum(x.astype(F64)))                # exact: 2**20 values of 24 bits fit in 53
print("(b) the same %s floats in [0, 1), summed as precision.cu's sumBlocks does" % f"{N_SUM:,}")
print("    exact sum %.4f" % ref)
print("    blocks x 256 threads   fp32 sum         abs error")
sums = []
for g in GRIDS:
    s = sum_blocks(x, g)
    sums.append(float(s))
    print("    %6d x 256          %.4f     %.4f" % (g, s, abs(F64(s) - ref)))
seq = x.cumsum(dtype=F32)[-1]
print("    one CPU loop           %.4f     %.4f" % (seq, abs(F64(seq) - ref)))
print("    -> %d block counts, %d summation trees, %d different fp32 sums; none is the exact one"
      % (len(GRIDS), len(GRIDS), len(set(sums))))
print()

# ---- (c) range ----
print("(c) range: FP16 overflows above 65504 and loses precision below 6.1e-05")
print("    value       fp16          fp32          bf16")
for v in (100000.0, 1e-5, 1e-8):
    h, f, bf = F16(v), F32(v), to_bf16(F32(v))
    print("    %-9g   %-12.6g  %-12.6g  %.6g" % (v, float(h), float(f), float(bf)))
print("    -> 1e-05 is an fp16 subnormal (relative error %.1e); 1e-08 rounds to 0"
      % abs(float(F16(1e-5)) / 1e-5 - 1))
print()

# ---- (d) softmax ----
row = np.array([2.0, 3.0, 1000.0, 1001.0], dtype=F32)
naive = np.exp(row) / np.exp(row).sum()
e = np.exp(row - row.max())
stable = e / e.sum()
r64 = np.exp(row.astype(F64) - 1001.0)
r64 /= r64.sum()
print("(d) softmax of x = [2, 3, 1000, 1001] in fp32")
print("    naive  exp(x) / sum      = %s   (exp(1000) is inf, inf / inf is nan)" % np.array2string(naive, precision=4))
print("    stable exp(x - max) / sum = %s" % np.array2string(stable, precision=4))
print("    stable against an FP64 reference: largest error %.2e" % np.max(np.abs(stable - r64)))
print()

# ---- (e) mixed precision ----
a16, b16 = dot_operands(1, 2)
dref = float(a16.astype(F64) @ b16.astype(F64))   # exact for these operands
mag = float(np.abs(a16.astype(F64) * b16.astype(F64)).sum())
d = K // BLOCK + int(math.log2(BLOCK))            # additions on the longest path: 16 + 8 = 24


def gamma(n, u):
    return n * u / (1 - n * u)


rows = [("fp16, fused (as compiled)", "fma", gamma(d, 2.0**-11)),
        ("fp16, rounded apart (_rn)", "rn", gamma(d + 1, 2.0**-11)),
        ("fp32 (mixed precision)", "fp32", gamma(d, 2.0**-24))]
print("(e) dot product of K = %s FP16 values, 256 threads and a tree, as precision.cu" % f"{K:,}")
print("    FP64 reference %.9f; sum of |a_i b_i| = %.4f" % (dref, mag))
print("    accumulator                  result         abs error   bound gamma_d * sum|a_i b_i|")
errs = {}
for name, mode, gm in rows:
    r = dot_kernel(a16, b16, mode)
    errs[mode] = abs(r - dref)
    print("    %-28s %.9f  %.3e   %.3e  %s" % (name, r, errs[mode], gm * mag, "within" if errs[mode] <= gm * mag else "OUTSIDE"))
print("    d = %d additions on the longest path (16 per thread, 8 in the tree; +1 when products round)" % d)
print("    fp32 is %.0fx more accurate than fused fp16 on these operands" % (errs["fma"] / errs["fp32"]))
ratios = []
for s in range(20):
    a, b = dot_operands(101 + 2 * s, 102 + 2 * s)
    rr = float(a.astype(F64) @ b.astype(F64))
    e16 = abs(dot_kernel(a, b, "fma") - rr)
    e32 = abs(dot_kernel(a, b, "fp32") - rr)
    ratios.append(e16 / e32 if e32 > 0 else math.inf)
fin = sorted(r for r in ratios if math.isfinite(r))
print("    over 20 other operand draws the ratio ran from %.0fx to %.0fx (median %.0fx)%s"
      % (fin[0], fin[-1], float(np.median(fin)),
         "" if len(fin) == 20 else "; %d draws had no fp32 error at all" % (20 - len(fin))))
print("    -> multiply FP16 values, accumulate in FP32; test against a bound, not with ==")
