"""scan_model.py — a CPU model of the two scan algorithms and stream compaction.

CUDA cannot run in this blog's test environment, so this model reproduces the
Hillis-Steele and Blelloch scans, checks them against a reference cumulative sum,
counts the work each does (the work-vs-depth trade), and runs the scan's killer
application, stream compaction, which uses an exclusive scan to compute output
positions in parallel. Run: python snippets/scan_model.py  (needs NumPy; from the post dir)
"""
import numpy as np

x = np.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=np.int64)
n = len(x)
inclusive_ref = np.cumsum(x)
exclusive_ref = inclusive_ref - x

# Hillis-Steele inclusive scan: log-stride doubling
hs = x.astype(np.int64).copy()
stride = 1
while stride < n:
    hs = hs + np.concatenate([np.zeros(stride, np.int64), hs[:-stride]])
    stride *= 2

# Blelloch exclusive scan: up-sweep then down-sweep
b = x.copy()
offset = 1
d = n >> 1
while d > 0:                                   # up-sweep
    for t in range(d):
        ai, bi = offset * (2 * t + 1) - 1, offset * (2 * t + 2) - 1
        b[bi] += b[ai]
    offset *= 2; d >>= 1
b[n - 1] = 0
d = 1
while d < n:                                   # down-sweep
    offset >>= 1
    for t in range(d):
        ai, bi = offset * (2 * t + 1) - 1, offset * (2 * t + 2) - 1
        tmp = b[ai]; b[ai] = b[bi]; b[bi] += tmp
    d *= 2

# work counts (additions)
M = 1 << 20
hs_work = M * int(np.log2(M))
bl_work = 2 * (M - 1)

# stream compaction: keep positive values, exclusive scan of the predicate = positions
data = np.array([3, -1, 7, 0, -2, 4, 1, -5, 6])
pred = (data > 0).astype(np.int64)
pos = np.concatenate([[0], np.cumsum(pred)[:-1]])    # exclusive scan
out = np.empty(int(pred.sum()), dtype=np.int64)
for i in range(len(data)):
    if pred[i]:
        out[pos[i]] = data[i]

print("scan model")
print(f"  Hillis-Steele inclusive == cumsum : {bool(np.array_equal(hs, inclusive_ref))}")
print(f"  Blelloch exclusive == reference   : {bool(np.array_equal(b, exclusive_ref))}")
print(f"  work for N=2^20: Hillis-Steele {hs_work:,} adds vs Blelloch {bl_work:,} ({hs_work/bl_work:.1f}x)")
print(f"  stream compaction: {data.tolist()}")
print(f"                  -> {out.tolist()}  (kept {len(out)} positives, in parallel)")
