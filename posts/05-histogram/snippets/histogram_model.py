"""histogram_model.py — a CPU model of why privatization beats global atomics.

CUDA cannot run in this blog's test environment, so this model reproduces the
two strategies and counts the contention on the hottest bin under the worst
case: a "solid color" input where every element is the same value, so every
thread targets one bin.

Global atomics put all N increments on one global address: N-way contention.
Privatization counts inside each block's shared histogram (contention bounded by
the block), then each block merges with one atomic per bin: the global hot bin
now sees only gridDim increments. It also checks a privatized count equals the
reference histogram. Run: python histogram_model.py  (needs NumPy)
"""
import numpy as np

N = 16 * 1024 * 1024
BINS, block, grid = 256, 256, 256
data = np.full(N, 128, dtype=np.int64)        # solid color: worst case
ref = np.bincount(data, minlength=BINS)

# privatized simulation: split into `grid` chunks, count each, then merge
chunks = np.array_split(data, grid)
priv = np.zeros(BINS, dtype=np.int64)
for c in chunks:
    priv += np.bincount(c, minlength=BINS)    # per-block shared histogram, then merge

global_hot = N                                # all N increments hit one global bin
shared_hot_per_block = max(len(c) for c in chunks)   # contention inside a block (shared)
global_merge = grid * BINS                    # one global atomic per bin per block
global_hot_after = grid                       # the hot global bin now sees grid adds

print("histogram model (solid-color worst case)")
print(f"  N                       = {N:,} elements, all value 128")
print(f"  privatized == reference : {bool(np.array_equal(priv, ref))}")
print(f"  global atomics on bin128 : {global_hot:,}  (all serialize on one address)")
print(f"  shared atomics / block   : {shared_hot_per_block:,}  (fast, ~20 cycles, stays local)")
print(f"  global merge atomics     : {global_merge:,}  ({grid} blocks x {BINS} bins)")
print(f"  global contention on bin128 drops from {global_hot:,} to {global_hot_after}")
