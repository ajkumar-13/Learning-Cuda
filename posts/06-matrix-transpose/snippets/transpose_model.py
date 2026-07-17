"""transpose_model.py — a CPU model of coalescing and bank conflicts in a transpose.

CUDA cannot run in this blog's test environment, so this model checks that the
index-swapped transpose is correct, then counts the two costs the post is about:

1. Global-memory transactions for one 32-thread warp. A coalesced access (a row)
   touches one 128-byte line; a strided access (a column, stride = width) touches
   one line per thread, up to 32. That is the naive transpose's slow write.
2. Shared-memory bank conflicts when 32 threads read one column of the tile.
   Each 4-byte word maps to bank (address % 32). With an unpadded [32][32] tile a
   column lands entirely in one bank (32-way conflict); padding to [32][33] spreads
   it diagonally across all 32 banks (conflict-free).

Run: python transpose_model.py  (needs NumPy)
"""
import numpy as np
from collections import Counter

W = H = 64
A = np.arange(W * H, dtype=np.float32).reshape(H, W)
ref = A.T

# index-swapped transpose (what the kernel does, element by element)
B = np.empty((W, H), dtype=np.float32)
for y in range(H):
    for x in range(W):
        B[x, y] = A[y, x]

def max_same_bank(stride):
    banks = [(row * stride) % 32 for row in range(32)]   # column col=0, rows 0..31
    return max(Counter(banks).values())

print("transpose model")
print(f"  index-swap == A.T   : {bool(np.array_equal(B, ref))}")
print("  global transactions per 32-thread warp:")
print("    coalesced (a row)    : 1   (one 128-byte line)")
print("    strided   (a column) : 32  (one line per thread)")
print("  shared-memory bank conflicts on a column read (lower is better):")
print(f"    tile[32][32] unpadded: {max_same_bank(32)}-way conflict")
print(f"    tile[32][33] padded  : {max_same_bank(33)}-way (conflict-free)")
