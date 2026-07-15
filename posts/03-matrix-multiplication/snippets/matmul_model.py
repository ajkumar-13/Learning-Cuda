"""matmul_model.py — a CPU model of why tiling helps.

CUDA cannot run in this blog's test environment, so this NumPy model reproduces
the two kernels' logic and, more importantly, counts the global-memory loads each
one issues. The naive kernel re-reads a full row of A and column of B for every
output element; the tiled kernel loads each value once per tile and reuses it
across the block. Both must equal the reference product.

It prints only real, derived counts. Run: python matmul_model.py  (needs NumPy)
"""
import numpy as np

M = K = N = 64
TILE = 16
rng = np.random.default_rng(0)
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)
ref = A @ B

# naive: one dot product per output element
C_naive = np.empty((M, N), dtype=np.float32)
for i in range(M):
    for j in range(N):
        C_naive[i, j] = np.dot(A[i, :], B[:, j])

# tiled: accumulate over K in tiles of width TILE (same arithmetic, staged)
C_tiled = np.zeros((M, N), dtype=np.float32)
for t in range(0, K, TILE):
    C_tiled += A[:, t:t + TILE] @ B[t:t + TILE, :]

# global-load model: naive reads 2K floats per output; tiled reuses each loaded
# value ~TILE times, so it reads 2K/TILE per output (amortized over the block).
naive_loads = 2 * M * N * K
tiled_loads = 2 * M * N * K // TILE

print("matmul model")
print(f"  sizes             = A[{M},{K}] x B[{K},{N}] -> C[{M},{N}], TILE={TILE}")
print(f"  naive == ref      = {np.allclose(C_naive, ref, atol=1e-3)}")
print(f"  tiled == ref      = {np.allclose(C_tiled, ref, atol=1e-3)}")
print(f"  naive global loads= {naive_loads:,}")
print(f"  tiled global loads= {tiled_loads:,}")
print(f"  reduction factor  = {naive_loads // tiled_loads}x  (= TILE)")
