"""convolution_model.py — a CPU model of convolution reuse and separability.

CUDA cannot run in this blog's test environment, so this NumPy model checks a
2D convolution against a direct reference, then counts the two savings the post
is about:

1. Global reads for one output tile. The naive kernel re-reads a K-by-K
   neighborhood for every output pixel: TILE^2 * K^2 reads. The tiled kernel
   loads the (TILE + K - 1)^2 input tile once into shared memory, so each input
   value is read from global memory exactly once. The ratio is the reuse factor.
2. Multiply-adds for a separable filter: a K-by-K filter that factors into a row
   and a column vector costs 2K per pixel instead of K^2.

Run: python convolution_model.py  (needs NumPy)
"""
import numpy as np

TILE, K = 16, 3
R = K // 2
rng = np.random.default_rng(0)
img = rng.random((TILE + 2 * R, TILE + 2 * R), dtype=np.float32)   # padded input
mask = np.full((K, K), 1.0 / 9.0, dtype=np.float32)               # 3x3 box blur

# reference: direct convolution of the inner TILE x TILE region
ref = np.empty((TILE, TILE), dtype=np.float32)
for y in range(TILE):
    for x in range(TILE):
        ref[y, x] = np.sum(img[y:y + K, x:x + K] * mask)

# "tiled" recomputation from the same staged input must match
tiled = np.empty((TILE, TILE), dtype=np.float32)
for y in range(TILE):
    for x in range(TILE):
        tiled[y, x] = float(np.tensordot(img[y:y + K, x:x + K], mask))

naive_reads = TILE * TILE * K * K
tiled_reads = (TILE + K - 1) ** 2

print("convolution model")
print(f"  tile={TILE}x{TILE}, filter={K}x{K} (radius {R})")
print(f"  tiled == reference   : {bool(np.allclose(tiled, ref, atol=1e-5))}")
print(f"  naive global reads   : {naive_reads:,}  (re-reads each {K}x{K} neighborhood)")
print(f"  tiled global reads    : {tiled_reads:,}  (load the {TILE + K - 1}x{TILE + K - 1} tile once)")
print(f"  reuse factor         : {naive_reads / tiled_reads:.1f}x")
print(f"  separable mul-adds    : {K * K} per pixel non-separable vs {2 * K} separable")
