"""matmul_model.py: a CPU model of the two matrix-multiply kernels of post 03 (no GPU needed).

It reproduces, in Python with NumPy, what the kernels in matmul.cu do, and prints only counts it derives:

1. The 2D launch: a 16 x 16 block per tile of the output, the grid sized by a ceiling division in each
   dimension, the (row, col) of one worked thread, and the surplus threads the guard switches off.
2. The naive kernel, one thread per output element, run thread by thread with every global load counted.
3. The tiled kernel, one block at a time: the block's threads copy a TILE x TILE tile of A into s_A and
   of B into s_B (each load counted), then every thread reads the tiles from shared memory.
Both results are compared with NumPy's product, and the counts are turned into loads and bytes per
output element and FLOP per byte loaded.

Run from the series root:  uv run python posts/03-matrix-multiplication/snippets/matmul_model.py
"""
import numpy as np

TILE = 16                       # tile width, and the block is TILE x TILE threads
M = K = N = 64                  # A: [M, K], B: [K, N], C: [M, N]
BYTES = 4                       # one float


def ceil_div(a, b):
    return (a + b - 1) // b


def global_rc(block_xy, thread_xy, block_dim=TILE):
    """(row, col) of a thread: row from the .y components, col from the .x components."""
    (bx, by), (tx, ty) = block_xy, thread_xy
    return by * block_dim + ty, bx * block_dim + tx


def naive(A, B):
    """Thread (row, col) reads row `row` of A and column `col` of B straight from global memory."""
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n), dtype=np.float32)
    loads = 0
    for row in range(m):
        for col in range(n):
            s = np.float32(0.0)
            for kk in range(k):
                s += A[row, kk] * B[kk, col]
                loads += 2                      # one load of A, one of B
            C[row, col] = s
    return C, loads


def tiled(A, B):
    """Each block walks K in steps of TILE; per step every thread loads one element into s_A and one
    into s_B (zero outside the matrix), the block waits at a barrier, then each thread adds TILE
    products read from the shared tiles."""
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n), dtype=np.float32)
    loads = 0
    for by in range(ceil_div(m, TILE)):
        for bx in range(ceil_div(n, TILE)):
            acc = np.zeros((TILE, TILE), dtype=np.float32)        # one register `sum` per thread
            for t in range(ceil_div(k, TILE)):
                s_A = np.zeros((TILE, TILE), dtype=np.float32)
                s_B = np.zeros((TILE, TILE), dtype=np.float32)
                for ty in range(TILE):
                    for tx in range(TILE):
                        row, col = global_rc((bx, by), (tx, ty))
                        a_col, b_row = t * TILE + tx, t * TILE + ty
                        if row < m and a_col < k:
                            s_A[ty, tx] = A[row, a_col]
                            loads += 1
                        if b_row < k and col < n:
                            s_B[ty, tx] = B[b_row, col]
                            loads += 1
                # __syncthreads(): the tiles are complete before any thread reads them
                acc += s_A @ s_B                  # thread (ty, tx) adds sum_k s_A[ty, k] * s_B[k, tx]
                # __syncthreads(): every thread is done before the next step overwrites the tiles
            r0, c0 = by * TILE, bx * TILE
            C[r0:r0 + TILE, c0:c0 + TILE] = acc[:min(TILE, m - r0), :min(TILE, n - c0)]
    return C, loads


rng = np.random.default_rng(0)
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)
ref = A @ B
C_naive, naive_loads = naive(A, B)
C_tiled, tiled_loads = tiled(A, B)
assert naive_loads == 2 * M * N * K and tiled_loads == 2 * M * N * K // TILE

print("1. launch: 16 x 16 blocks over a 1000 x 1000 output")
SIDE = 1000
g = ceil_div(SIDE, TILE)
print(f"  gridDim     = ({g}, {g}) blocks of ({TILE}, {TILE}) threads")
print(f"  threads     = {g * TILE} x {g * TILE} = {(g * TILE) ** 2:,}")
print(f"  surplus     = {(g * TILE) ** 2:,} - {SIDE * SIDE:,} = {(g * TILE) ** 2 - SIDE * SIDE:,}"
      "  (fail row < M && col < N)")
row, col = global_rc((2, 1), (5, 3))
print(f"  thread blockIdx = (2, 1), threadIdx = (5, 3): row = 1 * 16 + 3 = {row}, col = 2 * 16 + 5 = {col}")

print(f"2. global loads, A[{M},{K}] x B[{K},{N}] -> C[{M},{N}], TILE = {TILE}")
print(f"  naive == reference      : {bool(np.allclose(C_naive, ref, rtol=1e-5))}")
print(f"  tiled == reference      : {bool(np.allclose(C_tiled, ref, rtol=1e-5))}")
print(f"  naive global loads      : {naive_loads:,}")
print(f"  tiled global loads      : {tiled_loads:,}")
print(f"  reduction factor        : {naive_loads // tiled_loads}  (= TILE)")

per_out_naive = naive_loads // (M * N)
per_out_tiled = tiled_loads // (M * N)
print("3. per output element (2K FLOP each)")
print(f"  naive : {per_out_naive} loads = {per_out_naive * BYTES} bytes, "
      f"{2 * K / (per_out_naive * BYTES):.2f} FLOP per byte loaded")
print(f"  tiled : {per_out_tiled} loads = {per_out_tiled * BYTES} bytes, "
      f"{2 * K / (per_out_tiled * BYTES):.2f} FLOP per byte loaded")

n = 1024
print(f"4. the same counts at {n} x {n} x {n} (2 * M * N * K loads naive, divided by TILE tiled)")
print(f"  naive loads = {2 * n ** 3:,} = {2 * n ** 3 * BYTES / 2 ** 30:.0f} GiB of requests; "
      f"A and B hold {n * n * BYTES // 2 ** 20} MiB each")
print(f"  tiled loads = {2 * n ** 3 // TILE:,} = {2 * n ** 3 * BYTES // TILE / 2 ** 20:.0f} MiB of requests")
print(f"  FLOP        = 2 * {n}^3 = {2 * n ** 3:,}")
