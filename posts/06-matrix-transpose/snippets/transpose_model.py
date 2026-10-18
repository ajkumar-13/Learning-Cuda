"""transpose_model.py: a CPU model of the two costs in a matrix transpose, and of the tiled kernel's indexing.

It derives, without a GPU, the numbers post 06 reasons with:

1. Global memory for one warp of 32 threads moving 4-byte floats of a 4096 x 4096 row-major matrix. A row
   is 32 neighboring addresses; a column is 32 addresses 4096 floats apart. For each, the model counts the
   distinct 128-byte lines and 32-byte sectors the warp touches (on compute capability 6.0 and later a warp's
   accesses are served as the 32-byte transactions they need) and the share of those bytes the warp uses.
2. Shared-memory banks for one warp reading and writing the tile. Each 4-byte word lives in bank
   (word index % 32); lanes that hit different words of one bank are served one after another, so the
   access takes as many passes as the largest number of distinct words in one bank.
3. The tiled transpose of transpose.cu run block by block on the CPU: blocks of 32 x 8 threads, each thread
   moving 4 elements of a 32 x 32 tile, on a matrix whose sides are not multiples of 32, compared with A.T.

Run from the series root:  uv run python posts/06-matrix-transpose/snippets/transpose_model.py
"""
from collections import defaultdict

import numpy as np

WARP, BANKS, WORD = 32, 32, 4          # lanes, shared-memory banks, bytes per float
LINE, SECTOR = 128, 32                 # bytes
TILE, BLOCK_ROWS = 32, 8
N_BIG = 4096                           # the matrix side transpose.cu times


def global_cost(addresses):
    """Lines and sectors one warp touches, the bytes those sectors hold, and the share the warp uses."""
    lines = {a // LINE for a in addresses}
    sectors = {a // SECTOR for a in addresses}
    moved = len(sectors) * SECTOR
    return len(lines), len(sectors), moved, WARP * WORD / moved


def passes(words):
    """Passes a shared-memory access takes: the most distinct words that fall in one bank."""
    per_bank = defaultdict(set)
    for w in words:
        per_bank[w % BANKS].add(w)
    return max(len(s) for s in per_bank.values())


def tile_access(width):
    """Passes for the warp's row write tile[ty + j][tx] and column read tile[tx][ty + j], row length width."""
    ty_j = 0                                            # the warp's row; every row gives the same answer
    write = passes([ty_j * width + tx for tx in range(WARP)])
    read = passes([tx * width + ty_j for tx in range(WARP)])
    return write, read


def tiled_transpose(A):
    """transposePadded on the CPU, thread by thread, with its guards; returns B and how often each slot was written."""
    M, N = A.shape
    B = np.zeros((N, M), dtype=A.dtype)
    writes = np.zeros((N, M), dtype=int)
    for by in range((M + TILE - 1) // TILE):
        for bx in range((N + TILE - 1) // TILE):
            tile = np.zeros((TILE, TILE + 1), dtype=A.dtype)
            for ty in range(BLOCK_ROWS):               # phase 1: every thread loads 4 elements
                for tx in range(TILE):
                    col, row = bx * TILE + tx, by * TILE + ty
                    for j in range(0, TILE, BLOCK_ROWS):
                        if row + j < M and col < N:
                            tile[ty + j][tx] = A[row + j, col]
            for ty in range(BLOCK_ROWS):               # after the barrier: every thread stores 4 elements
                for tx in range(TILE):
                    col, row = by * TILE + tx, bx * TILE + ty
                    for j in range(0, TILE, BLOCK_ROWS):
                        if row + j < N and col < M:
                            B[row + j, col] = tile[tx][ty + j]
                            writes[row + j, col] += 1
    return B, writes


row = [(0 * N_BIG + c) * WORD for c in range(WARP)]       # 32 neighbors in row 0
col = [(r * N_BIG + 0) * WORD for r in range(WARP)]       # 32 rows of column 0
cost = {"row": global_cost(row), "column": global_cost(col)}
banks = {TILE: tile_access(TILE), TILE + 1: tile_access(TILE + 1)}
smem = {w: TILE * w * WORD for w in (TILE, TILE + 1)}
M_SMALL, N_SMALL = 70, 45
A = np.arange(M_SMALL * N_SMALL, dtype=np.float32).reshape(M_SMALL, N_SMALL)
B, writes = tiled_transpose(A)
ok = bool(np.array_equal(B, A.T))
once = bool((writes == 1).all())


def way(n):
    return "1-way (conflict-free)" if n == 1 else f"{n}-way conflict"


print("transpose model")
print(f"1. one warp, {WARP} lanes of 4-byte floats, A: [{N_BIG}, {N_BIG}] row-major")
print("                       128-byte lines  32-byte sectors  sector bytes    used")
for name, label in (("row", "a row (coalesced)"), ("column", "a column (strided)")):
    n_lines, n_sec, moved, used = cost[name]
    print(f"  {label:<20} {n_lines:>14} {n_sec:>16} {moved:>13,} {used:>6.1%}")
print(f"2. shared-memory banks for one warp ({BANKS} banks of 4-byte words)")
for w in (TILE, TILE + 1):
    name = f"tile[{TILE}][{w}]"
    print(f"  {name:<12} write a row: {way(banks[w][0])}; read a column: {way(banks[w][1])}")
extra = smem[TILE + 1] / smem[TILE] - 1
print(f"  shared memory per block: {smem[TILE]:,} bytes unpadded, {smem[TILE + 1]:,} padded ({extra:.1%} more)")
print(f"3. the tiled transpose on the CPU, A: [{M_SMALL}, {N_SMALL}], blocks of {TILE} x {BLOCK_ROWS} threads")
print(f"  grid {(N_SMALL + TILE - 1) // TILE} x {(M_SMALL + TILE - 1) // TILE} blocks, "
      f"{TILE // BLOCK_ROWS} elements per thread")
print(f"  every element of B written exactly once: {once}")
print(f"  B == A.T: {ok}")
