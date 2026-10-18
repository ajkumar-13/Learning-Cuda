"""Post 06, section 4: a tile is loaded by rows, turned inside shared memory, and stored by rows.

Run from the series root:  uv run python posts/06-matrix-transpose/diagrams/src/02-tiled-transpose.py
Writes posts/06-matrix-transpose/diagrams/02-tiled-transpose.svg.
cuda_fig_design.md section 3: a tile of A, the shared-memory tile and the tile of B are memory (green cells); one
warp's accesses are compute (orange outlines: solid for the load, dashed for the store); the barrier between the
load and the store is sync (a purple bar). A 4 x 4 tile holding 0 to 15 stands for the 32 x 32 one, as the
subtitle says. The values are computed here: the tile of B is the transpose of the tile of A, and the store's row 0
of B is asserted to be the shared tile's column 0, the index swap tile[threadIdx.x][threadIdx.y + j] of the post.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "tile[threadIdx.y + j][threadIdx.x] = A[(row + j) * N + col];" in BODY
assert "B[(row + j) * M + col] = tile[threadIdx.x][threadIdx.y + j];" in BODY
T = 4
A = [[r * T + c for c in range(T)] for r in range(T)]
TILE = [row[:] for row in A]                                   # the load copies rows into rows
B = [[TILE[c][r] for c in range(T)] for r in range(T)]         # the store reads tile[tx][ty]: B = A transposed
assert B[0] == [TILE[r][0] for r in range(T)] == [0, 4, 8, 12] and all(B[r][c] == A[c][r] for r in range(T) for c in range(T))

fig = Figure(
    "02-tiled-transpose", "Load rows, turn the tile in shared memory, store rows",
    "Three 4 by 4 grids. The tile of A holds 0 to 15 by rows; a warp loads its row 0, 0 1 2 3, into row 0 of the "
    "shared-memory tile, a coalesced read. A purple bar, __syncthreads(), separates the load from the store. After it "
    "the warp reads column 0 of the tile, 0 4 8 12, and writes it as row 0 of the tile of B, a coalesced write; B's "
    "tile holds the transpose. The tile of A sits at block row blockIdx.y and block column blockIdx.x, the tile of "
    "B at block row blockIdx.x and block column blockIdx.y.",
    subtitle="A 4 × 4 tile stands for the 32 × 32 one; one warp's accesses outlined.", data_w=True)

CELL, Y = 48, 160
XA, XT, XB = 40, 376, 712
heads = [(XA, "Tile of A, global"), (XT, "tile, shared memory"), (XB, "Tile of B, global")]
for x, h in heads:
    fig.text(x, Y - 24, h, "head")
fill = lambda i, j: "memory-soft"  # noqa: E731
ga = fig.grid(XA, Y, T, T, cell=CELL, values=A, fill=fill)
gt = fig.grid(XT, Y, T, T, cell=CELL, values=TILE, fill=fill)
gb = fig.grid(XB, Y, T, T, cell=CELL, values=B, fill=fill)

load, store = "compute", "compute"
fig.outline(ga.span(0, 0, 1, T), load, width=2.5)
fig.outline(gt.span(0, 0, 1, T), load, width=2.5)
fig.outline(gt.span(0, 0, T, 1).inset(4), store, width=1.5, dash="proj")
fig.outline(gb.span(0, 0, 1, T).inset(4), store, width=1.5, dash="proj")

fig.arrow((XA + T * CELL + 8, Y + 24), (XT - 8, Y + 24), label="load", color="compute")
BAR = 640
fig.fill(Box(BAR - 4, Y - 8, 8, T * CELL + 40), "sync")
fig.text(BAR, Y + T * CELL + 64, "__syncthreads()", "code", anchor="middle")
LANE = Y + T * CELL + 28                                       # the store leaves the tile's column 0, below the grids
fig.arrow((XT + CELL // 2, Y + T * CELL + 4), (XB - 8, Y + 24), color="compute", dash="proj",
          via=[(XT + CELL // 2, LANE), (XB - 24, LANE), (XB - 24, Y + 24)])
fig.text(XT + CELL, LANE - 8, "store column 0 as row 0", "note", color="compute")

fig.text(XA, Y + T * CELL + 32, "tile[ty + j][tx] = A[...]", "code")
fig.text(XB, Y + T * CELL + 32, "B[...] = tile[tx][ty + j]", "code")
fig.text(XA, Y + T * CELL + 64, "at block row blockIdx.y, column blockIdx.x", "note")
fig.text(XB, Y + T * CELL + 64, "at block row blockIdx.x,", "note")
fig.text(XB, Y + T * CELL + 84, "column blockIdx.y", "note")
fig.legend(40, 468, [dict(color=load, label="a warp loads a row", mark="outline"),
                     dict(color=store, label="it stores a tile column as a row of B", mark="dashed-outline"),
                     dict(color="sync", label="barrier")], direction="row")
fig.caption("Both global accesses are rows; the transpose happens inside shared memory.")
fig.write()
