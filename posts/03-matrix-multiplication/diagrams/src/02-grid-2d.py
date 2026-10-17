"""Post 03, section 2: a 2D grid of 16 x 16 blocks over the output, and the (row, col) of one thread.

Run from the series root:  uv run python posts/03-matrix-multiplication/diagrams/src/02-grid-2d.py
Writes posts/03-matrix-multiplication/diagrams/02-grid-2d.svg.
cuda_fig_design.md section 1 (a launch as a compact grid, one block opened into its threads) in 2D. Every number comes
from snippets/matmul_model.py: TILE, the ceiling division for a 1000 x 1000 output (63 blocks a side), the surplus
threads (16,064), and the worked thread blockIdx (2, 1), threadIdx (5, 3) -> (row, col) = (19, 37).
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure, num  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import matmul_model as mm  # noqa: E402

T = mm.TILE
SIDE = mm.SIDE
G = mm.ceil_div(SIDE, T)
SURPLUS = (G * T) ** 2 - SIDE * SIDE
BX, BY, TX, TY = 2, 1, 5, 3
ROW, COL = mm.global_rc((BX, BY), (TX, TY))
assert (T, SIDE, G, SURPLUS, ROW, COL) == (16, 1000, 63, 16064, 19, 37)
BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "row = 1 * 16 + 3 = 19, col = 2 * 16 + 5 = 37" in BODY

fig = Figure(
    "02-grid-2d", "A 2D grid of 16 × 16 blocks: every thread finds (row, col)",
    f"Left, the top-left corner of the grid over a {num(SIDE)} by {num(SIDE)} output C: block cells in 4 rows "
    f"(blockIdx.y 0 to 3) and 5 columns (blockIdx.x 0 to 4), with block ({BX}, {BY}) emphasized; the full grid is "
    f"{G} by {G} blocks. Middle, that block opened into {T} by {T} threads, with thread ({TX}, {TY}) emphasized. "
    f"Right, a card works its index: row = blockIdx.y {BY} times blockDim.y {T} plus threadIdx.y {TY} = {ROW}, col = "
    f"blockIdx.x {BX} times blockDim.x {T} plus threadIdx.x {TX} = {COL}. A note says the {G} by {G} grid launches "
    f"{num((G * T) ** 2)} threads for {num(SIDE * SIDE)} outputs, so {num(SURPLUS)} fail the guard.",
    subtitle=f"Derived by snippets/matmul_model.py: a {num(SIDE)} × {num(SIDE)} output, blocks of {T} × {T} threads.",
    data_w=True)

# 1. the corner of the grid of blocks
GX, GY, CB = 96, 160, 40
fig.text(40, 128, "grid", "head")
fig.text(96, 128, f"gridDim = ({G}, {G})", "code")
gb = fig.grid(GX, GY, 4, 5, cell=CB, fill=lambda i, j: "compute-soft" if (i, j) == (BY, BX) else None)
gb.window(BY, BX, 1, 1, "compute")
gb.col_labels({j: str(j) for j in range(5)}, side="below")
gb.row_labels({i: str(i) for i in range(4)}, style="tick")
fig.text(GX + 5 * CB // 2, GY + 4 * CB + 48, "blockIdx.x", "code", anchor="middle")
fig.text(GX - 40, GY - 12, "blockIdx.y", "code")
fig.text(GX + 5 * CB // 2, GY + 4 * CB + 76, f"and so on, to {G} a side", "tick", anchor="middle")

# 2. block (2, 1) opened into its threads
CT = 16
TXo, TYo = 376, 160
gt = fig.grid(TXo, TYo, T, T, cell=CT, lines_on_top=True,
              fill=lambda i, j: "compute" if (i, j) == (TY, TX) else "compute-soft")
fig.zoom(gb.cell(BY, BX), gt.box)
gt.col_labels({0: "0", TX: str(TX), T - 1: str(T - 1)}, side="below")
gt.row_labels({0: "0", TY: str(TY), T - 1: str(T - 1)}, side="right", style="tick")
fig.text(TXo + T * CT // 2, TYo + T * CT + 48, "threadIdx.x", "code", anchor="middle")
fig.text(TXo, TYo - 12, f"block ({BX}, {BY}): {T} × {T} threads", "label")
fig.text(TXo + T * CT + 24, TYo - 12, "threadIdx.y", "code", anchor="middle")

# 3. the worked index
card = fig.card(672, 160, 248, None, heading=f"thread ({TX}, {TY}) of block ({BX}, {BY})",
                lines=[f"row = {BY} * {T} + {TY} = {ROW}", f"col = {BX} * {T} + {TX} = {COL}"],
                style="code", color="compute", fit=True)
fig.note(Box(672, card.bottom + 16, 248, 0),
         ["row from the .y parts,", "col from the .x parts"])
fig.text(672, 360, f"{num((G * T) ** 2)} threads launched,", "label")
fig.text(672, 384, f"{num(SURPLUS)} past the edge fail", "label", color="hazard")
fig.text(672, 408, "row < M && col < N", "code", color="hazard")
fig.caption(f"Blocks come whole: {G} × {T} = {num(G * T)} threads a side for {num(SIDE)} outputs.")
fig.write()
