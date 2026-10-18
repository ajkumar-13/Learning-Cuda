"""Post 06, section 5: a column of the unpadded tile sits in one bank; padding one column spreads it over 32.

Run from the series root:  uv run python posts/06-matrix-transpose/diagrams/src/03-bank-conflict.py
Writes posts/06-matrix-transpose/diagrams/03-bank-conflict.svg.
cuda_fig_design.md section 3: the tile is shared memory, each cell printed with the bank of its word; the column a
warp reads is outlined, red (hazard) where every element sits in one bank, green (memory) where the elements spread
over distinct banks. The top-left 8 x 8 corner of the 32 x 32 tile is drawn, as the subtitle says. Every bank is
computed here as (row * width + col) % 32, and the passes per column read come from snippets/transpose_model.py
(passes, tile_access), asserted against the post.
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its blocks on import
    from transpose_model import BANKS, TILE, banks, smem  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert banks == {32: (1, 32), 33: (1, 1)} and (smem[32], smem[33]) == (4096, 4224)
assert "tile[32][32] write a row: 1-way (conflict-free); read a column: 32-way conflict" in BODY
assert "tile[32][33] write a row: 1-way (conflict-free); read a column: 1-way (conflict-free)" in BODY
SHOWN = 8


def bank(width, r, c):
    return (r * width + c) % BANKS


fig = Figure(
    "03-bank-conflict", "Padding spreads a tile column over all 32 banks",
    f"Two grids, the top-left {SHOWN} by {SHOWN} corner of the {TILE} by {TILE} tile, each cell printed with the bank "
    "of its word. Unpadded, tile[32][32]: a cell's bank is its column, so column 0 is bank 0 in every row, outlined "
    "red: a warp reading the column makes 32 requests to bank 0, a 32-way conflict, 32 passes. Padded, tile[32][33]: "
    "a cell's bank is row plus column modulo 32, so column 0 holds banks 0 to 7 here and 0 to 31 over the whole "
    "column, outlined green: one pass. The pad costs 4,224 bytes against 4,096 per block.",
    subtitle=f"The top-left {SHOWN} × {SHOWN} corner of the {TILE} × {TILE} tile; each cell shows its bank, "
             "word index mod 32.",
    height=720, data_w=True)

CELL, Y = 40, 192
panels = [(80, TILE, "tile[32][32]", "bank = column", "hazard"),
          (560, TILE + 1, "tile[32][33]", "bank = (row + column) mod 32", "memory")]
for x, width, name, rule, hue in panels:
    fig.text(x - 40, Y - 56, name, "code")
    fig.text(x - 40 + 128, Y - 56, rule, "label")
    g = fig.grid(x, Y, SHOWN, SHOWN, cell=CELL, values=lambda i, j, w=width: bank(w, i, j),
                 fill=lambda i, j, h=hue: f"{h}-soft" if j == 0 else None)
    g.col_labels([str(c) for c in range(SHOWN)], side="above")
    g.row_labels([str(r) for r in range(SHOWN)], side="left")
    g.window(0, 0, SHOWN, 1, hue, form="grid")
fig.text(40, Y - 8, "row", "tick")
fig.text(520, Y - 8, "row", "tick")
for x in (80, 560):
    fig.text(x + SHOWN * CELL // 2, Y - 32, "column", "tick", anchor="middle")

ty = Y + SHOWN * CELL + 40
fig.text(40, ty, "A warp reads column 0: banks 0, 0, 0, and so on", "label")
fig.text(40, ty + 24, f"{banks[TILE][1]} words in bank 0: a {banks[TILE][1]}-way conflict, {banks[TILE][1]} passes",
         "label", color="hazard")
fig.text(520, ty, "A warp reads column 0: banks 0, 1, 2, up to 31", "label")
fig.text(520, ty + 24, f"{BANKS} different banks: {banks[TILE + 1][1]} pass", "label", color="memory")
fig.text(520, ty + 56, f"cost: {smem[TILE + 1]:,} bytes per block, not {smem[TILE]:,}", "note")
fig.caption("The pad column is never read; it only shifts each row one bank along.")
fig.write()
