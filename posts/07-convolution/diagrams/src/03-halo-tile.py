"""Post 07, section 4: a 16 x 16 output tile loads an 18 x 18 input tile, its halo one pixel wide.

Run from the series root:  uv run python posts/07-convolution/diagrams/src/03-halo-tile.py
Writes posts/07-convolution/diagrams/03-halo-tile.svg.
cuda_fig_design.md section 3: the tile in shared memory, s, is drawn cell by cell; its inner 16 x 16 cells (the
threads that compute an output) are orange, the one-pixel halo ring (threads that only load) is red-edged, the barrier
between the load and the compute is a purple bar. The counts (324 threads, 256 computing, 2,304 against 324 reads,
7.1, 16.0 and 25.9 reuse) come from snippets/convolution_model.py and are asserted against the body.
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
with contextlib.redirect_stdout(io.StringIO()) as printed:   # the model prints its tables on import
    import convolution_model as model  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
OUT = printed.getvalue()
TILE, K = model.TILE, 3
R, BLK = K // 2, model.TILE + K - 1
ROWS = []
for k in (3, 5, 7):
    blk = TILE + k - 1
    ROWS.append((k, TILE * TILE * k * k, blk * blk, TILE * TILE * k * k / blk ** 2))
assert (TILE, BLK, ROWS[0][1], ROWS[0][2]) == (16, 18, 2304, 324)
assert [round(r[3], 1) for r in ROWS] == [7.1, 16.0, 25.9]
for line in OUT.splitlines()[5:9]:
    assert line in BODY, line                       # the model's reuse table is pasted in the body
assert "tiled == reference: True" in OUT

fig = Figure(
    "03-halo-tile", "The tile loads a halo one pixel wide",
    f"An {BLK} by {BLK} tile in shared memory, s, for a {TILE} by {TILE} output tile and a 3 by 3 filter. The inner "
    f"{TILE} by {TILE} cells, orange, are the {TILE * TILE} threads that load a pixel and compute an output; the "
    f"one-pixel ring around them, red-edged, is the halo: {BLK * BLK - TILE * TILE} threads that load a pixel from "
    f"the neighboring tiles and then idle. Step 1, all {BLK * BLK} threads load; a barrier; step 2, the inner threads "
    f"compute. A table gives naive against tile reads: 3 by 3, {num(ROWS[0][1])} against {ROWS[0][2]}, "
    f"{ROWS[0][3]:.1f} times; 5 by 5, {num(ROWS[1][1])} against {ROWS[1][2]}, {ROWS[1][3]:.1f} times; 7 by 7, "
    f"{num(ROWS[2][1])} against {ROWS[2][2]}, {ROWS[2][3]:.1f} times.",
    subtitle="One block of convTiled, 3 × 3 filter; counts derived by snippets/convolution_model.py.",
    data_w=True)

C, GX, GY = 16, 104, 144
fig.text(GX, 128, f"s, [{BLK}, {BLK}]: one thread per cell", "head")
g = fig.grid(GX, GY, BLK, BLK, cell=C, lines_on_top=True,
             fill=lambda i, j: "compute-soft" if R <= i < BLK - R and R <= j < BLK - R else "hazard-soft")
g.window(R, R, TILE, TILE, "compute")
g.outline(0, 0, BLK, BLK, "hazard", width=1.5)
fig.brace(GY, GY + BLK * C, GX - 8, side="left", vertical=True)
fig.text(GX - 24, GY + BLK * C // 2 + 5, str(BLK), "label", anchor="end")
fig.brace(GX + R * C, GX + (BLK - R) * C, GY + BLK * C + 8, side="below")
fig.text(GX + BLK * C // 2, GY + BLK * C + 40, f"{TILE} outputs", "label", anchor="middle")

LX = 432
fig.legend(LX, 152, [dict(color="compute-soft", label=f"{TILE * TILE} threads load and compute"),
                     dict(color="hazard-soft", label=f"halo: {BLK * BLK - TILE * TILE} threads only load")])
fig.text(LX, 236, f"1. all {BLK * BLK} threads: one pixel into s, 0 outside the image", "label")
fig.fill(Box(LX, 256, 200, 8), "sync")
fig.text(LX + 216, 265, "__syncthreads()", "code")
fig.text(LX, 300, f"2. the inner {TILE * TILE} threads: one output each from s", "label")
fig.table(LX, 340, [["filter", "naive reads", "tile reads", "reuse"]] +
          [[f"{k} × {k}", num(n), num(t), f"{u:.1f}"] for k, n, t, u in ROWS],
          [96, 120, 104, 80], row_h=28, col_align=["start", "end", "end", "end"], highlight_row={1: "compute"})
fig.caption("Each pixel is read from global memory once per tile instead of once per window.")
fig.write()
