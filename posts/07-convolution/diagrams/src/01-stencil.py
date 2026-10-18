"""Post 07, hero: each output pixel is a weighted sum of a 3 x 3 window, and neighboring windows overlap.

Run from the series root:  uv run python posts/07-convolution/diagrams/src/01-stencil.py
Writes posts/07-convolution/diagrams/01-stencil.svg.
cuda_fig_design.md section 3: the input is a grid of memory-green cells; the window thread (2, 2) reads is orange
(compute), with the emphasis outline; the window of its right neighbor, thread (2, 3), is a dashed outline; the
filter and the output pixels are ink. Every value is computed here from the 5 x 5 input below with the 3 x 3 box
filter (all ones, times 1/9), by the post's stencil sum with zero padding; the overlap (6 of 9 pixels) is counted.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
IMG = [[1, 3, 2, 4, 0],
       [2, 5, 4, 5, 3],
       [0, 6, 5, 3, 2],
       [3, 4, 7, 6, 1],
       [2, 1, 0, 5, 4]]
K, RAD = 3, 1


def window_sum(row, col):
    """The stencil sum of the post with every weight 1: the box filter before its 1/9."""
    return sum(IMG[row + i - RAD][col + j - RAD] for i in range(K) for j in range(K)
               if 0 <= row + i - RAD < 5 and 0 <= col + j - RAD < 5)


A, B = (2, 2), (2, 3)
SA, SB = window_sum(*A), window_sum(*B)
CELLS_A = {(A[0] + i - RAD, A[1] + j - RAD) for i in range(K) for j in range(K)}
CELLS_B = {(B[0] + i - RAD, B[1] + j - RAD) for i in range(K) for j in range(K)}
SHARED = len(CELLS_A & CELLS_B)
assert (SA, SB, SHARED) == (45, 36, 6) and SA % 9 == 0 and SB % 9 == 0
OA, OB = SA // 9, SB // 9
assert r"\text{out}[y][x] = \sum_{i=-r}^{r}\sum_{j=-r}^{r} \text{in}[y+i][x+j] \cdot \text{mask}[i+r][j+r]" in BODY

fig = Figure(
    "01-stencil", "Each output pixel sums a window; neighbors overlap",
    "A 5 by 5 input with a 3 by 3 window, shaded orange, around pixel (2, 2), and a dashed window around its right "
    f"neighbor (2, 3); the two windows share {SHARED} of their 9 pixels. The window is multiplied by a 3 by 3 filter of "
    f"ones times 1/9, the box blur, and summed: out[2][2] = {SA} / 9 = {OA}, and the neighbor's window gives "
    f"out[2][3] = {SB} / 9 = {OB}.",
    subtitle="A 3 × 3 box blur, radius 1; every value computed from the 5 × 5 input shown.",
    data_w=True)

CELL, Y = 48, 168
GX = 80
fig.text(GX, 136, "in: [5, 5]", "head")
g = fig.grid(GX, Y, 5, 5, cell=CELL, values=IMG, fill=lambda i, j: "memory-soft")
g.window(A[0] - RAD, A[1] - RAD, K, K, "compute", fill="compute")
fig.outline(g.span(B[0] - RAD, B[1] - RAD, K, K).inset(4), "compute", width=1.5, dash="lead")
g.col_labels([str(j) for j in range(5)], side="above")
g.row_labels([str(i) for i in range(5)], side="left")
fig.text(GX, Y + 5 * CELL + 32, f"thread (2, 3)'s window, dashed, shares {SHARED} of 9 pixels", "note")

ox = GX + 5 * CELL + 40
fig.op(ox, Y + 2 * CELL + CELL // 2, "×")
FX = ox + 40
fig.text(FX, 136, "mask: [3, 3]", "head")
f = fig.grid(FX, Y + CELL, 3, 3, cell=CELL, values=[[1] * 3 for _ in range(3)])
fig.text(FX, Y + 4 * CELL + 32, "each weight × 1/9", "note")

ex = FX + 3 * CELL + 40
fig.op(ex, Y + 2 * CELL + CELL // 2, "=")
OX = ex + 40
fig.text(OX, 136, "out: two pixels", "head")
o = fig.grid(OX, Y + 2 * CELL, 1, 2, cell=CELL, values=[[OA, OB]])
o.window(0, 0, color="compute")
fig.outline(o.cell(0, 1).inset(4), "compute", width=1.5, dash="lead")
o.col_labels(["[2][2]", "[2][3]"], side="above")
NX = OX
fig.text(NX, Y + 3 * CELL + 32, f"out[2][2] = {SA} / 9 = {OA}", "label")
fig.text(NX, Y + 3 * CELL + 56, f"out[2][3] = {SB} / 9 = {OB}", "label")

fig.caption("A naive kernel reads every shared pixel once per window that covers it.")
fig.write()
