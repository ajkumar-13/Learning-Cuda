"""Post 03, sections 3 and 4: the naive kernel re-reads global memory; the tiled kernel loads a tile once into
shared memory, behind two barriers, and reuses it.

Run from the series root:  uv run python posts/03-matrix-multiplication/diagrams/src/03-naive-vs-tiled.py
Writes posts/03-matrix-multiplication/diagrams/03-naive-vs-tiled.svg.
cuda_fig_design.md sections 2 and 3: global memory and shared memory green, threads orange, the redundant reads red
(hazard), the two __syncthreads() barriers purple bars. The load counts (524,288 against 32,768 for a 64 x 64 x 64
product, 128 against 8 per output) are printed by snippets/matmul_model.py and imported here; the tile is drawn
4 x 4 to stand for TILE x TILE, which the subtitle says.
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

NL, TL, T = mm.naive_loads, mm.tiled_loads, mm.TILE
PER_N, PER_T = NL // (mm.M * mm.N), TL // (mm.M * mm.N)
assert (NL, TL, T, PER_N, PER_T, mm.K) == (524288, 32768, 16, 128, 8, 64)

fig = Figure(
    "03-naive-vs-tiled", "Naive re-reads global memory; tiled loads once per block",
    "Left, the naive kernel: four threads computing elements 0 to 3 of row 2 of C each read the whole of row 2 of A "
    f"from global memory, drawn as four red arrows to the same green row: {PER_N} loads per output at K = {mm.K}, "
    f"{num(NL)} in all. Right, the tiled kernel: a tile of A and a tile of B in global memory are loaded once, one "
    "element per thread, into the shared-memory tiles s_A and s_B; a purple __syncthreads() bar marks that the tiles "
    "are loaded; the threads then do TILE multiply-adds from shared memory; a second purple bar marks that every "
    f"thread is done before the next load; an arrow loops back to the next tile. {PER_T} loads per output, "
    f"{num(TL)} in all, {NL // TL} times fewer.",
    subtitle=f"Load counts derived by snippets/matmul_model.py for 64 × 64 × 64, TILE = {T}; tiles drawn 4 × 4.",
    height=720, data_w=True)

# left: naive
fig.text(40, 128, "naive", "head")
fig.text(56, 168, "row 2 of A, global memory", "label")
row = fig.strip(56, 184, 8, cell=40, fill=lambda i: "memory-soft")
fig.outline(row.box, "memory", width=2.5)
for i in range(4):
    b = Box(56 + i * 96, 464, 72, 48)
    fig.fill(b, "compute-soft", radius=8, fit=False)
    fig.outline(b, "compute-line", radius=8)
    fig.text(b.cx, b.y + 30, f"C[2][{i}]", "code", anchor="middle")
    fig.arrow((b.cx, b.y), (row.box.x + 40 + i * 80, row.box.bottom), color="hazard")
fig.text(56, 544, "4 threads read the same row 4 times", "label", color="hazard")
fig.text(56, 608, f"{PER_N} loads per output (2K)", "label")
fig.text(56, 632, f"{num(NL)} loads in all", "label")

# right: tiled
X0 = 520
fig.text(X0, 128, "tiled", "head")
fig.text(X0, 168, "tiles of A and B, global memory", "label")
gA = fig.grid(X0, 184, 4, 4, cell=24, fill=lambda i, j: "memory-soft")
gB = fig.grid(X0 + 160, 184, 4, 4, cell=24, fill=lambda i, j: "memory-soft")
fig.arrow((X0 + 128, 288), (X0 + 128, 336), label="loaded once, 1 per thread", side="right")
fig.text(X0, 360, "s_A and s_B, shared memory", "label")
sA = fig.grid(X0, 376, 4, 4, cell=24, fill=lambda i, j: "memory")
sB = fig.grid(X0 + 160, 376, 4, 4, cell=24, fill=lambda i, j: "memory")
fig.fill(Box(X0, 488, 352, 8), "sync", fit=False)
fig.text(X0, 520, "__syncthreads(): tiles loaded", "code", color="sync")
tb = Box(X0, 528, 352, 40)
fig.fill(tb, "compute-soft", radius=8, fit=False)
fig.outline(tb, "compute-line", radius=8)
fig.text(X0 + 16, 553, "each thread: TILE multiply-adds", "label")
fig.fill(Box(X0, 576, 352, 8), "sync", fit=False)
fig.text(X0, 608, "__syncthreads(): done before next load", "code", color="sync")
fig.arrow((X0 + 352, 580), (X0 + 264, 248), via=((X0 + 368, 580), (X0 + 368, 248)))
fig.text(X0 + 288, 272, "next tile", "tick")
for x, a in ((X0 + 104, "A"), (X0 + 264, "B")):
    fig.text(x, 204, a, "code")
for x, a in ((X0 + 104, "s_A"), (X0 + 264, "s_B")):
    fig.text(x, 396, a, "code")
fig.text(X0, 640, f"{PER_T} loads per output, {num(TL)} in all: {NL // TL} × fewer", "label", color="memory")
fig.caption("Each loaded value is reused TILE times from shared memory before the block moves on.")
fig.write()
