"""Post 01, section 4: the launch of N = 1000 in blocks of 256, from the grid down to one thread's index.

Run from the series root:  uv run python posts/01-introduction-to-cuda/diagrams/src/03-thread-hierarchy.py
Writes posts/01-introduction-to-cuda/diagrams/03-thread-hierarchy.svg.
This is the series' reference drawing of a launch (cuda_fig_design.md section 1): the grid as a row of block cards,
the hardware (the SMs) below with the mapping drawn as arrows, one block opened into its warps with the warp
boundary every 32 lanes, and the global index worked for one thread. Every count comes from snippets/thread_model.py
(N, BLOCK, WARP, global_index); the SM count is read from the device_query output pasted in the post (measured).
Which four SMs receive the four blocks is the hardware's choice; the figure draws SMs 0 to 3 and says so.
"""
import contextlib
import io
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its tables on import
    from thread_model import BLOCK, N, WARP, global_index  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
SMS = int(re.search(r"streaming multiproc\. : (\d+) SMs", BODY).group(1))
GRID = -(-N // BLOCK)
SURPLUS = GRID * BLOCK - N
B, T = 3, 5                                   # the worked thread
I = global_index(B, T)
assert (N, BLOCK, WARP, GRID, SURPLUS, I, SMS) == (1000, 256, 32, 4, 24, 773, 14)
assert "Thread 5 of block 3 owns element $3 \\times 256 + 5 = 773$" in BODY
assert "The last 24 threads of block 3" in BODY and "at most four of the GTX 1650's 14 SMs" in BODY
WARPS = BLOCK // WARP
first_surplus = N - B * BLOCK                  # threadIdx.x of the first surplus thread in block 3: 232
assert first_surplus == 232

fig = Figure(
    "03-thread-hierarchy", "Grid, block, warp, lane: where thread 773 lives",
    f"Top, the grid of one launch over N = {N}: {GRID} block cards, block 0 to block 3, each of {BLOCK} threads "
    f"covering i = 0 to 255, 256 to 511, 512 to 767 and 768 to 1023; block 3 is emphasized. Middle, the hardware: "
    f"{SMS} streaming multiprocessors, of which SM 0 to SM 3 each receive one block, drawn as arrows, and the "
    f"other ten are idle. Bottom, block 3 opened on its SM into {WARPS} rows of {WARP} lanes, one row per warp. "
    f"Lane 5 of warp 0 is emphasized, and a card works its index: blockIdx.x 3 times blockDim.x 256 plus "
    f"threadIdx.x 5 equals {I}. The last {SURPLUS} lanes of warp 7, i = 1000 to 1023, are empty and outlined "
    "in red: the surplus threads that fail the guard i < N.",
    subtitle=f"Derived by snippets/thread_model.py: N = {N}, {BLOCK} threads per block; {SMS} SMs measured.",
    height=720, data_w=True)

# 1. the grid: GRID block cards
fig.text(40, 128, "grid", "head")
fig.text(96, 128, f"gridDim.x = {GRID}, blockDim.x = {BLOCK}", "code")
cards = []
for b in range(GRID):
    x = 40 + b * 224
    lo, hi = b * BLOCK, (b + 1) * BLOCK - 1
    fig.card(x, 144, 208, 88, heading=f"block {b}", lines=[f"i = {lo} to {hi}"], style="note", color="compute")
    cards.append(Box(x, 144, 208, 88))
fig.outline(cards[B], "compute", width=2.5, radius=8)

# 2. the hardware: SMS cards; the four blocks land on four of them
SM_X, SM_W, SM_Y = 200, 44, 296
fig.text(40, SM_Y + 34, f"{SMS} SMs", "head")
sm = []
for k in range(SMS):
    box = Box(SM_X + k * (SM_W + 8), SM_Y, SM_W, 56)
    fig.fill(box, "compute-soft" if k < GRID else "neutral-soft", radius=8, fit=False)
    fig.outline(box, "compute-line" if k < GRID else "border", radius=8)
    fig.text(box.cx, SM_Y + 24, f"SM {k}", "tick", anchor="middle")
    if k >= GRID:
        fig.text(box.cx, SM_Y + 44, "idle", "tick", anchor="middle")
    sm.append(box)
for b in range(GRID):
    fig.arrow((cards[b].cx, cards[b].bottom), (sm[b].cx, sm[b].y))

# 3. block 3 opened on its SM: WARPS warps of WARP lanes
GX, GY, C = 168, 432, 16
g = fig.grid(GX, GY, WARPS, WARP, cell=C, lines_on_top=True,
             fill=lambda w, l: "compute" if global_index(B, w * WARP + l) < N else None)
fig.zoom(sm[B], g.box)
g.row_labels([f"warp {w}" for w in range(WARPS)], style="tick")
g.col_labels({0: "0", 8: "8", 16: "16", 24: "24", 31: "31"}, side="below")
fig.text(GX - 8, GY + WARPS * C + 20, "lane", "tick", anchor="end")
g.window(T // WARP, T % WARP, 1, 1, "compute", fill="compute")
w7, l0 = divmod(first_surplus, WARP)
g.outline(w7, l0, 1, WARP - l0, "hazard", width=1.5)
fig.text(GX + l0 * C, GY + WARPS * C + 48, f"{SURPLUS} surplus lanes: i = {N} to {GRID * BLOCK - 1} fail i < N",
         "label", color="hazard")

fig.card(712, GY, 208, None, heading=f"thread {T}, block {B}",
         lines=[f"blockIdx.x  = {B}", f"blockDim.x  = {BLOCK}", f"threadIdx.x = {T}", f"i = {B} * {BLOCK} + {T}",
                f"  = {I}"], style="code", color="compute", fit=True)
fig.caption("Four blocks occupy at most four SMs; which four is the hardware's choice.")
fig.write()
