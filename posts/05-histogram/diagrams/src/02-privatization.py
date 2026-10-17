"""Post 05, section 3: the privatized kernel in three phases, zero, count, merge, with its two barriers.

Run from the series root:  uv run python posts/05-histogram/diagrams/src/02-privatization.py
Writes posts/05-histogram/diagrams/02-privatization.svg.
cuda_fig_design.md sections 2 and 3: each block's private histogram s and the global bins are memory green, the
barriers purple bars across the blocks they hold (one bar per block, since __syncthreads holds one block), the merge
arrows into global memory purple (they are atomics). Two of the 256 blocks are drawn, block 0 and block 255, with
three of the 256 bins (0, 128, 255), on the solid-color input. Every count comes from snippets/histogram_model.py
(65,536 per block's s[128], 256 merges, 16,777,216 in global bins[128]) and is asserted against the body.
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
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its tables on import
    from histogram_model import N, GRID, BLOCK, BINS, rows  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
SOLID = rows["solid"]
PER_BLOCK = SOLID["shared_hot"]
assert (N, GRID, BLOCK, BINS, PER_BLOCK, SOLID["merge_hot"]) == (16777216, 256, 256, 256, 65536, 256)
assert PER_BLOCK * GRID == N and SOLID["equal"]
assert "for (int b = threadIdx.x; b < BINS; b += blockDim.x) s[b] = 0;   // 1. zero" in BODY
assert "atomicAdd(&bins[b], s[b]);            // 3. merge" in BODY

fig = Figure(
    "02-privatization", "Count in shared memory, then merge once per bin",
    f"The privatized kernel on {num(N)} solid-color values, drawn for two of its {GRID} blocks, block 0 and block "
    f"{GRID - 1}, and three of the {BINS} bins, 0, 128 and 255. Phase 1: each block's threads zero its private "
    "shared array s. A purple barrier per block. Phase 2: each block counts its "
    f"{num(PER_BLOCK)} values into s, so s[128] holds {num(PER_BLOCK)}. A second barrier. Phase 3: each block adds "
    f"its bins to the global bins with one atomic per bin; global bins[128] receives {GRID} merges and ends at "
    f"{num(N)}.",
    subtitle=f"Two of {GRID} blocks and three of {BINS} bins, solid-color input; counts derived by "
             "snippets/histogram_model.py.",
    height=720, data_w=True)

CW, CH = 96, 40
COLS = [(168, "block 0"), (584, f"block {GRID - 1}")]
SHOWN = ["0", "128", "255"]


def strip(x, y, values, labels=False):
    g = fig.grid(x, y, 1, 3, cell_w=CW, cell_h=CH, values=[values], font=16, fill=lambda i, j: "memory-soft")
    if labels:
        g.col_labels([f"s[{b}]" for b in SHOWN], side="above")
    return g


# column headings
for x, name in COLS:
    fig.text(x + 3 * CW // 2, 144, f"{name}, {BLOCK} threads", "head", anchor="middle")
MID = (COLS[0][0] + 3 * CW + COLS[1][0]) // 2           # the gap between the two blocks

# phase 1: zero
Y1 = 192
for x, _ in COLS:
    strip(x, Y1, [0, 0, 0], labels=True)
fig.text(40, Y1 + 25, "1. zero s", "head")

# barrier 1
B1 = Y1 + CH + 24
for x, _ in COLS:
    fig.fill(Box(x, B1, 3 * CW, 8), "sync")
fig.text(40, B1 + 9, "__syncthreads()", "code")

# phase 2: count
Y2 = B1 + 32
p2 = [strip(x, Y2, ["0", num(PER_BLOCK), "0"]) for x, _ in COLS]
fig.text(40, Y2 + 25, "2. count", "head")
for y in (Y1, Y2):
    fig.text(MID, y + 25, "...", "head", anchor="middle")
fig.text(COLS[0][0], Y2 + CH + 24, f"each block: {num(PER_BLOCK)} shared atomics on s[128]", "note")

# barrier 2
B2 = Y2 + CH + 48
for x, _ in COLS:
    fig.fill(Box(x, B2, 3 * CW, 8), "sync")
fig.text(40, B2 + 9, "__syncthreads()", "code")

# phase 3: merge into the global bins
YG = B2 + 104
GW = CW + 16
GX = MID - 3 * GW // 2
gl = fig.grid(GX, YG, 1, 3, cell_w=GW, cell_h=CH, values=[["0", num(N), "0"]], font=16,
              fill=lambda i, j: "memory-soft")
gl.col_labels([f"bins[{b}]" for b in SHOWN], side="below")
gl.window(0, 1, 1, 1, "memory", form="grid")
fig.text(40, B2 + 60, "3. merge", "head")
fig.text(40, YG + 25, "global bins", "head")
dst = gl.cell(0, 1)
for k, g in enumerate(p2):
    c = g.cell(0, 1)
    fig.arrow((c.cx, B2 + 16), (dst.cx + (-24 if k == 0 else 24), dst.y), color="sync")
fig.text(MID, YG + CH + 64, f"one atomic per bin per block, so bins[128] receives {GRID} merges:", "label",
         anchor="middle")
fig.text(MID, YG + CH + 84, f"{GRID} × {num(PER_BLOCK)} = {num(N)}", "label", anchor="middle")

fig.legend(40, 640, [dict(color="memory-soft", label="histogram bins, shared or global"),
                     dict(color="sync", label="barrier, one block"),
                     dict(color="sync", label="atomic merge", mark="line")], direction="row")
fig.caption("The long queue stays inside each block; global memory sees one merge per bin per block.")
fig.write()
