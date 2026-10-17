"""Post 04, section 3: interleaved addressing splits warps at every step; sequential addressing only at the end.

Run from the series root:  uv run python posts/04-reduction/diagrams/src/02-addressing.py
Writes posts/04-reduction/diagrams/02-addressing.svg.
Lanes against steps for one block of 256 threads (cuda_fig_design.md sections 1 and 3), compressed to one cell per
warp: each cell prints how many of the warp's 32 lanes add at that step; a warp with every lane active is orange, a
split (divergent) warp is red-edged, an idle warp is empty. The split count per step is at the right of each grid,
and the totals 47 and 5 below. Every count comes from snippets/reduction_model.py (active_interleaved,
active_sequential, warp_states, divergent_warp_steps), asserted against the post.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
from reduction_model import (BLOCK, WARP, active_interleaved, active_sequential,  # noqa: E402
                             divergent_warp_steps, warp_states)

BODY = (POST / "index.md").read_text(encoding="utf-8")
WARPS = BLOCK // WARP
SCHEMES = [("Interleaved", "tid % (2 * stride) == 0", active_interleaved()),
           ("Sequential", "tid < stride", active_sequential())]
TOTALS = [divergent_warp_steps(steps) for _, _, steps in SCHEMES]
assert (BLOCK, WARP, WARPS, TOTALS) == (256, 32, 8, [47, 5])
assert "total divergent warp-steps: interleaved 47, sequential 5" in BODY


def lanes_active(active, w):
    return sum(1 for lane in range(WARP) if w * WARP + lane in active)


fig = Figure(
    "02-addressing", "Sequential addressing keeps whole warps active",
    f"Two grids for one block of {BLOCK} threads, 8 steps down by 8 warps across; each cell gives how many of the "
    "warp's 32 lanes add at that step. Interleaved addressing, tid mod 2 stride equals 0: 16, 8, 4, 2 and 1 lanes "
    "in every warp at steps 1 to 5, red-edged as split warps, then 1 lane in 4, 2 and 1 warps; 47 divergent "
    "warp-steps. Sequential addressing, tid less than stride: 4, 2 and 1 whole warps of 32 lanes in orange at steps "
    "1 to 3, then 16, 8, 4, 2 and 1 lanes in warp 0 only; 5 divergent warp-steps.",
    subtitle=f"One block of {BLOCK} threads, {WARPS} warps of {WARP}; counts derived by snippets/reduction_model.py.",
    data_w=True)

CW, CH, Y = 40, 32, 160
XS = [112, 528]
for (name, cond, steps), x, total in zip(SCHEMES, XS, TOTALS):
    fig.text(x, 116, name, "head")
    fig.text(x + 8 * CW, 116, cond, "code", anchor="end")
    states = [warp_states(a) for a in steps]
    vals = [[lanes_active(a, w) for w in range(WARPS)] for a in steps]
    g = fig.grid(x, Y, len(steps), WARPS, cell_w=CW, cell_h=CH, font=14,
                 values=lambda i, j: str(vals[i][j]) if vals[i][j] else "",
                 fill=lambda i, j: "compute-soft" if states[i][j] == "full" else (
                     "hazard-soft" if states[i][j] == "split" else None))
    for i in range(len(steps)):
        for j in range(WARPS):
            if states[i][j] == "split":
                g.outline(i, j, 1, 1, "hazard", width=1.5, form="grid")
    g.col_labels({j: str(j) for j in range(WARPS)}, side="above")
    fig.text(x + 8 * CW + 16, Y - 8, "split", "tick")
    for i, st in enumerate(states):
        n = st.count("split")
        fig.text(x + 8 * CW + 16, Y + i * CH + 21, str(n), "label", color="hazard" if n else None)
    fig.text(x, Y + 8 * CH + 28, f"{total} divergent warp-steps", "label", color="hazard", bold=True)
for i in range(8):
    fig.text(40, Y + i * CH + 21, f"step {i + 1}", "label")
fig.text(40, Y - 8, "warp", "tick")

fig.legend(40, 476, [dict(color="compute", label="all 32 lanes add"),
                     dict(color="hazard", label="split warp: some lanes idle", mark="outline"),
                     dict(color="ink-muted", label="empty: warp idle", mark="outline")], direction="row")
fig.caption("Both do the same 255 additions; only where the active threads sit differs.")
fig.write()
