"""Post 01, section 5: a warp that branches on i % 2 runs both paths, one pass each, with half its lanes idle.

Run from the series root:  uv run python posts/01-introduction-to-cuda/diagrams/src/04-warp-divergence.py
Writes posts/01-introduction-to-cuda/diagrams/04-warp-divergence.svg.
A full warp of 32 lanes (cuda_fig_design.md section 1: active lanes orange, idle lanes red-edged and empty, a
divergent branch drawn as one row per pass, the passes counted). Which lanes run in which pass, and the pass counts
for one 256-thread block, come from snippets/thread_model.py (passes, WARP, BLOCK), asserted against the post.
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
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its tables on import
    from thread_model import BLOCK, WARP, passes  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
WARPS = range(0, BLOCK, WARP)
BRANCHES = [("no branch", lambda i: 0), ("i % 2", lambda i: i % 2), ("(i / 32) % 2", lambda i: (i // WARP) % 2)]
COUNTS = [passes(f, WARPS) for _, f in BRANCHES]
assert (WARP, BLOCK, COUNTS) == (32, 256, [8, 16, 8])
assert passes(lambda i: 0, [0]) == 1 and passes(lambda i: i % 2, [0]) == 2
assert "no branch                  = 8" in BODY and "branch on i % 2            = 16" in BODY
assert "if (i % 2 == 0) doA();" in BODY

fig = Figure(
    "04-warp-divergence", "A warp that splits runs both paths, one after the other",
    "One warp of 32 lanes, lane 0 to lane 31. With no branch, all 32 lanes run in one pass, in orange. Branching on "
    "i mod 2 takes two passes: in pass 1 the 16 even lanes run doA() and the 16 odd lanes are idle, empty and "
    "outlined in red; in pass 2 the odd lanes run doB() and the even lanes are idle. A table gives the passes for "
    f"one block of {BLOCK} threads, {BLOCK // WARP} warps: {COUNTS[0]} with no branch, {COUNTS[1]} branching on i mod 2, "
    f"and {COUNTS[2]} branching on (i / 32) mod 2, where whole warps agree.",
    subtitle=f"One warp of {WARP} lanes; pass counts derived by snippets/thread_model.py.", data_w=True)

X, C = 168, 20


def lanes(y, active):
    g = fig.grid(X, y, 1, WARP, cell=C, fill=lambda i, j: "compute" if active(j) else None)
    for j in range(WARP):
        if not active(j):
            g.outline(0, j, 1, 1, "hazard", width=1.5, form="grid")
    return g


fig.text(40, 128, "No branch", "head")
lanes(144, lambda j: True)
fig.text(40, 160, "1 pass", "label")

fig.text(40, 216, "A branch splits the warp: 2 passes", "head")
fig.text(X + WARP * C, 216, "if (i % 2 == 0) doA(); else doB();", "code", anchor="end")
lanes(232, lambda j: j % 2 == 0)
fig.text(40, 248, "pass 1", "label")
fig.text(X + WARP * C + 16, 248, "doA()", "code")
g = lanes(256, lambda j: j % 2 == 1)
fig.text(40, 272, "pass 2", "label")
fig.text(X + WARP * C + 16, 272, "doB()", "code")
g.col_labels({0: "0", 8: "8", 16: "16", 24: "24", 31: "31"}, side="below")
fig.text(X - 8, 296, "lane", "tick", anchor="end")

fig.text(40, 336, f"One block of {BLOCK} threads, {BLOCK // WARP} warps", "head")
fig.table(40, 352, [["branch on", "passes"]] + [[name, str(n)] for (name, _), n in zip(BRANCHES, COUNTS)],
          [200, 96], row_h=28, col_align=["start", "end"], highlight_row={2: "hazard"})
fig.legend(424, 400, [dict(color="compute", label="lane runs this pass"),
                      dict(color="hazard", label="lane idle in this pass", mark="outline")])
fig.caption("Branch on what whole warps agree on: (i / 32) % 2 splits no warp.")
fig.write()
