"""Post 04, the hero: eight values fold to one in three parallel steps of sequential addressing.

Run from the series root:  uv run python posts/04-reduction/diagrams/src/01-tree-reduction.py
Writes posts/04-reduction/diagrams/01-tree-reduction.svg.
A tree drawn top to bottom (cuda_fig_design.md section 3): each row is the shared array s after one step, cell t
at the position of index t; the inputs are memory (green), the partial sums written by active threads compute
(orange), the final sum green and emphasized; the number of threads that add is written at the right of each
level. Every value comes from snippets/reduction_model.py (tree_sequential, EIGHT), asserted against the post.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
from reduction_model import EIGHT, N, tree_sequential  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
LEVELS = tree_sequential(EIGHT)
STEPS = len(LEVELS) - 1
DEPTH_N = N.bit_length() - 1
assert LEVELS == [[3, 1, 7, 0, 4, 1, 6, 3], [7, 2, 13, 3], [20, 5], [25]] and STEPS == 3 and DEPTH_N == 24
assert "stride 4, 4 adds  : 7 2 13 3" in BODY and "stride 1, 1 add   : 25" in BODY
assert "eight values need three parallel steps, and 16,777,216 need 24" in BODY

vals = " ".join(str(v) for v in LEVELS[0])
fig = Figure(
    "01-tree-reduction", "A tree sums eight values in three parallel steps",
    f"The shared array s drawn four times, top to bottom. It starts with eight values, {vals}, in green. At stride 4, "
    "threads 0 to 3 each add the element four places up into their own, giving 7 2 13 3 in orange; at stride 2 "
    "threads 0 and 1 give 20 and 5; at stride 1 thread 0 gives 25, the sum, emphasized in green. Arrows show the "
    "two inputs of every addition. At the right, 4, 2 and 1 threads add at the three steps; 16,777,216 values "
    "would need 24 steps.",
    subtitle="Sequential addressing: at stride s, thread t < s adds s[t + s] into s[t]. Derived by reduction_model.py.",
    data_w=True)

X, CW, CH = 200, 80, 40
YS = [120, 216, 312, 408]
rows = []
for k, (y, level) in enumerate(zip(YS, LEVELS)):
    last = k == STEPS
    color = "memory" if k == 0 else ("memory" if last else "compute")
    g = fig.strip(X, y, len(level), cell_w=CW, cell_h=CH, values=level, font=16,
                  fill=lambda j, c=color: c + "-soft")
    rows.append(g)
    if k == 0:
        fig.text(40, y + 26, "start: 8 values", "label")
        g.col_labels({j: str(j) for j in range(8)}, side="above")
    else:
        stride = len(LEVELS[0]) >> k
        fig.text(40, y + 26, f"stride {stride}", "label")
        n = len(level)
        if not last:
            fig.text(X + n * CW + 24, y + 26, f"{n} threads add", "label", color="compute")
fig.text(X - 16, YS[0] - 8, "index t", "tick", anchor="end")
rows[-1].window(0, 0, color="memory")
fig.callout(rows[-1].cell(0, 0), "1 thread adds: the sum", side="right", length=24)

# arrows: the two inputs of every addition, from the row above to cell t of the row below
for k in range(1, STEPS + 1):
    above, below = rows[k - 1], rows[k]
    stride = len(LEVELS[k])
    for t in range(stride):
        dst = below.cell(0, t)
        for src_i in (t, t + stride):
            src = above.cell(0, src_i)
            fig.arrow((src.cx, src.bottom + 2), (dst.cx, dst.y - 2), color="compute", width=1)

fig.legend(40, 468, [dict(color="memory", label="value in shared memory"),
                     dict(color="compute", label="partial sum written by an active thread")], direction="row")
fig.caption(f"Depth is log₂ of the count: 3 steps for 8 values, {DEPTH_N} for 16,777,216, the same N − 1 adds either way.")
fig.write()
