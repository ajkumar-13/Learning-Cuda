"""Post 04, section 4: a warp sums in its registers; each lane reads the register of the lane `off` above it.

Run from the series root:  uv run python posts/04-reduction/diagrams/src/03-warp-shuffle.py
Writes posts/04-reduction/diagrams/03-warp-shuffle.svg.
Eight lanes shown (a warp has 32; the caption says so), one row of registers per step, drawn as a tree in
registers (cuda_fig_design.md section 3): every lane adds at every step, as the hardware does, and the lanes whose
values feed lane 0 are orange, with an arrow for each register read that feeds them; the lanes whose values nothing
reads are left unfilled; lane 0's final value, the sum, is green. Every value comes from
snippets/reduction_model.py (shuffle_down, EIGHT), which hands a lane past the end its own value as
__shfl_down_sync does; the 32-lane check is the model's, asserted against the post.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
from reduction_model import EIGHT, WARP, shuffle_down  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
OFFS = [4, 2, 1]
LEVELS = shuffle_down(EIGHT, OFFS)
FULL = shuffle_down(list(range(WARP)), [16, 8, 4, 2, 1])
assert LEVELS[-1][0] == sum(EIGHT) == 25 and FULL[-1][0] == sum(range(WARP)) == 496
assert LEVELS[1][:4] == [7, 2, 13, 3] and LEVELS[2][:2] == [20, 5]
assert "8 lanes, off 4 2 1  : lane 0 = 25 (sum 25)" in BODY
assert "v += __shfl_down_sync(0xffffffff, v, off);" in BODY

fig = Figure(
    "03-warp-shuffle", "A warp sums in its registers, no shared memory",
    "Eight lanes, lane 0 to lane 7, each holding one register: 3 1 7 0 4 1 6 3. At offset 4 every lane adds the "
    "register of the lane four above; lanes 0 to 3 now hold 7 2 13 3, in orange, with arrows from lanes 4 to 7. "
    "At offset 2 lanes 0 and 1 hold 20 and 5; at offset 1 lane 0 holds 25, the sum, in green. The other lanes also "
    "add at every step, and their values, unfilled, are never read. A full warp of 32 lanes takes five steps, "
    "offsets 16, 8, 4, 2 and 1.",
    subtitle="At offset k, each lane adds the register of lane + k to its own. Eight lanes shown; a warp has 32.",
    data_w=True)

X, CW, CH = 200, 80, 40
YS = [120, 208, 296, 384]
rows = []
for k, (y, level) in enumerate(zip(YS, LEVELS)):
    used = len(EIGHT) >> k                       # lanes whose value lane 0 still needs
    last = k == len(OFFS)
    g = fig.strip(X, y, len(level), cell_w=CW, cell_h=CH, values=level, font=16,
                  fill=lambda j, k=k, used=used, last=last: (
                      "memory-soft" if k == 0 or (last and j == 0) else ("compute-soft" if j < used else None)))
    rows.append(g)
    fig.text(40, y + 26, "registers" if k == 0 else f"offset {OFFS[k - 1]}", "label")
rows[0].col_labels({j: f"lane {j}" for j in range(8)}, side="above")
rows[-1].window(0, 0, color="memory")

for k in range(1, len(rows)):
    off = OFFS[k - 1]
    for lane in range(len(EIGHT) >> k):
        src, dst = rows[k - 1].cell(0, lane + off), rows[k].cell(0, lane)
        fig.arrow((src.cx, src.bottom + 2), (dst.cx, dst.y - 2), color="compute", width=1)

fig.text(X + 8 * CW, 452, "v += __shfl_down_sync(0xffffffff, v, off);", "code", anchor="end")
fig.legend(40, 476, [dict(color="compute", label="value lane 0 still needs"),
                     dict(color="ink-muted", label="computed, never read", mark="outline"),
                     dict(color="memory", label="the sum", mark="outline")], direction="row")
fig.caption("A full warp of 32 lanes takes five steps, offsets 16, 8, 4, 2, 1. Values from reduction_model.py.")
fig.write()
