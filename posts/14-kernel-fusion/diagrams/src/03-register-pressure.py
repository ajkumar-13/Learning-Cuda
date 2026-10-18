"""Post 14, section 5: theoretical occupancy against registers per thread on sm_75, 256-thread blocks.

Run from the series root:  uv run python posts/14-kernel-fusion/diagrams/src/03-register-pressure.py
Writes posts/14-kernel-fusion/diagrams/03-register-pressure.svg.
Every value is derived by snippets/fusion_model.py (imported), which applies post 10's occupancy rule (256-register
units per warp, four quarters of 16,384 registers, 32 warp slots, 16 blocks); the curve is computed for every
register count from 1 to 255 and asserted equal to post 10's roofline_model.occupancy. The kernels of fusion.cu are
placed at the register counts nvcc -Xptxas=-v printed (10 for the fused chain, 39 for the tiled multiply), asserted
against the pasted compile output in index.md. cuda_fig_design.md section 5: one series in ink; the kernels are
compute orange marks; the counts past the cliff are hazard red marks.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
sys.path.insert(0, str(POST.parent / "10-roofline" / "snippets"))
from figkit import Figure  # noqa: E402
import fusion_model as m  # noqa: E402
import roofline_model as p10  # noqa: E402

occ = {r: 100 * m.occupancy(r)[2] for r in range(1, 256)}
assert all(m.occupancy(r)[0] == p10.occupancy(r, 0, 256)[3] for r in occ)
assert (occ[10], occ[39], occ[64], occ[65], occ[96], occ[128], occ[255]) == (100, 100, 100, 75, 50, 50, 25)
BODY = (POST / "index.md").read_text(encoding="utf-8")
for line in ("Used 10 registers, 380 bytes cmem[0]", "Used 39 registers, 2048 bytes smem, 380 bytes cmem[0]",
             "  100% up to 64 registers per thread (65,536 / 1,024 threads)"):
    assert line in BODY, line

xs, ys = [], []                              # a step curve: flat over each register count's unit
for r in range(1, 256):
    xs += [r - 0.5, r + 0.5]
    ys += [occ[r], occ[r]]

fig = Figure(
    "03-register-pressure", "Above 64 registers per thread, occupancy falls",
    "A step curve of theoretical occupancy against registers per thread, 1 to 255, for 256-thread blocks on "
    "compute capability 7.5: 100% up to 64 registers, 75% from 65 to 80, 50% from 81 to 128, 25% from 129. Orange marks: the "
    "fused elementwise chain at 10 registers and the tiled matrix multiplication at 39, both at 100%. Red marks: "
    "65 registers at 75% and 128 at 50%.",
    subtitle="256-thread blocks, compute capability 7.5; derived by fusion_model.py with post 10's rule.",
    data_w=True)
body = fig.panel(fig.row(1)[0])
ax = fig.line_chart(body, [dict(xs=xs, ys=ys, color="ink", points=False, label=None)],
                    x=(0, 256, [0, 32, 64, 96, 128, 160, 192, 224, 256]), y=(0, 125, [0, 25, 50, 75, 100]),
                    x_label="registers per thread", y_label="theoretical occupancy (%)", label_w=40,
                    ref_lines=[dict(x=64, label="64 = 65,536 / 1,024")])
for r, c, txt, dx, dy, anchor in ((10, "compute", "fused chain, 10", -4, -38, "start"),
                                  (39, "compute", "tiled matmul, 39", 8, -14, "end")):
    ax.point(r, occ[r], "diamond", c, size=10)
    ax.text(r, occ[r], txt, style="label", anchor=anchor, dx=dx, dy=dy)
for r, txt, dx, dy, anchor in ((65, "65: 75%", 10, -12, "start"), (128, "128: 50%", 10, -12, "start")):
    ax.point(r, occ[r], "circle", "hazard", size=10)
    ax.text(r, occ[r], txt, style="label", anchor=anchor, dx=dx, dy=dy)
fig.caption("A fused kernel pays for registers only past 64 per thread; then whole blocks leave the SM.")
fig.write()
