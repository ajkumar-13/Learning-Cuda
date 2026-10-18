"""Post 10, sections 3 and 4 (hero): the GTX 1650's roofline with the series' measured kernels placed under it.

Run from the series root:  uv run python posts/10-roofline/diagrams/src/01-roofline.py
Writes posts/10-roofline/diagrams/01-roofline.svg.
cuda_fig_design.md section 5: a log-log line_chart, intensity in FLOP per byte against GFLOP/s, the bandwidth roof
sloped and the compute roof flat, both derived by snippets/roofline_model.py (imported here). The dots are measured
values that the model carries as constants from posts 02, 03, 04 and 06; each is asserted here against the table that
post pasted, so a change there fails this script. Measured dots are compute orange; the derived roof is ink; the naive
kernel's lower bound on its real intensity is a hollow mark joined by a dashed arrow.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure, num  # noqa: E402
import roofline_model as m  # noqa: E402

POSTS = POST.parent
BODY = (POST / "index.md").read_text(encoding="utf-8")


def text(slug):
    return (POSTS / slug / "index.md").read_text(encoding="utf-8")


# the measured constants against the tables the earlier posts pasted
assert re.search(r"(?m)^\s+10000000\s+20\s.*\s145\.0\s+ok$", text("02-vector-addition"))
assert re.search(r"(?m)^\s+1024\s+10\.939\s+196\s+7\.032\s+305\s+0\.981\s+2189\s", text("03-matrix-multiplication"))
assert re.search(r"(?m)^warp shuffle\s+1024\s+0\.434\s+154\.6\s", text("04-reduction"))
assert re.search(r"(?m)^padded\s+1\.079\s+124\.4\s", text("06-matrix-transpose"))
M = {name: (ai, gf) for name, ai, gf, _ in m.MEASURED}
assert M["vector add"][1] == 145.0 / 12 and M["reduction, shuffle"][1] == 154.6 / 4
assert (M["matmul naive"][1], M["matmul tiled, TILE 16"][1], M["cuBLAS"][1]) == (196.0, 305.0, 2189.0)
assert m.TRANSPOSE_GBS == 124.4

PI = m.PEAK_FLOPS / 1e9
BETA = m.PEAK_BW / 1e9
RIDGE = m.RIDGE
NAIVE_I, NAIVE_GF = M["matmul naive"]
REAL_I = NAIVE_GF / BETA                     # the least intensity at which 160 GB/s could feed 196 GFLOP/s
for claim in ["3,118 GFLOP/s", "ridge point of 19.5 FLOP per byte", "at least $196 / 160 = 1.225$ FLOP per byte",
              "4.9 times", "transpose, padded: I = 0, 124.4 GB/s = 78% of beta", "moves 78% of $\\beta$"]:
    assert claim in BODY, claim
assert round(PI) == 3118 and round(RIDGE, 1) == 19.5 and round(REAL_I, 3) == 1.225
assert round(NAIVE_GF / (m.ceiling(NAIVE_I) / 1e9), 1) == 4.9

fig = Figure(
    "01-roofline", "Measured kernels under the GTX 1650's two roofs",
    "A log-log chart of attainable performance in GFLOP/s against arithmetic intensity in FLOP per byte, from 0.01 "
    f"to 1,000. The derived roof rises along the 160 GB/s bandwidth slope to the ridge point at {RIDGE:.1f} FLOP per "
    f"byte and stays flat at the compute peak of {num(round(PI))} GFLOP/s. Measured points: vector add "
    f"{M['vector add'][1]:.1f} GFLOP/s at 0.083, the shuffle reduction {M['reduction, shuffle'][1]:.1f} at 0.25, "
    f"the tiled matrix multiply 305 at 3.97, cuBLAS {num(2189)} at 171 (its intensity is at most that), and the naive "
    f"matrix multiply 196 at 0.25, above its slope roof of 40; a dashed arrow leads to a hollow mark at "
    f"{REAL_I:.3f} FLOP per byte, the least intensity that 196 GFLOP/s implies. A note says the padded transpose, "
    "with no arithmetic, moved 124.4 GB/s and cannot be drawn on a log axis.",
    subtitle="Roof derived; dots measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026.",
    data_w=True)

body = fig.panel(fig.row(1)[0])
ax = fig.line_chart(
    body,
    [dict(xs=[0.01, RIDGE, 1000], ys=[0.01 * BETA, PI, PI], color="ink", points=False, label=None)],
    x=(0.01, 1000, [0.01, 0.1, 1, 10, 100, 1000]), y=(1, 10000, [1, 10, 100, 1000, 10000]),
    x_label="arithmetic intensity, FLOP per byte", y_label="GFLOP/s", x_log=True, y_log=True, minor=True,
    label_w=16, ref_lines=[dict(x=RIDGE, label=f"ridge {RIDGE:.1f}")])

# the roofs, named on the line
ax.text(1.5, 14, "slope: 160 GB/s × intensity", style="note", anchor="start")
ax.text(1000, PI, f"peak {num(round(PI))} GFLOP/s", style="note", anchor="end", dx=-4, dy=-10)

# measured kernels
for name, label, dx, dy, anchor in [
        ("vector add", "vector add 12.1", 12, 26, "start"),
        ("reduction, shuffle", "reduction 38.6", 12, 26, "start"),
        ("matmul tiled, TILE 16", "tiled 305", 12, 5, "start"),
        ("cuBLAS", "cuBLAS 2,189, intensity ≤ 171", 0, 30, "middle")]:
    ai, gf = M[name]
    ax.point(ai, gf, "diamond", "compute", size=10)
    ax.text(ai, gf, label, style="value", anchor=anchor, color="compute-ink", dx=dx, dy=dy)
ax.point(NAIVE_I, NAIVE_GF, "diamond", "hazard", size=10)
ax.text(NAIVE_I, NAIVE_GF, "naive matmul 196", style="value", anchor="end", color="hazard-ink", dx=-12, dy=5)
fig.arrow(ax.to_px(NAIVE_I, NAIVE_GF), ax.to_px(REAL_I, NAIVE_GF), color="hazard", dash="proj")
ax.point(REAL_I, NAIVE_GF, "circle", "hazard", size=10, hollow=True)
ax.text(REAL_I, NAIVE_GF, f"real intensity ≥ {REAL_I:.3f}", style="note", anchor="start", dx=12, dy=28)
ax.text(0.05, 1.6, "padded transpose: no FLOP, 124.4 GB/s, not drawn", style="note", anchor="start")

fig.caption("Each dot sits under its roof, except the naive kernel: the caches cut its DRAM traffic below its count.")
fig.write()
