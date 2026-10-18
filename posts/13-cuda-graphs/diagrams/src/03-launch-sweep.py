"""Post 13, section 4: the speedup of graph replay over direct launches against kernel length, measured and modeled.

Run from the series root:  uv run python posts/13-cuda-graphs/diagrams/src/03-launch-sweep.py
Writes posts/13-cuda-graphs/diagrams/03-launch-sweep.svg.
cuda_fig_design.md section 5. The measured points are the speedup column of the launch sweep pasted in index.md
section 2 (snippets/graphs.cu on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026), drawn
solid. The curve is derived: it imports snippets/graphs_model.py, whose inputs are that run, and draws its
speedup (L + W) / (l + T_build / N + W) for N = 10,000 kernels, dashed; its 1.1x crossing is the model's printed
6.4 us. The model's inputs are asserted equal to the pasted run.
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
from figkit import Figure  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import graphs_model as m  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROWS = re.findall(r"^ +(\d+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)x$", BODY, re.MULTILINE)[:6]
# [:6]: the harness's sweep comes first; the model's check table further down matches the same pattern.
ws = [int(r[0]) for r in ROWS]
speed = [float(r[5]) for r in ROWS]
assert ws == [0, 2, 4, 8, 16, 32] and speed == [2.34, 1.25, 1.15, 1.06, 1.04, 1.02], (ws, speed)
for r in ROWS:   # the model's inputs are this run
    assert m.SWEEP[int(r[0])] == tuple(float(v) for v in r[1:5])
assert "build (capture 20 empty kernels + instantiate): %.1f us" % m.BUILD in BODY
N = 10000
w11 = (m.L - 1.1 * (m.ELL + m.BUILD / N)) / 0.1
assert round(w11, 1) == 6.4 and "W where the speedup falls to 1.1x   : 6.4 us" in BODY
assert round(m.L, 2) == 1.54 and round(m.ELL, 2) == 0.81

curve_w = [x / 4 for x in range(0, 129)]
curve = [m.speedup(N, w) for w in curve_w]

fig = Figure(
    "03-launch-sweep", "A graph pays off only for kernels a few µs long",
    "Speedup of graph replay over direct launches against kernel length W from 0 to 32 µs. Measured points: 2.34 at "
    "0, 1.25 at 2, 1.15 at 4, 1.06 at 8, 1.04 at 16 and 1.02 at 32 µs. The dashed model curve, from L of 1.54 µs "
    "and l of 0.81 µs per kernel over 10,000 kernels, starts at 1.89, passes through the other points and crosses "
    "the 1.1 line at 6.4 µs.",
    subtitle="Points measured on the author's GTX 1650 (7.5), CUDA 12.3, October 2026; curve derived by graphs_model.py.",
    data_w=True)

body = fig.panel(fig.row(1, h=328)[0], "Speedup, direct time over graph time")
ax = fig.line_chart(body, [
    dict(xs=curve_w, ys=curve, color="ink-muted", dash="proj", points=False, label=None),
], x=(0, 32, [0, 8, 16, 24, 32]), y=(0.9, 2.5, [1.0, 1.5, 2.0, 2.5]), x_label="kernel length W, µs",
    fmt_x=lambda v: f"{v:.0f}", fmt_y=lambda v: f"{v:.1f}", label_w=56,
    ref_lines=[dict(y=1.1, label="1.1×", at="right")])
for w, s in zip(ws, speed):
    ax.point(w, s, "circle", "ink", size=10)
ax.point(w11, 1.1, "diamond", "hazard", size=10)
ax.text(w11, 1.1, f"model: 1.1× at {w11:.1f} µs", style="note", dx=12, dy=-12)
ax.text(0, speed[0], f"empty kernel, {speed[0]:.2f}×", style="note", dx=14, dy=5)
ax.text(2, speed[1], f"{speed[1]:.2f}× at 2 µs", style="note", dx=12, dy=-10)
fig.legend(body.x, 476, [dict(color="ink", label="measured", mark="circle"),
                         dict(color="ink-muted", label="model, derived", mark="dash", dash="proj"),
                         dict(color="hazard", label="1.1× crossing", mark="diamond")], direction="row")
fig.write()
