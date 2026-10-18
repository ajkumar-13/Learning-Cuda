"""Post 16, section 2: why the order of a sum matters. Left, the FP32 number line near 10^8, where neighbors are 8
apart and 10^8 + 1 rounds back to 10^8. Right, the absolute error of one sum of 1,048,576 floats with 1, 14, 56
and 224 blocks.

Run from the series root:  uv run python posts/16-numerical-precision/diagrams/src/02-sum-order.py
Writes posts/16-numerical-precision/diagrams/02-sum-order.svg.
The neighbors of 10^8 are computed here with numpy.spacing. The errors are parsed from the model's block (b) and
from precision.cu's block (b) as pasted in index.md (run on the author's GTX 1650, compute capability 7.5, CUDA
12.3, October 2026); the script asserts that the two agree bit for bit and match the prose. cuda_fig_design.md
section 5: a chart with one series is ink; the reference spacing is a ref line.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROW = re.compile(r"^ +(\d+) x 256 +(\d+\.\d{4}) +(\d+\.\d{4})$", re.MULTILINE)
rows = ROW.findall(BODY)
assert len(rows) == 8, rows
model, gpu = rows[:4], rows[4:]
assert model == gpu, (model, gpu)              # the GPU's sums equal the model's, digit for digit
grids = [int(r[0]) for r in model]
errs = [float(r[2]) for r in model]
assert grids == [1, 14, 56, 224] and errs == [0.0406, 0.0219, 0.0406, 0.0844]
assert len({r[1] for r in model}) == 3
cpu = float(re.search(r"one CPU loop +\d+\.\d{4} +(\d+\.\d{4})", BODY).group(1))
assert cpu == 2.9594
SPACING = float(np.spacing(np.float32(524781.0)))
assert SPACING == 0.0625
BIG = np.float32(1e8)
gap = float(np.spacing(BIG))
assert gap == 8.0 and np.float32(BIG + np.float32(1.0)) == BIG
lo_n, hi_n = int(BIG) - int(gap), int(BIG) + int(gap)

fig = Figure(
    "02-sum-order", "Every addition rounds to a neighbor",
    "Two panels. Left: the FP32 number line near 10 to the 8, where the neighbors 99,999,992, 100,000,000 and "
    "100,000,008 are 8 apart; the exact sum 100,000,001 lies within half a spacing, 4, of 100,000,000 and rounds "
    "back to it, so the 1 is lost. Right: bars of the absolute error of one FP32 sum of 1,048,576 floats with "
    "1, 14, 56 and 224 blocks of 256 threads: 0.0406, 0.0219, 0.0406 and 0.0844, against one FP32 spacing of "
    "0.0625 near the sum, drawn as a reference line; a single CPU loop is off the scale at 2.9594.",
    subtitle="Block-count errors run on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026.",
    data_w=True)

left, right = fig.row(2, h=336)

# ---- left: the number line near 1e8 ----
body = fig.panel(left, "FP32 numbers near 100,000,000")
y = 288
x0, x1 = body.x + 16, body.right - 16
PX = (x1 - x0) / 20.0                          # 20 units of value across the line, 99,999,990 to 100,000,010
sx = lambda v: x0 + (v - (int(BIG) - 10)) * PX  # noqa: E731
fig.edge((x0, y), (x1, y), color="ink-muted", width=1.5)
for v in (lo_n, int(BIG), hi_n):
    fig.edge((sx(v), y - 12), (sx(v), y + 12), color="ink", width=1.5)
    fig.text(sx(v), y + 36, f"{v:,}", "tick", anchor="middle")
with fig.data():
    fig.edge((sx(int(BIG) + 4), y - 8), (sx(int(BIG) + 4), y + 8), color="ink-muted", width=1, dash="ref")
fig.text(sx(int(BIG) + 4), y + 60, "halfway, 100,000,004", "note", anchor="middle")
fig.brace(sx(int(BIG)), sx(hi_n), y - 24, label=f"spacing {gap:.0f}", side="above", kind="bracket")
with fig.data():
    fig.marker(sx(int(BIG) + 1), y, "circle", "hazard", size=10, hollow=True)
fig.text(sx(int(BIG) + 1), y - 104, "exact 100,000,001", "label", anchor="middle", color="hazard-ink")
fig.arrow((sx(int(BIG) + 1), y - 92), (sx(int(BIG) + 1), y - 10), color="hazard")
fig.text(body.x, y + 116, "100,000,001 is within 4 of 100,000,000: it rounds there", "note")

# ---- right: errors by block count ----
rb = fig.panel(right, "One sum of 1,048,576 floats, abs error")
bars = [(f"{g} block" + ("s" if g > 1 else ""), e, "ink") for g, e in zip(grids, errs)]
fig.bar_chart(rb, bars, 0, 0.1, [0, 0.025, 0.05, 0.075, 0.1], label_w=96, value_w=64,
              axis_label=f"absolute error; one CPU loop: {cpu:.4f}, off the scale",
              fmt_tick=lambda v: f"{v:g}", fmt_value=lambda v: f"{v:.4f}",
              ref_lines=[dict(x=SPACING, label="one spacing, 0.0625")])
fig.caption("A term below half a spacing vanishes; each block count rounds at different places.")
fig.write()
