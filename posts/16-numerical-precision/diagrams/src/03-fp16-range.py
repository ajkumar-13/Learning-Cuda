"""Post 16, section 3: the ranges of FP16, BF16 and FP32 on one logarithmic axis, and what each format does with
1e-8, 1e-5 and 100,000.

Run from the series root:  uv run python posts/16-numerical-precision/diagrams/src/03-fp16-range.py
Writes posts/16-numerical-precision/diagrams/03-fp16-range.svg.
FP16's limits (smallest subnormal, smallest normal, largest) and the stored values are parsed from the model's
format table and block (c) as pasted in index.md and asserted against numpy; precision.cu's block (c), run on
the GTX 1650, must print the same FP16 and BF16 values. cuda_fig_design.md section 0: values held are memory
green, values lost (inf, 0) are hazard red.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure, MINUS, TIMES, rich, sup  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
f16 = np.finfo(np.float16)
SUB, NORM, MAX = float(f16.smallest_subnormal), float(f16.smallest_normal), float(f16.max)
assert "FP16 subnormals reach 5.960e-08" in BODY and re.search(r"FP16 +16 +5 +10 +65504 +6\.104e-05", BODY)
model = re.findall(r"^    (100000|1e-05|1e-08) +(\S+) +(\S+) +(\S+)$", BODY, re.MULTILINE)
gpu = re.findall(r"^    (100000|1e-05|1e-08) +(\S+) +(\S+)$", BODY, re.MULTILINE)
assert len(model) == 3 and len(gpu) == 3, (model, gpu)
assert [(v, h, b) for v, h, _, b in model] == gpu      # the device's conversions print the model's values
VALUES = [1e-8, 1e-5, 1e5]
stored16 = {float(v): h for v, h, _, _ in model}
assert stored16 == {1e5: "inf", 1e-5: "1.00136e-05", 1e-8: "0"}
with np.errstate(over="ignore"):
    assert float(np.float16(1e5)) == float("inf") and float(np.float16(1e-8)) == 0.0

ROWS = {"FP32": 1, "BF16": 2, "FP16": 3}


def p10(e):
    return rich("10", sup("", (MINUS if e < 0 else "") + str(abs(e)), italic=False)) if e else "1"


def sci(v):
    mant, exp = ("%.1e" % v).split("e")
    return rich(mant, f" {TIMES} ", p10(int(exp)))


fig = Figure(
    "03-fp16-range", "FP16 holds about 12 decades; BF16 holds FP32's",
    "Three rows on a logarithmic axis from 10 to the minus 9 to 10 to the 6. FP16: normal numbers from 6.1 times "
    "10 to the minus 5 to 65,504, a solid green band, and subnormal numbers down to 6.0 times 10 to the minus 8, a "
    "dashed band. BF16 and FP32: normal numbers across the whole axis and far past both ends. The values 10 to "
    "the minus 8, 10 to the minus 5 and 100,000 are marked on each row; on the FP16 row 100,000 becomes inf and "
    "10 to the minus 8 becomes 0, red diamonds, and 10 to the minus 5 is kept as a subnormal, a hollow circle.",
    subtitle="FP16 limits derived by precision_model.py; the device's conversions on the GTX 1650 print the same.",
    data_w=True)

box = Box(40, 112, 880, 352)
ax = fig.scatter(box, [dict(points=[], color="memory", shape="circle", label="value kept"),
                       dict(points=[], color="memory", shape="circle", hollow=True, label="kept as a subnormal"),
                       dict(points=[], color="hazard", shape="diamond", label="lost: inf or 0")],
                 x=(1e-9, 1e6, [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6]), y=(0.4, 3.6, [1, 2, 3]),
                 x_log=True, x_label="magnitude, logarithmic", label_w=24,
                 fmt_x=lambda v: p10(int(round(np.log10(v)))),
                 fmt_y=lambda v: {1: "FP32", 2: "BF16", 3: "FP16"}.get(int(v), ""))
for name in ("FP32", "BF16"):
    fig.range_mark(ax.sx(1e-9), ax.sx(1e6), ax.sy(ROWS[name]), color="memory")
y16 = ax.sy(3)
fig.range_mark(ax.sx(NORM), ax.sx(MAX), y16, color="memory")
with fig.data():
    sb = Box(ax.sx(SUB), y16 - 12, ax.sx(NORM) - ax.sx(SUB), 24)
    fig.outline(sb, "memory", dash="lead")
for name, r in ROWS.items():
    for v in VALUES:
        if name == "FP16" and v != 1e-5:
            ax.point(v, r, "diamond", "hazard", size=10)
        else:
            ax.point(v, r, "circle", "memory", size=10, hollow=(name == "FP16"))
ax.text(1e5, 3, "inf", "value", anchor="middle", color="hazard-ink", dy=-20)
ax.text(1e-8, 3, "0", "value", anchor="middle", color="hazard-ink", dy=-20)
ax.text(MAX, 3, "65,504", "note", anchor="end", dx=-28, dy=-20)
ax.text(NORM, 3, sci(NORM), "note", anchor="middle", dy=32)
ax.text(SUB, 3, sci(SUB), "note", anchor="middle", dy=32)
fig.text(916, ax.sy(1.5) + 5, "both continue far past each end", "note", anchor="end")
fig.caption("One large activation or one tiny gradient leaves FP16's range; BF16 and FP32 keep it.")
fig.write()
