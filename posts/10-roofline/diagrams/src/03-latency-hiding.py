"""Post 10, section 6: attainable bandwidth against resident warps per SM, by Little's law, for 1, 2 and 4 loads in flight.

Run from the series root:  uv run python posts/10-roofline/diagrams/src/03-latency-hiding.py
Writes posts/10-roofline/diagrams/03-latency-hiding.svg.
Derived from snippets/roofline_model.py (imported): the bandwidth a kernel can draw is the bytes in flight divided by
the latency, 14 SMs x warps x loads per warp x 128 bytes / latency, capped at beta = 160 GB/s. The latency of 400 cycles
at 1,740 MHz is illustrative (Volkov, GTC 2010, gives "400+ cycles for memory"), and the subtitle says so. The knees
(21, 11 and 6 warps) are the model's printed values, asserted against the body. A section 5 line chart; the three
lines are compute orange, told apart by the marker at each knee and its label.
"""
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
import roofline_model as m  # noqa: E402

BETA = m.PEAK_BW / 1e9
LAT = m.LATENCY_CYCLES / m.CLOCK_HZ
PER_SM = m.PEAK_BW * LAT / m.SMS
BODY = (POST / "index.md").read_text(encoding="utf-8")


def gbs(warps, ilp):
    return min(BETA, m.SMS * warps * ilp * m.WARP_LOAD_BYTES / LAT / 1e9)


KNEES = {ilp: math.ceil(PER_SM / (ilp * m.WARP_LOAD_BYTES)) for ilp in (1, 2, 4)}
assert KNEES == {1: 21, 2: 11, 4: 6}
for ilp, w in KNEES.items():
    assert f"{ilp} independent 128-byte load(s) per warp: {w:2d} warps per SM" in BODY
    assert gbs(w, ilp) == BETA and gbs(w - 1, ilp) < BETA

WS = list(range(0, 33))
SHAPES = {1: "circle", 2: "square", 4: "diamond"}
fig = Figure(
    "03-latency-hiding", "More warps help until the bytes in flight cover the latency",
    "A line chart of attainable bandwidth in GB/s, 0 to 160, against resident warps per SM, 0 to 32 (0 to 100% "
    "occupancy), for an illustrative memory latency of 400 cycles at 1,740 MHz. With one 128-byte load in flight per "
    f"warp the line rises to 160 GB/s at {KNEES[1]} warps; with two independent loads at {KNEES[2]} warps; with four "
    f"at {KNEES[4]} warps. Each line is flat at 160 GB/s after its knee.",
    subtitle="Derived by roofline_model.py from Little's law; the 400-cycle latency is illustrative.",
    data_w=True)
body = fig.panel(fig.row(1)[0])
ax = fig.line_chart(
    body,
    [dict(xs=WS, ys=[gbs(w, ilp) for w in WS], color="compute", points=False, label=None) for ilp in (1, 2, 4)],
    x=(0, 32, [0, 8, 16, 24, 32]), y=(0, 200, [0, 40, 80, 120, 160, 200]),
    x_label="resident warps per SM (occupancy)", y_label="attainable bandwidth, GB/s",
    fmt_x=lambda v: f"{v:.0f} ({v / 32:.0%})", label_w=48,
    ref_lines=[dict(y=BETA, label="β = 160 GB/s", at="middle")])
for ilp, w in KNEES.items():
    ax.point(w, BETA, SHAPES[ilp], "compute", size=10)
    ax.text(w, BETA, f"{w} warps", style="value", anchor="middle", color="compute-ink", dy=-14)
fig.legend(ax.x + 16, ax.y + 24, [dict(color="compute", label=f"{k} {'load' if k == 1 else 'loads'} in flight per warp",
                                       mark=SHAPES[k]) for k in (1, 2, 4)], direction="row")
fig.caption("Past its knee a line is flat: more warps queue instead of adding bytes in flight.")
fig.write()
