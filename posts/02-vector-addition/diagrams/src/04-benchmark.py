"""Post 02, section 4: the benchmark of vector addition, four measured times against N on log-log axes.

Run from the series root:  uv run python posts/02-vector-addition/diagrams/src/04-benchmark.py
Writes posts/02-vector-addition/diagrams/04-benchmark.svg.
A Section 5 chart of cuda_fig_design.md: every plotted time is parsed from the vector_add_bench output pasted in
section 4 of the post (measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026). The
dashed floor is derived: 12 bytes per element at the theoretical 160 GB/s, recomputed here and asserted against the
traffic table of snippets/vector_add.py pasted in section 5. CPU lines are host blue, GPU lines compute orange; the
two lines of each hue differ by marker shape and fill, and a legend names all five.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
BODY = (POST / "index.md").read_text(encoding="utf-8")


def us(cell):
    v, unit = cell.split()
    return float(v) * (1000.0 if unit == "ms" else 1.0)


ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+([\d.]+ [um]s)\s+([\d.]+ [um]s)\s+([\d.]+ [um]s)\s+([\d.]+ [um]s)"
                 r"\s+([\d.]+)\s+ok$", re.M)
rows = ROW.findall(BODY)
assert len(rows) == 5, rows
NS = [int(r[0]) for r in rows]
assert NS == [1000, 10000, 100000, 1000000, 10000000]
CPU1, OMP, COPIES, KERNEL = ([us(r[k]) for r in rows] for k in (2, 3, 4, 5))
GBS = [float(r[6]) for r in rows]
assert "GTX 1650 with Max-Q Design (compute capability 7.5)" in BODY

PEAK = 160e9
FLOOR = [n * 12 / PEAK * 1e6 for n in NS]
for n, f in zip(NS, FLOOR):                          # the model's printed floor, section 5
    assert f"{n:>10,} elements:" in BODY and f"at least {f:>7.2f} us" in BODY, (n, f)

fig = Figure(
    "04-benchmark", "The copies set a floor; only the kernel alone pulls away",
    "A log-log chart of time in microseconds against N, from 1,000 to 10 million elements, for four measured "
    f"variants of vector addition. CPU on one thread: {', '.join(num(v, 1) if v < 1000 else num(v / 1000, 2) + ' ms' for v in CPU1)}"
    f" (microseconds unless marked). CPU with OpenMP: {', '.join(num(v, 1) if v < 1000 else num(v / 1000, 2) + ' ms' for v in OMP)}. "
    f"GPU with copies: {', '.join(num(v, 1) if v < 1000 else num(v / 1000, 2) + ' ms' for v in COPIES)}. "
    f"GPU kernel alone: {', '.join(num(v, 1) for v in KERNEL)}. A dashed line is the derived floor at 160 GB/s, "
    f"from {num(FLOOR[0], 3)} to {num(FLOOR[-1], 0)} microseconds; at 10 million elements the kernel reaches "
    f"{num(GBS[-1], 1)} GB/s.",
    subtitle="Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026; floor derived.",
    data_w=True)

YT = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
body = Box(40, 104, 880, 372)
ax = fig.line_chart(
    body,
    [dict(xs=NS, ys=CPU1, color="host", shape="circle", label=None),
     dict(xs=NS, ys=OMP, color="host", shape="square", hollow=True, label=None),
     dict(xs=NS, ys=COPIES, color="compute", shape="triangle", label=None),
     dict(xs=NS, ys=KERNEL, color="compute", shape="diamond", hollow=True, label=None),
     dict(xs=NS, ys=FLOOR, color="ink-muted", dash="ref", points=False, label=None)],
    x=(1000, 1e7, [1000, 1e4, 1e5, 1e6, 1e7]), y=(0.01, 100000, YT),
    x_label="N, elements", y_label="time per run, µs", x_log=True, y_log=True, label_w=232)

LX = ax.right + 24
fig.legend(LX, 136, [dict(color="host", label="CPU, 1 thread", mark="circle"),
                     dict(color="host", label="CPU, OpenMP", mark="square", hollow=True),
                     dict(color="compute", label="GPU with copies", mark="triangle"),
                     dict(color="compute", label="GPU kernel alone", mark="diamond", hollow=True),
                     dict(color="ink-muted", label="floor at 160 GB/s", mark="dash")])
ax.text(NS[-1], KERNEL[-1], f"kernel alone: {num(GBS[-1], 1)} GB/s", style="value", anchor="end", dx=-4, dy=64,
        color="compute-ink")
ax.text(NS[1], COPIES[1], "copies and launch", style="note", anchor="middle", dy=-16)
fig.caption(f"At 10 million elements the kernel alone takes {num(KERNEL[-1], 1)} µs against a floor of {num(FLOOR[-1], 0)} µs.")
fig.write()
