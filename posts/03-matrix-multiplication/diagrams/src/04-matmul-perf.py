"""Post 03, section 5: GFLOP/s of the naive kernel, the tiled kernel and cuBLAS at four sizes, measured on the GTX 1650.

Run from the series root:  uv run python posts/03-matrix-multiplication/diagrams/src/04-matmul-perf.py
Writes posts/03-matrix-multiplication/diagrams/04-matmul-perf.svg.
cuda_fig_design.md section 5: a speed comparison as bar charts on one scale, the number on each bar, the library
baseline in ink. Every value is read from the output of snippets/matmul.cu pasted in section 5 of the post (measured on
the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026) and asserted against the ratios the body states.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
BODY = (POST / "index.md").read_text(encoding="utf-8")
ROWS = {}
for m in re.finditer(r"(?m)^ +(\d+) +([\d.]+) +(\d+) +([\d.]+) +(\d+) +([\d.]+) +(\d+) +[\d.e+-]+$", BODY):
    n, nms, ng, tms, tg, cms, cg = m.groups()
    ROWS[int(n)] = dict(naive=(float(nms), int(ng)), tiled=(float(tms), int(tg)), cublas=(float(cms), int(cg)))
assert sorted(ROWS) == [256, 512, 1024, 2048]
r = ROWS[1024]
assert round(r["naive"][0] / r["tiled"][0], 1) == 1.6 and round(r["tiled"][0] / r["cublas"][0], 1) == 7.2
assert "measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026".lower() in BODY.lower()

fig = Figure(
    "04-matmul-perf", "Tiling lifts the naive kernel; cuBLAS is far beyond both",
    "Four horizontal bar charts on one scale from 0 to 2,500 GFLOP/s, one per square size, each with three bars: "
    "the naive kernel (orange), the tiled kernel with TILE 16 (orange as well) and cuBLAS (ink). "
    + " ".join(f"At {num(n)} cubed: naive {num(v['naive'][1])}, tiled {num(v['tiled'][1])}, cuBLAS "
               f"{num(v['cublas'][1])}." for n, v in ROWS.items()),
    subtitle="Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026; median of 5 runs.",
    height=720, data_w=True)

panels = fig.row(2, y=136, h=232) + fig.row(2, y=416, h=232)
for box, (n, v) in zip(panels, ROWS.items()):
    body = fig.panel(box, f"{num(n)} × {num(n)} × {num(n)}")
    fig.bar_chart(body, [("naive", v["naive"][1], "compute"), ("tiled", v["tiled"][1], "compute"),
                         ("cuBLAS", v["cublas"][1], "ink")],
                  0, 2500, [0, 1000, 2000], label_w=72, value_w=56, axis_label="GFLOP/s")
fig.caption("At 1,024 cubed: tiled 1.6 times faster than naive, cuBLAS 7.2 times faster than tiled.")
fig.write()
