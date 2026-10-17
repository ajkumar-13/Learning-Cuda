"""Post 05, section 4: privatization beats global atomics on both inputs, measured on the GTX 1650.

Run from the series root:  uv run python posts/05-histogram/diagrams/src/03-histogram-perf.py
Writes posts/05-histogram/diagrams/03-histogram-perf.svg.
Every bar is parsed from the output of snippets/histogram.cu pasted in index.md section 4 (measured on the author's
GTX 1650, compute capability 7.5, CUDA 12.3, October 2026) and asserted against the table and the ratios the body
states. cuda_fig_design.md section 5: a speed comparison is a bar_chart on one scale with the number on each bar; the
contended global atomics are hazard red, the private shared-memory histogram memory green; the dotted line is the
card's 160 GB/s theoretical peak, from post 01's device query.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROW = re.compile(r"^(random|solid)\s+(histGlobal|histShared)\s+([\d.]+)\s+([\d.]+)\s+yes$", re.MULTILINE)
GBS = {(d, k): float(g) for d, k, ms, g in ROW.findall(BODY)}
MS = {(d, k): float(ms) for d, k, ms, g in ROW.findall(BODY)}
assert GBS == {("random", "histGlobal"): 13.5, ("random", "histShared"): 117.2,
               ("solid", "histGlobal"): 6.6, ("solid", "histShared"): 43.1}, GBS
assert "| `histGlobal`, global atomics | 4.967 ms, 13.5 GB/s | 10.193 ms, 6.6 GB/s |" in BODY
assert "| `histShared`, privatized | 0.573 ms, 117.2 GB/s | 1.557 ms, 43.1 GB/s |" in BODY
RATIO = {d: MS[(d, "histGlobal")] / MS[(d, "histShared")] for d in ("random", "solid")}
assert (round(RATIO["random"], 1), round(RATIO["solid"], 1)) == (8.7, 6.5)
assert "8.7 times on random data, 6.5 times on the solid color" in BODY
PEAK = 160
assert "160 GB/s theoretical peak" in BODY

fig = Figure(
    "03-histogram-perf", "Privatization wins on random and on solid data",
    "Two horizontal bar charts on one scale of 0 to 160 GB/s, measured on the author's GTX 1650. Random values: "
    f"global atomics {GBS[('random', 'histGlobal')]} GB/s, privatized {GBS[('random', 'histShared')]} GB/s, "
    f"{RATIO['random']:.1f} times faster. Solid color: global atomics {GBS[('solid', 'histGlobal')]} GB/s, "
    f"privatized {GBS[('solid', 'histShared')]} GB/s, {RATIO['solid']:.1f} times faster. A dotted line marks the "
    f"{PEAK} GB/s theoretical peak.",
    subtitle="Measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026; one of 5 matching runs.",
    data_w=True)

panels = fig.row(2, y=128, h=280)
for box, (data, head) in zip(panels, [("random", "Random values 0 to 255"), ("solid", "Solid color, every value 128")]):
    body = fig.panel(box, head)
    fig.bar_chart(body, [("global atomics", GBS[(data, "histGlobal")], "hazard"),
                         ("privatized", GBS[(data, "histShared")], "memory")],
                  0, PEAK, [0, 40, 80, 120, 160], label_w=120, value_w=56, axis_label="input read, GB/s",
                  fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.1f}",
                  ref_lines=[dict(x=PEAK, label=f"peak {PEAK}")])
    fig.text(body.x, body.bottom + 36, f"privatized: {RATIO[data]:.1f} times faster", "head")
fig.caption("Global atomics stay under 14 GB/s on either input; the private one reaches 43 to 117 GB/s.")
fig.write()
