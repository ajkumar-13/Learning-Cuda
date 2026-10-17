"""Post 04, section 6: achieved bandwidth of the six rows of reduction.cu against the card's theoretical peak.

Run from the series root:  uv run python posts/04-reduction/diagrams/src/04-reduction-perf.py
Writes posts/04-reduction/diagrams/04-reduction-perf.svg.
A bar chart (cuda_fig_design.md section 5) of the GB/s column of the reduction.cu output pasted in the post
(measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026), with the 160 GB/s theoretical
peak as a reference line. The CPU row is host (blue), the per-element atomic hazard (red, a contended atomic), the
tree kernels ink. Every value is parsed from the pasted output, checked against its own time column, and the ratios
the prose of section 6 quotes are recomputed here and asserted to appear in the post.
"""
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
block = re.search(r"```text\n(NVIDIA GeForce GTX 1650[^\n]*\n.*?)```", BODY, re.S).group(1)
head = block.splitlines()[0]
PEAK = float(re.search(r"theoretical peak (\d+) GB/s", head).group(1))
BYTES = int(re.search(r"\((\d+) bytes\)", block).group(1))
ROWS = {}
for line in block.splitlines():
    m = re.match(r"(CPU, 1 thread|naive atomic|interleaved|sequential|seq\. \+ stride|warp shuffle)\s+(\S+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(.*)", line)
    if m:
        ROWS[m.group(1)] = dict(ms=float(m.group(3)), gbs=float(m.group(4)), pct=float(m.group(5)), check=m.group(6))
assert PEAK == 160 and BYTES == 67108864 and len(ROWS) == 6
for name, r in ROWS.items():
    lo, hi = BYTES / (r["ms"] + 0.0005) / 1e6, BYTES / (r["ms"] - 0.0005) / 1e6   # time printed to 0.001 ms
    assert lo - 0.05 <= r["gbs"] <= hi + 0.05, name
    assert abs(100 * r["gbs"] / PEAK - r["pct"]) < 0.1, name
    assert name == "CPU, 1 thread" or r["check"].startswith("PASS"), name

t = {k: r["ms"] for k, r in ROWS.items()}
assert t["naive atomic"] >= t["CPU, 1 thread"]              # "no faster than one CPU thread"
assert t["naive atomic"] > t["interleaved"] > t["sequential"] > t["seq. + stride"] > t["warp shuffle"]
claims = [f"is {t['naive atomic'] / t['interleaved']:.1f} times faster than the atomic",
          f"sequential addressing {t['interleaved'] / t['sequential']:.1f} times faster again",
          f"Of the fixes to the tree, the largest, {t['sequential'] / t['seq. + stride']:.1f} times, is the "
          f"grid-stride loop",
          f"cuts the remaining time by {100 * (1 - t['warp shuffle'] / t['seq. + stride']):.0f}%",
          f"reaches {ROWS['warp shuffle']['pct']:.0f}% of the theoretical peak"]
for c in claims:
    assert c in BODY, c
assert t["sequential"] / t["seq. + stride"] > max(t["interleaved"] / t["sequential"],
                                                  t["seq. + stride"] / t["warp shuffle"])

LABELS = {"CPU, 1 thread": "CPU, 1 thread", "naive atomic": "naive atomic", "interleaved": "interleaved",
          "sequential": "sequential", "seq. + stride": "sequential + stride", "warp shuffle": "warp shuffle"}
COLORS = {"CPU, 1 thread": "host", "naive atomic": "hazard"}
rows = [(LABELS[k], r["gbs"], COLORS.get(k, "ink")) for k, r in ROWS.items()]
listing = ", ".join(f"{LABELS[k]} {r['gbs']:.1f}" for k, r in ROWS.items())

fig = Figure(
    "04-reduction-perf", f"The warp-shuffle kernel reads at {ROWS['warp shuffle']['pct']:.0f}% of peak bandwidth",
    f"Horizontal bars of achieved bandwidth in GB/s for summing 16,777,216 floats, 64 MiB, on the GTX 1650: "
    f"{listing}. The CPU bar is blue, the per-element atomic red, the tree kernels ink. A dashed line marks the "
    f"theoretical peak of {PEAK:.0f} GB/s.",
    subtitle="Measured: GTX 1650, compute capability 7.5, CUDA 12.3, October 2026; median of 21 launches.",
    data_w=True)
full = fig.panel(fig.row(1)[0])
body = Box(full.x, full.y + 24, full.w, full.h - 24)           # room above the plot for the peak label
plot = fig.bar_chart(body, rows, 0, 160, [0, 40, 80, 120, 160], fmt=lambda v: f"{v:.1f}",
                     fmt_tick=lambda v: f"{v:.0f}", unit="", label_w=184, value_w=64, axis_label="GB/s")
# the peak as a reference line that stops above the last bar, so the last bar's value stays clear of it
step = plot.h / len(rows)
bh = max(8, min(24, math.floor(step * 0.6 / 8) * 8))           # the kit's bar height for this plot
last_top = plot.y + step * (len(rows) - 0.5) - bh / 2
px = plot.x + plot.w * PEAK / 160
with fig.data():
    fig.edge((px, plot.y - 8), (px, last_top - 8), color="ink-muted", width=1, dash="ref")
fig.text(px, plot.y - 16, f"theoretical peak {PEAK:.0f} GB/s", "note", anchor="end")
fig.caption("Bandwidth is the 64 MiB input over the time; each GPU row below the atomic adds one fix to the row above.")
fig.write()
