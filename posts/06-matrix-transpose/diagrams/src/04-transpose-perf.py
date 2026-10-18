"""Post 06, section 6: the three transposes against a plain copy, measured on the GTX 1650.

Run from the series root:  uv run python posts/06-matrix-transpose/diagrams/src/04-transpose-perf.py
Writes posts/06-matrix-transpose/diagrams/04-transpose-perf.svg.
cuda_fig_design.md section 5: a bar_chart on one scale with the number on each bar. Every bar is parsed from the output
of snippets/transpose.cu pasted in index.md section 6 (measured on the author's GTX 1650, compute capability 7.5,
CUDA 12.3, October 2026), checked against its own time column, and the ratios the prose quotes are recomputed and
asserted to appear in the post. The two kernels that still carry a hazard (strided global writes, a 32-way bank
conflict) are red; the padded kernel is ink; the copy kernel, the ceiling, is a dotted reference line in ink, and so is
the card's 160 GB/s theoretical peak printed on the output's first line.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
block = re.search(r"```text\n(device: NVIDIA GeForce GTX 1650[^\n]*\n.*?)```", BODY, re.S).group(1)
PEAK = float(re.search(r"theoretical peak (\d+) GB/s", block).group(1))
ROWS = {}
for m in re.finditer(r"^(copy|naive|shared|padded)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\S+)$", block, re.M):
    ROWS[m.group(1)] = dict(ms=float(m.group(2)), gbs=float(m.group(3)), pct=float(m.group(4)), check=m.group(5))
assert PEAK == 160 and list(ROWS) == ["copy", "naive", "shared", "padded"]
BYTES = 2 * 4096 * 4096 * 4
for name, r in ROWS.items():
    assert r["check"] == "ok", name
    lo, hi = BYTES / (r["ms"] + 0.0005) / 1e6, BYTES / (r["ms"] - 0.0005) / 1e6     # time printed to 0.001 ms
    assert lo - 0.05 <= r["gbs"] <= hi + 0.05, name
    assert abs(100 * r["gbs"] / ROWS["copy"]["gbs"] - r["pct"]) < 0.1, name
t = {k: r["ms"] for k, r in ROWS.items()}
claims = [f"$3.235 / 1.727 = {t['naive'] / t['shared']:.1f}$ times",
          f"$1.727 / 1.079 = {t['shared'] / t['padded']:.1f}$ times more, ${t['naive'] / t['padded']:.1f}$ times together",
          f"reaches {ROWS['padded']['pct']:.1f}% of the copy and {100 * ROWS['padded']['gbs'] / PEAK:.0f}% of the theoretical peak",
          f"it reaches 139.1 GB/s, {100 * ROWS['copy']['gbs'] / PEAK:.0f}% of the card's 160 GB/s theoretical peak"]
for c in claims:
    assert c in BODY, c
COPY = ROWS["copy"]["gbs"]

rows = [("naive", ROWS["naive"]["gbs"], "hazard"),
        ("shared, unpadded", ROWS["shared"]["gbs"], "hazard"),
        ("shared, padded", ROWS["padded"]["gbs"], "ink")]
fig = Figure(
    "04-transpose-perf", "The padded tile reaches 89.5% of a plain copy",
    f"Horizontal bars of bandwidth in GB/s for transposing a 4,096 by 4,096 float matrix on the GTX 1650: naive "
    f"{ROWS['naive']['gbs']} and shared unpadded {ROWS['shared']['gbs']}, in red for their hazards, and shared padded "
    f"{ROWS['padded']['gbs']} in ink. Dotted lines mark the copy kernel at {COPY} GB/s and the theoretical peak at "
    f"{PEAK:.0f} GB/s. As shares of the copy: 30%, 56% and 89.5%.",
    subtitle="Measured: GTX 1650, compute capability 7.5, CUDA 12.3, October 2026; median of 5 rounds of 20 launches.",
    data_w=True)
body = fig.panel(fig.row(1)[0])
fig.bar_chart(body, rows, 0, 180, [0, 40, 80, 120, 160], label_w=152, value_w=64, axis_label="GB/s, read plus write",
              fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.1f}",
              ref_lines=[dict(x=COPY, label=f"copy {COPY}"), dict(x=PEAK, label=f"peak {PEAK:.0f}")])
fig.caption("The tile fixes the strided writes (1.9 times); the pad fixes the bank conflict (1.6 times more).")
fig.write()
