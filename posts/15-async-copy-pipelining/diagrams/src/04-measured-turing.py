"""Post 15, section 5: GFLOP/s of the three kernels at 1,024 cubed on the GTX 1650, with four blocks per SM and
with one.

Run from the series root:  uv run python posts/15-async-copy-pipelining/diagrams/src/04-measured-turing.py
Writes posts/15-async-copy-pipelining/diagrams/04-measured-turing.svg.
Every value is parsed from the output of snippets/async_pipeline_sm75.cu pasted in index.md section 5 (measured on
the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026) and asserted against the numbers the prose
quotes. cuda_fig_design.md section 5: a speed comparison is a bar_chart on one scale with the number on each bar and
the baseline in ink. The __pipeline_* rows are the software fallback (no cp.async on 7.5), drawn muted and named so.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROW = re.compile(r"^  (synchronous \(post 03\)|pipelined, __pipeline_\*|pipelined, via registers) +(\d+) +(\d+) +"
                 r"([\d.]+) +(\d+) +([\d.]+)x$", re.MULTILINE)
rows = ROW.findall(BODY)
assert len(rows) == 6, rows
full, one = rows[:3], rows[3:]
assert [int(r[1]) for r in full] == [4, 4, 4] and [int(r[1]) for r in one] == [1, 1, 1]
gf_full = [int(r[4]) for r in full]
gf_one = [int(r[4]) for r in one]
assert gf_full == [332, 296, 323] and gf_one == [174, 175, 281], (gf_full, gf_one)
assert float(one[2][5]) == 1.61
clock = re.search(r"SM clock: (\d+) MHz before the timed launches, (\d+) MHz after", BODY).groups()
assert clock == ("1445", "1500")
assert "332 to 174 GFLOP/s" in BODY and "at 281 GFLOP/s" in BODY

NAMES = ["synchronous", "fallback, no cp.async", "via registers"]
COLORS = ["ink", "ink-muted", "compute"]

fig = Figure(
    "04-measured-turing", "Pipelining pays only when few warps share an SM",
    "Two bar charts of GFLOP/s for a 1,024 cubed matrix multiply. Left, four blocks and 32 warps per SM: the "
    "synchronous kernel 332, the __pipeline_ software fallback 296, the register-staged pipeline 323. Right, one "
    "block and 8 warps per SM: synchronous 174, fallback 175, register-staged pipeline 281, 1.61 times the "
    "synchronous kernel. Notes under the charts give the ratios: 0.89 and 0.97 with four blocks, 1.00 and 1.61 with one.",
    subtitle="Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, at 1,445 to 1,500 MHz.",
    data_w=True)

left, right = fig.row(2, h=312)
for box, head, vals in ((left, "Four blocks per SM, 32 warps", gf_full), (right, "One block per SM, 8 warps", gf_one)):
    body = fig.panel(box, head)
    bars = [(n, v, c) for n, v, c in zip(NAMES, vals, COLORS)]
    fig.bar_chart(body, bars, 0, 400, [0, 100, 200, 300, 400], label_w=152, value_w=40,
                  axis_label="GFLOP/s, fastest of 5 passes",
                  fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.0f}")
fig.note(left, [f"fallback {float(full[1][5]):.2f}×, via registers {float(full[2][5]):.2f}× the synchronous kernel"])
fig.note(right, [f"fallback {float(one[1][5]):.2f}×, via registers {float(one[2][5]):.2f}× the synchronous kernel"])
fig.caption("With four blocks per SM other blocks cover the stall; with one, only the register pipeline recovers it.")
fig.write()
