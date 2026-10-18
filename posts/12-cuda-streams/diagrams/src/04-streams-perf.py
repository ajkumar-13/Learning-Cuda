"""Post 12, section 6: the measured end-to-end table of snippets/streams.cu, for both kernels.

Run from the series root:  uv run python posts/12-cuda-streams/diagrams/src/04-streams-perf.py
Writes posts/12-cuda-streams/diagrams/04-streams-perf.svg.
cuda_fig_design.md section 5: a bar_chart per kernel on one scale, the number on each bar, ink bars, the two
pageable rows hazard red (the staged copy is waste). The reference line is the two copies measured alone end to
end (copy in plus copy out), the floor for a pipeline whose copies take turns. Every value is parsed from the output
pasted in section 6 of the post (measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026)
and the parse is asserted to have found all nine rows.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
CLOCK = int(re.search(r"SM clock: \d+ MHz before the timed rows, (\d+) MHz after", BODY).group(1))
assert f"about {CLOCK:,} MHz" in BODY
NAMES = ["serial, pageable", "serial, pinned", "1 stream, pinned", "2 streams, pinned", "4 streams, pinned",
         "8 streams, pinned", "16 streams, pinned", "32 streams, pinned", "4 streams, pageable"]
ROWS = {}
for name in NAMES:
    m = re.search(rf"^{re.escape(name)} +([\d.]+) +[\d.]+ +[\d.]+x +([\d.]+) +[\d.]+x +ok$", BODY, re.M)
    ROWS[name] = (float(m.group(1)), float(m.group(2)))
T_IN = float(re.search(r"stages alone, pinned: copy A and B in ([\d.]+) ms", BODY).group(1))
T_OUT = float(re.search(r"copy C out ([\d.]+) ms \(", BODY).group(1))
COPIES = T_IN + T_OUT
assert f"{COPIES:.2f} ms" in BODY
HI = 70
assert max(max(v) for v in ROWS.values()) < HI


def label(name):
    return name.replace("serial, ", "serial ").replace(" streams, pinned", " streams").replace(" stream, pinned", " stream").replace(
        " streams, pageable", " streams pageable")


desc_rows = "; ".join(f"{n}: {a:.2f} and {b:.2f}" for n, (a, b) in ROWS.items())
fig = Figure(
    "04-streams-perf", "Pinned memory buys the most; streams then hide the kernel",
    "Two horizontal bar charts of end-to-end time in milliseconds, vector addition on the left and the kernel with "
    f"64 extra rounds on the right, for nine methods. Times, vector add and extra rounds: {desc_rows}. The two "
    f"pageable rows are red. A dashed line at {COPIES:.2f} ms marks the two copies measured alone, end to end.",
    subtitle=f"Fastest of 5 passes, the author's GTX 1650 (7.5) at about {CLOCK:,} MHz, CUDA 12.3, October 2026.",
    height=540, data_w=True)

left, right = fig.row(2)
for k, (box, heading) in enumerate(((left, "Vector addition, ms"), (right, "With 64 extra rounds, ms"))):
    body = fig.panel(box, heading)
    bars = [(label(n), ROWS[n][k], "hazard" if "pageable" in n else "ink") for n in NAMES]
    fig.bar_chart(body, bars, 0, HI, [0, 20, 40, 60], label_w=144, value_w=56,
                  fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.2f}",
                  ref_lines=[dict(x=COPIES, label="copies alone")], axis_label="time (ms)")
fig.caption("Every pinned pipeline lands near the copies' own time: the bus, not the kernel, sets the floor.")
fig.write()
