"""Post 11, hero: the loop of the post, with the check before the measurement and every fix sent back through both.

Run from the series root:  uv run python posts/11-profiling-debugging/diagrams/src/01-profiling-loop.py
Writes posts/11-profiling-debugging/diagrams/01-profiling-loop.svg.
cuda_fig_design.md section 0: the kernel is compute (orange), the measurement taken on the device is host (blue,
"the profiler reading the device"), the check and the model are neutral cards. The numbers on the cards are the
body's: memcheck's and racecheck's clean runs (section 2 and 3), the 145.2 GB/s of the measured table (section 5)
and the 91% of the roofline ceiling (section 6), each asserted against index.md.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "========= ERROR SUMMARY: 0 errors" in BODY
assert "`0 hazards displayed (0 errors, 0 warnings)`" in BODY
assert "  256          4        32         100%" in BODY and "145.2 GB/s" in BODY
assert "91% of that ceiling" in BODY
GBS, ROOF = "145.2 GB/s", "91%"

fig = Figure(
    "01-profiling-loop", "Check before timing, and again after every fix",
    "A loop of five cards from left to right. 1, write: the kernel, saxpy. 2, check: memcheck and racecheck, "
    f"0 errors. 3, measure: CUDA events, {GBS}. 4, read the bottleneck: memory bound, {ROOF} of its roofline "
    "ceiling. 5, fix: one limiter and nothing else. An arrow runs from the fix back under the row to the check, labeled: every "
    "fix is checked and measured again. Orange marks the kernel, blue the measurement on the device, neutral the "
    "check and the model.",
    subtitle="SAXPY on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026.",
    data_w=True)

W, GAP, Y, H = 160, 20, 144, 144
steps = [
    ("1. Write", ["the kernel"], "saxpy", "compute"),
    ("2. Check", ["memcheck and", "racecheck"], "0 errors", None),
    ("3. Measure", ["CUDA events"], GBS, "host"),
    ("4. Read", ["memory bound"], f"{ROOF} of its roof", None),
    ("5. Fix", ["one limiter"], "and nothing else", "compute"),
]
cards = []
for k, (head, lines, result, color) in enumerate(steps):
    x = 40 + k * (W + GAP)
    body = fig.card(x, Y, W, H, heading=head, lines=lines, color=color)
    cards.append(body)
    if result is not None:
        style = "code" if result == "saxpy" else ("label" if color == "compute" else "value")
        fig.text(body.x, body.y + 16 + 48 + 12, result, style)

boxes = [c for c in cards]
for a, b in zip(boxes, boxes[1:]):
    fig.arrow((a.right + 16, Y + H // 2), (b.x - 16, Y + H // 2))

bottom = Y + H
loop_y = bottom + 56
fig.arrow((cards[4].cx, bottom), (cards[1].cx, bottom), via=((cards[4].cx, loop_y), (cards[1].cx, loop_y)))
fig.text((cards[1].cx + cards[4].cx) / 2, loop_y + 28, "every fix is checked and measured again", "note", anchor="middle")

fig.legend(40, 432, [dict(color="compute-soft", label="the kernel"),
                     dict(color="host-soft", label="measured on the device"),
                     dict(color="ink-muted", label="the check and the model", mark="outline")], direction="row")
fig.caption("Nsight Systems and Nsight Compute belong to steps 3 and 4; both needed administrator rights here.")
fig.write()
