"""Post 09, figure 2: a barrier inside a branch waits for threads that never arrive; after the branch, all arrive.

Run from the series root:  uv run python posts/09-synchronization/diagrams/src/02-barrier.py
Writes posts/09-synchronization/diagrams/02-barrier.svg.
cuda_fig_design.md section 4 (a barrier is a purple vertical bar across the lanes it holds; a stall is red).
A block of eight threads stands for the post's 256, with the branch threadIdx.x < 4 standing for < 32; the
caption line says so. The outcomes named in the left note are the Programming Guide's ("likely to hang or
produce unintended side effects") and the post's section 2, asserted against the body.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "likely to hang or produce unintended side effects" in BODY
assert "will not succeed until reached by all non-exited threads in the block" in BODY
T, BR = 8, 4                                             # threads drawn; threads < BR take the branch

fig = Figure(
    "02-barrier", "A barrier must be reached by every thread of the block",
    "Two panels, each a block of eight threads on a time axis. Left: threads 0 to 3 take the branch "
    "if (t < 4), do its work and stop at a purple __syncthreads() bar inside it, then wait, drawn red; threads 4 "
    "to 7 skip the branch and never reach that barrier, so the outcome is undefined: the block can hang, or pass "
    "with its phases paired wrongly. Right: the barrier after the branch spans all eight threads; all arrive, "
    "then all continue.",
    subtitle="Eight threads stand for a block; the branch t < 4 stands for the post's threadIdx.x < 32.",
    data_w=True)

left, right = fig.row(2)
RH, Y0 = 28, 180
AW, BARW = 128, 8


def panel(box, heading, code, inside):
    fig.text(box.x, 124, heading, "head")
    fig.text(box.x, 152, code, "code")
    x0 = box.x + 80
    for t in range(T):
        y = Y0 + RH * t
        fig.text(box.x, y + 19, f"thread {t}", "label")
        b = Box(x0, y + 4, AW - 8, RH - 8)
        if t < BR:
            fig.fill(b, "compute-soft")
            fig.outline(b, "compute-line")
            fig.text(b.cx, y + 19, "work", "label", anchor="middle")
        else:
            fig.outline(b, "rule", dash="lead")
            fig.text(b.cx, y + 19, "skips", "note", anchor="middle")
    bar_x = x0 + AW
    n = BR if inside else T
    fig.fill(Box(bar_x, Y0, BARW, RH * n), "sync")
    after = bar_x + BARW + 8
    aw = box.right - after
    for t in range(T):
        y = Y0 + RH * t
        if inside and t < BR:
            b = Box(after, y + 4, aw, RH - 8)
            fig.fill(b, "hazard-soft")
            fig.outline(b, "hazard")
            fig.text(b.cx, y + 19, "waits", "label", anchor="middle", color="hazard")
        elif not inside:
            b = Box(after, y + 4, aw, RH - 8)
            fig.fill(b, "compute-soft")
            fig.outline(b, "compute-line")
            fig.text(b.cx, y + 19, "continues", "label", anchor="middle")
    return x0, bar_x, after, aw


x0, bar_x, after, aw = panel(left, "Barrier inside the branch", "if (t < 4) { work; __syncthreads(); }", True)
fig.text(after, Y0 + RH * 5 + 24, "threads 4 to 7 never", "label")
fig.text(after, Y0 + RH * 6 + 20, "arrive at this barrier", "label")
fig.text(left.x, 432, "Undefined: a hang, or phases paired wrongly.", "note")

x0, bar_x, after, aw = panel(right, "Barrier after the branch", "if (t < 4) { work; }  __syncthreads();", False)
fig.text(right.x, 432, "All eight threads arrive, then all continue.", "note")

fig.legend(left.x, 456, [dict(color="compute-soft", label="work"), dict(color="sync", label="__syncthreads()"),
                         dict(color="hazard-soft", label="waiting"), dict(color="rule", label="branch skipped", mark="dashed-outline")], direction="row")
fig.caption("A barrier counts the whole block, so it belongs where every thread goes.")
fig.write()
