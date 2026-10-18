"""Post 13, the hero: twenty 2 us kernels launched directly against the same kernels replayed from a graph.

Run from the series root:  uv run python posts/13-cuda-graphs/diagrams/src/01-launch-overhead.py
Writes posts/13-cuda-graphs/diagrams/01-launch-overhead.svg.
cuda_fig_design.md section 4 (a timeline: host work blue on the host row, kernels orange on the GPU row, a stall
red). Every length is the W = 2 row of the launch sweep pasted in index.md section 2 (snippets/graphs.cu on the
author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026): per kernel, the direct issue time 1.75 us, the
direct done time 3.52 us, the graph issue time 0.28 us and the graph done time 2.81 us. The drawing places twenty
kernels at those averages, so it is drawn to scale but is not a trace: the gap on the GPU row is the done time minus
the 2 us of the kernel, 1.52 us direct and 0.81 us in the graph.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROW = re.search(r"^ +2 +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)x$", BODY, re.MULTILINE)
D_ISSUE, D_DONE, G_ISSUE, G_DONE = (float(ROW.group(k)) for k in range(1, 5))
assert (D_ISSUE, D_DONE, G_ISSUE, G_DONE) == (1.75, 3.52, 0.28, 2.81)
W, N = 2.0, 20
D_GAP, G_GAP = round(D_DONE - W, 2), round(G_DONE - W, 2)
D_END, G_END, G_CALL = round(N * D_DONE, 1), round(N * G_DONE, 1), round(N * G_ISSUE, 1)
assert (D_GAP, G_GAP, D_END, G_END, G_CALL) == (1.52, 0.81, 70.4, 56.2, 5.6)
assert "twenty kernels end at 70.4 µs launched directly and at 56.2 µs replayed" in BODY

fig = Figure(
    "01-launch-overhead", "A graph replay narrows the gap before every kernel",
    "Two timelines of twenty kernels of 2 µs each, drawn to scale from measured averages. Direct launches: the host "
    "row shows twenty launch calls of 1.75 µs each, done after 35 µs; on the GPU row every kernel follows a gap of "
    "1.52 µs and the twentieth ends at 70.4 µs. Graph replay: the host row shows one cudaGraphLaunch call of 5.6 µs; "
    "on the GPU row the gaps shrink to 0.81 µs and the twentieth kernel ends at 56.2 µs.",
    subtitle="graphs.cu averages on the author's GTX 1650 (7.5), CUDA 12.3, October 2026; to scale, not a trace.",
    data_w=True)

X0, X1, T_MAX = 160, 784, 72.0
SX = (X1 - X0) / T_MAX
RH = 32


def bar(t0, t1, y, color):
    with fig.data():
        fig.fill(Box(X0 + t0 * SX, y, (t1 - t0) * SX, RH), color, fit=False)


def lane(y, title, host_bars, gap, done):
    fig.text(40, y - 12, title, "head")
    fig.text(40, y + 21, "host", "label")
    fig.text(40, y + RH + 8 + 21, "GPU", "label")
    for t0, t1 in host_bars:
        bar(t0, t1, y, "host-soft")
        with fig.data():
            fig.outline(Box(X0 + t0 * SX, y, (t1 - t0) * SX, RH), "host-line")
    gy = y + RH + 8
    for i in range(N):
        start = i * done
        bar(start, start + gap, gy, "hazard-soft")
        bar(start + gap, start + done, gy, "compute")
    with fig.data():
        fig.text(X0 + N * done * SX + 8, gy + 21, f"ends at {N * done:.1f} µs", "value", snap=False)
    return gy


# top: direct launches, one host call per kernel, each a small gap behind the previous one
y1 = 136
gy1 = lane(y1, "Direct: 20 launches", [(i * D_ISSUE, (i + 1) * D_ISSUE - 0.25) for i in range(N)], D_GAP, D_DONE)
with fig.data():
    fig.text(X0 + N * D_ISSUE * SX + 8, y1 + 21, "issued after 35 µs", "note", snap=False)
    fig.text(X0, gy1 + RH + 24, f"gap {D_GAP:.2f} µs before each kernel", "note", color="hazard", snap=False)

# bottom: one cudaGraphLaunch for all twenty kernels
y2 = 304
gy2 = lane(y2, "Graph: 1 cudaGraphLaunch", [(0.0, G_CALL)], G_GAP, G_DONE)
with fig.data():
    fig.text(X0 + G_CALL * SX + 8, y2 + 21, f"one call, {G_CALL:.1f} µs", "note", snap=False)
    fig.text(X0, gy2 + RH + 24, f"gap {G_GAP:.2f} µs", "note", color="hazard", snap=False)

# the time axis under both
ay = 432
with fig.data():
    fig.edge((X0, ay), (X0 + T_MAX * SX, ay), color="ink-muted")
    for t in range(0, 71, 10):
        x = X0 + t * SX
        fig.edge((x, ay), (x, ay + 4), color="ink-muted")
        fig.text(x, ay + 20, f"{t}", "tick", anchor="middle", snap=False)
fig.text(40, ay + 20, "time, µs", "tick")
fig.legend(X0, 488, [dict(color="host-soft", label="host call"),
                     dict(color="compute", label="kernel, 2 µs"),
                     dict(color="hazard-soft", label="gap on the GPU")], direction="row")
fig.write()
