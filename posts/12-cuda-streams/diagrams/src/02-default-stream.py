"""Post 12, section 2: what a stream orders, and what the legacy default stream orders.

Run from the series root:  uv run python posts/12-cuda-streams/diagrams/src/02-default-stream.py
Writes posts/12-cuda-streams/diagrams/02-default-stream.svg.
cuda_fig_design.md section 4 (a timeline: one lane per stream, copies green, kernels orange, an ordering point
purple, a stall red-hatched). The rules drawn are the CUDA 12.3 Programming Guide's, sections 3.2.8.5.2 and
3.2.8.5.4, quoted in the post: commands in one stream run in order, commands in different streams may overlap,
and any command to the NULL stream (the legacy default stream) keeps two commands of other streams from running
concurrently. The order is the point; the lengths of the boxes are illustrative, not measured, and the subtitle
says so.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "any CUDA command to the NULL stream" in BODY

fig = Figure(
    "02-default-stream", "Streams overlap until a default-stream command fences them",
    "Two timelines with lanes for stream 1, stream 2 and the default stream. Top: stream 1 copies in, runs a "
    "kernel and copies out, in that order; stream 2 does the same, and its copy and kernel overlap those of stream "
    "1, and both finish at time 11. Bottom: the same work with one cudaMemcpy on the default stream issued after "
    "stream 1's kernel. It waits for the work issued before it, everything issued after it in both streams waits "
    "for it, the waits are red, and the work finishes at time 16.",
    subtitle="The order follows the CUDA Programming Guide; the box lengths are illustrative, not measured.",
    height=540, data_w=True)

LX, X0, U, LH = 40, 184, 40, 36          # label column, time origin, one time unit, lane height


def op(lane_y, t0, t1, kind, text):
    b = Box(X0 + U * t0, lane_y + 4, U * (t1 - t0), LH - 8)
    fill = {"copy": "memory-soft", "kernel": "compute-soft", "default": "sync-soft"}[kind]
    line = {"copy": "memory", "kernel": "compute", "default": "sync"}[kind]
    fig.fill(b, fill, radius=4)
    fig.outline(b, line, radius=4)
    fig.text(b.cx, b.y + 19, text, "label", anchor="middle")
    return b


def gap(lane_y, t0, t1, at=None):
    b = Box(X0 + U * t0, lane_y + 4, U * (t1 - t0), LH - 8)
    fig.fill(b, "hazard-soft", radius=4)
    fig.outline(b, "hazard", dash="lead", radius=4)
    fig.text(X0 + U * at if at is not None else b.cx, b.y + 19, "waits", "label", anchor="middle", color="hazard-ink")


def lanes(y0, heading, names):
    fig.text(LX, y0 - 12, heading, "head")
    for r, name in enumerate(names):
        fig.text(LX, y0 + LH * r + 23, name, "label")
        fig.edge((X0, y0 + LH * (r + 1)), (X0 + U * 18 - 8, y0 + LH * (r + 1)), color="border", width=0.75)


NAMES = ["stream 1", "stream 2", "default stream"]

# top: two streams, no default-stream command
Y1 = 136
lanes(Y1, "Two streams: each in order, the two overlapping", NAMES)
op(Y1, 0, 3, "copy", "copy in")
op(Y1, 3, 5, "kernel", "kernel")
op(Y1, 5, 8, "copy", "copy out")
op(Y1 + LH, 3, 6, "copy", "copy in")
op(Y1 + LH, 6, 8, "kernel", "kernel")
op(Y1 + LH, 8, 11, "copy", "copy out")
fig.text(X0 + U * 11 + 16, Y1 + LH + 23, "done at 11", "note")

# bottom: one cudaMemcpy on the default stream between the two streams' work
Y2 = 312
lanes(Y2, "The same work with one cudaMemcpy on the default stream", NAMES)
op(Y2, 0, 3, "copy", "copy in")
op(Y2, 3, 5, "kernel", "kernel")
op(Y2 + 2 * LH, 5, 8, "default", "cudaMemcpy")
gap(Y2, 5, 8)
gap(Y2 + LH, 3, 8, at=4)
op(Y2, 8, 11, "copy", "copy out")
op(Y2 + LH, 8, 11, "copy", "copy in")
op(Y2 + LH, 11, 13, "kernel", "kernel")
op(Y2 + LH, 13, 16, "copy", "copy out")
fig.text(X0 + U * 16 + 16, Y2 + LH + 23, "done at 16", "note")
for t in (5, 8):
    fig.edge((X0 + U * t, Y2 - 4), (X0 + U * t, Y2 + 3 * LH + 4), color="sync", width=1.5)
fig.text(X0 + U * 5, Y2 + 3 * LH + 24, "waits for earlier work", "note", anchor="middle")
fig.text(X0 + U * 8 + 8, Y2 + 3 * LH + 24, "both streams wait for it", "note")

fig.legend(LX, 476, [dict(color="memory-soft", label="copy"), dict(color="compute-soft", label="kernel"),
                     dict(color="sync-soft", label="default-stream command"),
                     dict(color="hazard-soft", label="idle, waiting")], direction="row")
fig.caption("One command on the default stream turns two overlapping streams back into a serial program.")
fig.write()
