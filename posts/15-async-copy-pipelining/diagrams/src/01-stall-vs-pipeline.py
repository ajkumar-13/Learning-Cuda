"""Post 15, the hero: the synchronous tile loop stalls its arithmetic during every load; the pipelined loop overlaps
the load of tile t + 1 with the compute of tile t.

Run from the series root:  uv run python posts/15-async-copy-pipelining/diagrams/src/01-stall-vs-pipeline.py
Writes posts/15-async-copy-pipelining/diagrams/01-stall-vs-pipeline.svg.
cuda_fig_design.md section 4 (a timeline: one lane per engine, a copy green, work orange, a stall red) and section 4's
pipeline form. The schedule is the balanced case of snippets/pipeline_model.py (8 tiles, load 10 and compute 10 model
units each); the start and end of every bar are scheduled here by the model's rules and the two makespans are asserted
against the model's makespans() and the 160, 90 and 1.78x the post quotes. Model units, not a measurement.
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from pipeline_model import makespans  # noqa: E402

TILES, L, C = 8, 10, 10
T_SYNC, T_PIPE = makespans([L] * TILES, [C] * TILES)
assert (T_SYNC, T_PIPE) == (160, 90)
BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "eight tiles fall from 160 to 90 units, 1.78 times faster" in BODY

# synchronous: load t, then compute t, in series
sync_loads = [(t * (L + C), t * (L + C) + L) for t in range(TILES)]
sync_comps = [(a + L, a + L + C) for a, _ in sync_loads]
# pipelined: load 0 alone (prologue); step t runs load t + 1 beside compute t; compute 7 alone (epilogue)
pipe_loads = [(0, L)]
pipe_comps = []
clock = L
for t in range(TILES - 1):
    step = max(L, C)
    pipe_loads.append((clock, clock + L))
    pipe_comps.append((clock, clock + C))
    clock += step
pipe_comps.append((clock, clock + C))
assert sync_comps[-1][1] == T_SYNC and pipe_comps[-1][1] == T_PIPE

fig = Figure(
    "01-stall-vs-pipeline", "Pipelining hides each load behind the previous tile's math",
    "Two timelines in model units with a load lane and a compute lane, for 8 tiles that take 10 units to load and 10 "
    "to compute. Synchronous: load and compute alternate, the compute lane stalls in red during each of the 8 loads, "
    "and the loop ends at 160 units. Pipelined: load 0 alone is the prologue, then the load of tile t + 1 runs beside "
    "the compute of tile t for 7 steps, then compute 7 alone is the epilogue; the compute lane stalls only during the "
    "first load and the loop ends at 90 units, 1.78 times sooner.",
    subtitle="Model units from snippets/pipeline_model.py: 8 tiles, 10 units to load and 10 to compute each.",
    data_w=True)

X0, SCALE, LH = 160, 4.2, 32
LANE_X = 40


def px(u):
    return X0 + u * SCALE


def bar(span, y, color, label):
    a, b = span
    box = Box(px(a), y, (b - a) * SCALE, LH)
    fig.fill(box, color)
    fig.outline(box, "rule")
    if label is not None:
        fig.text(box.cx, y + LH / 2 + 5, label, "label", anchor="middle", snap=False)


def timeline(y0, heading, loads, comps, end, end_label):
    fig.text(LANE_X, y0 - 12, heading, "head")
    fig.text(LANE_X, y0 + 21, "load tile", "label")
    fig.text(LANE_X, y0 + LH + 8 + 21, "compute tile", "label")
    with fig.data():
        for t, s in enumerate(loads):
            bar(s, y0, "memory-soft", str(t))
        busy = sorted(comps)
        idle, cur = [], 0
        for a, b in busy:
            if a > cur:
                idle.append((cur, a))
            cur = b
        for s in idle:
            bar(s, y0 + LH + 8, "hazard-soft", None)
        for t, s in enumerate(comps):
            bar(s, y0 + LH + 8, "compute-soft", str(t))
        fig.text(px(end) + 12, y0 + LH + 8 + 21, end_label, "value", snap=False)


Y1, Y2 = 136, 268
timeline(Y1, "Synchronous, as post 03: load, barrier, compute", sync_loads, sync_comps, T_SYNC, f"{T_SYNC} units")
timeline(Y2, "Pipelined: load tile t + 1 while computing tile t", pipe_loads, pipe_comps, T_PIPE,
         f"{T_PIPE} units, {T_SYNC / T_PIPE:.2f}×")

# phases of the pipelined loop
YB = Y2 + 2 * LH + 8 + 8
with fig.data():
    fig.brace(px(0), px(L), YB, label="prologue", side="below")
    fig.brace(px(L), px(T_PIPE - C), YB, label="steady state: both lanes busy", side="below")
    fig.brace(px(T_PIPE - C), px(T_PIPE), YB, label="epilogue", side="below")

# time axis
YA = 412
with fig.data():
    fig.edge((px(0), YA), (px(T_SYNC), YA), color="ink-muted")
    for u in range(0, T_SYNC + 1, 40):
        fig.edge((px(u), YA), (px(u), YA + 6), color="ink-muted")
        fig.text(px(u), YA + 24, str(u), "tick", anchor="middle", snap=False)
fig.text(LANE_X, YA + 24, "time, units", "tick")

fig.legend(px(0), 464, [dict(color="memory-soft", label="load of a tile into shared memory"),
                        dict(color="compute-soft", label="multiply-adds on a tile"),
                        dict(color="hazard-soft", label="arithmetic stalled")], direction="row")
fig.caption("Eight balanced tiles: one exposed load instead of eight.")
fig.write()
