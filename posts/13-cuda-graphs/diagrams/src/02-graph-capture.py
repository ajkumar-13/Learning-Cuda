"""Post 13, section 3: stream capture records launches as graph nodes; instantiation is paid once; a replay is one
call.

Run from the series root:  uv run python posts/13-cuda-graphs/diagrams/src/02-graph-capture.py
Writes posts/13-cuda-graphs/diagrams/02-graph-capture.svg.
cuda_fig_design.md section 0: the host program and the graph launch are host (blue), kernel nodes compute (orange),
the pool's allocation and free memory (green), dependency edges sync (purple). The layout is a diagram, not a
measurement; its three numbers are parsed from the run of snippets/graphs.cu pasted in index.md (the author's
GTX 1650, compute capability 7.5, CUDA 12.3, October 2026): the 59.9 us build of the 20-kernel graph, the 0.28 us
of host time per kernel on replay (W = 0 row), and the 3 nodes of the captured poolRoundTrip.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
BUILD = float(re.search(r"graph build \(capture 20 empty kernels \+ instantiate\): ([\d.]+) us, once", BODY).group(1))
G_ISSUE = float(re.search(r"^ +0 +[\d.]+ +[\d.]+ +([\d.]+) +[\d.]+ +[\d.]+x$", BODY, re.MULTILINE).group(1))
NODES = int(re.search(r"poolRoundTrip becomes a graph of (\d+) nodes", BODY).group(1))
assert (BUILD, G_ISSUE, NODES) == (59.9, 0.28, 3)
CU = (POST / "snippets" / "graphs.cu").read_text(encoding="utf-8")
assert "const int KERNELS = 20;" in CU and "cudaGraphInstantiate(&exec, graph, 0)" in CU

fig = Figure(
    "02-graph-capture", "Capture records the stream as a graph, built once",
    "Left: the capture code of graphs.cu, cudaStreamBeginCapture, twenty launches, cudaStreamEndCapture and "
    "cudaGraphInstantiate. Middle: the graph it records, kernel 1 to kernel 20 joined by dependency edges in stream "
    "order. Right: instantiation costs 59.9 µs once; each cudaGraphLaunch then issues all twenty kernels for 0.28 µs "
    "of host time per kernel. Bottom: a captured poolRoundTrip becomes three nodes, cudaMallocAsync, the kernel and "
    "cudaFreeAsync.",
    subtitle="graphs.cu on the author's GTX 1650 (7.5), CUDA 12.3, October 2026; times measured, layout not.",
    data_w=True)

# left: the capture code
code = ["cudaStreamBeginCapture(s, ...);",
        "for (k = 0; k < 20; ++k)",
        "    launch(s, cycles);",
        "cudaStreamEndCapture(s, &graph);",
        "cudaGraphInstantiate(&exec,",
        "    graph, 0);"]
fig.text(40, 120, "1. Capture a stream", "head")
cb = fig.card(40, 136, 312, None, lines=code, style="code", color="host", fit=True)

# middle: the recorded graph, a chain of kernel nodes
fig.text(424, 120, "2. The recorded graph", "head")
NX, NW, NH = 424, 152, 32
ys = [136, 192, 248, 304]
names = ["kernel 1", "kernel 2", "⋮", "kernel 20"]
boxes = []
for y, name in zip(ys, names):
    b = Box(NX, y, NW, NH)
    if name != "⋮":
        fig.fill(b, "compute-soft")
        fig.outline(b, "compute-line")
    fig.text(b.cx, y + 21, name, "label", anchor="middle")
    boxes.append(b)
for a, b in zip(boxes, boxes[1:]):
    fig.arrow((a.cx, a.bottom), (b.cx, b.y), color="sync")
fig.text(NX + NW + 12, 228, "edges from", "note")
fig.text(NX + NW + 12, 248, "stream order", "note")
fig.arrow((cb.right + 16, 200), (NX - 8, 200), color="host")

# right: instantiate once, replay many
fig.text(704, 120, "3. Instantiate, replay", "head")
ib = fig.card(704, 136, 216, None, lines=["instantiate: once", f"{BUILD:.1f} µs"], fit=True, color="host")
fig.card(704, 248, 216, None, lines=["each cudaGraphLaunch:", "20 kernels,", f"{G_ISSUE:.2f} µs host time each"],
         fit=True, color="host")
fig.arrow((NX + NW, 152), (704, 152), color="host")
fig.arrow((812, ib.bottom + 16), (812, 248), color="host")

# bottom: the pool round trip, captured
fig.text(40, 408, f"poolRoundTrip captured: {NODES} nodes", "head")
bx = [Box(40, 424, 200, 40), Box(296, 424, 136, 40), Box(488, 424, 200, 40)]
for b, name, col in zip(bx, ["cudaMallocAsync", "kernel", "cudaFreeAsync"], ["memory", "compute", "memory"]):
    fig.fill(b, f"{col}-soft")
    fig.outline(b, f"{col}-line")
    fig.text(b.cx, b.y + 25, name, "code" if name != "kernel" else "label", anchor="middle")
for a, b in zip(bx, bx[1:]):
    fig.arrow((a.right, a.cy), (b.x, b.cy), color="sync")
fig.text(712, 449, "cudaMalloc is refused", "note")
fig.caption("Stream order becomes edges; the setup is paid once, before the first replay.")
fig.write()
