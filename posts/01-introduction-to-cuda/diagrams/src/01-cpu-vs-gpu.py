"""Post 01, the hero: a CPU spends its area on a few large cores, cache and control; a GPU on many lanes.

Run from the series root:  uv run python posts/01-introduction-to-cuda/diagrams/src/01-cpu-vs-gpu.py
Writes posts/01-introduction-to-cuda/diagrams/01-cpu-vs-gpu.svg.
The CPU side is a schematic (no numbers; cuda_fig_design.md section 0: the CPU is host blue). The GPU side is the
author's GTX 1650: the SM count and the L2 size are read from the device_query output pasted in section 7 of the
post (measured on the card, CUDA 12.3, October 2026), and the 64 lanes per SM are the Programming Guide's
throughput for compute capability 7.5 (a cited value, labeled "cited"), asserted against the body.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
BODY = (POST / "index.md").read_text(encoding="utf-8")

SMS = int(re.search(r"streaming multiproc\. : (\d+) SMs", BODY).group(1))       # measured, device_query.cu
L2_KB = int(re.search(r"L2 cache {13}: (\d+) KB", BODY).group(1))              # measured, device_query.cu
CITED = {"lanes per SM": (64, "CUDA C++ Programming Guide, arithmetic throughput, compute capability 7.5")}
LANES = CITED["lanes per SM"][0]
assert (SMS, L2_KB, LANES) == (14, 1024, 64)
assert "each finish 64 single-precision" in BODY and f"That is {SMS * LANES} arithmetic lanes." in BODY
assert "The CPU side is a schematic, not a floor plan" in BODY

fig = Figure(
    "01-cpu-vs-gpu", "A CPU spends area on control; a GPU spends it on lanes",
    "Two panels. On the left, a schematic CPU: four large blue cores, a large green cache block and a gray control "
    "block for branch prediction and out-of-order execution, with no numbers because it is not to scale. On the "
    f"right, the GTX 1650: {SMS} streaming multiprocessors in two rows of 7, each a tile of {LANES} small orange "
    f"lanes, {SMS * LANES} in all, above one green L2 cache of {num(L2_KB)} KB shared by every SM. The SM count and "
    "the L2 size are measured by device_query; the 64 lanes per SM are cited from the CUDA Programming Guide.",
    subtitle="GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026: SMs and L2 measured; 64 lanes per SM cited.",
    data_w=True)

TOP = 136
# the CPU, schematic
fig.text(40, TOP - 16, "CPU, schematic", "head")
for k in range(4):
    x, y = 40 + (k % 2) * 152, TOP + (k // 2) * 88
    fig.card(x, y, 136, 72, heading="core", lines=[], color="host")
    fig.text(x + 16, y + 56, "one fast stream", "note")
fig.card(40, TOP + 176, 288, 72, heading="cache", lines=[], color="memory")
fig.text(56, TOP + 232, "L1, L2, L3", "note")
fig.card(40, TOP + 264, 288, 72, heading="control", lines=[], style="note")
fig.text(56, TOP + 320, "branch prediction, out of order", "note")

# the GPU: SMS tiles of LANES lanes, two rows
X0 = 376
fig.text(X0, TOP - 16, f"GTX 1650: {SMS} SMs × {LANES} lanes (cited) = {SMS * LANES}", "head")
per_row = (SMS + 1) // 2
side = int(LANES ** 0.5)
for k in range(SMS):
    x, y = X0 + (k % per_row) * 80, TOP + (k // per_row) * 128
    fig.outline(Box(x, y, 72, 120), "compute-line", radius=8)
    fig.text(x + 36, y + 20, f"SM {k}", "tick", anchor="middle")
    fig.grid(x + 8, y + 48, side, side, cell=7, fill=lambda i, j: "compute", lines_on_top=True)
fig.card(X0, TOP + 264, per_row * 80 - 8, 72, heading="L2 cache", lines=[], color="memory")
fig.text(X0 + 16, TOP + 320, f"{num(L2_KB)} KB, shared by every SM", "note")
fig.caption("Less control per thread, far more arithmetic: the GPU needs many threads to pay off.")
fig.write()
