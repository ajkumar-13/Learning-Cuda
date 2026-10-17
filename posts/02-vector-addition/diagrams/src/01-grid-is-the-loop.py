"""Post 02, hero (section 1): the CPU loop walks six indices in six steps; the GPU grid covers them in one.

Run from the series root:  uv run python posts/02-vector-addition/diagrams/src/01-grid-is-the-loop.py
Writes posts/02-vector-addition/diagrams/01-grid-is-the-loop.svg.
A Section 1 form of cuda_fig_design.md: the array is memory green, the CPU's single instruction stream host blue, the
GPU's threads compute orange. Both panels share one array C of six elements and one time axis of steps, so the
reader sees the loop's six steps collapse into one. The six elements are the post's small example (section 2 of the
post draws the same N = 6 as a launch); the step counts are the definition of a loop and of one thread per element,
not a measurement, so no timing is drawn.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure  # noqa: E402

POST = Path(__file__).resolve().parents[2]
BODY = (POST / "index.md").read_text(encoding="utf-8")
N = 6
assert "The figure at the top draws six elements both ways." in BODY

fig = Figure(
    "01-grid-is-the-loop", "The grid is the loop: one thread per element",
    f"Two panels over the same array C of {N} elements, C[0] to C[5], drawn in green above a time axis of steps. "
    f"Left, the CPU: one thread walks the loop, so in step 1 it writes C[0], in step 2 C[1], and so on to C[5] in "
    f"step {N}: a blue diagonal of {N} cells over {N} steps. Right, the GPU: {N} threads, one per index, write all "
    f"{N} elements in step 1, drawn as one orange row; steps 2 to {N} are an empty dashed region. A downward arrow on each side marks time. Each thread finds its index with "
    "i = blockIdx.x * blockDim.x + threadIdx.x.",
    subtitle=f"C[i] = A[i] + B[i] over {N} elements: {N} steps in sequence against one step in parallel.",
    data_w=True)

C, GY = 40, 200
for k, (px, head, color) in enumerate([(40, "CPU: one thread walks the loop", "host"),
                                        (520, "GPU: one thread per element", "compute")]):
    gx = px + 64
    fig.text(px, 128, head, "head")
    fig.strip(gx, 144, N, cell=C, values=[f"C[{j}]" for j in range(N)], font=13, fill=lambda j: "memory-soft")
    fig.text(px, 168, "array", "tick")
    rows = N if k == 0 else 1
    if k == 0:
        g = fig.grid(gx, GY, rows, N, cell=C, values=lambda i, j: f"i = {j}" if i == j else None, font=13,
                     fill=lambda i, j: "host-soft" if i == j else None)
        g.row_labels([f"step {t + 1}" for t in range(rows)], style="tick")
        fig.text(px, 476 - 8, f"for (i = 0; i < {N}; ++i) C[i] = A[i] + B[i];", "code")
    else:
        g = fig.grid(gx, GY, 1, N, cell=C, values=lambda i, j: f"i = {j}", font=13, fill=lambda i, j: "compute-soft")
        g.row_labels(["step 1"], style="tick")
        fig.outline(Box(gx, GY + C, N * C, (N - 1) * C), "ink-muted", width=1, dash="lead")
        fig.text(gx + 16, GY + 104, f"steps 2 to {N}: no work left;", "note")
        fig.text(gx + 16, GY + 128, "each thread wrote its own C[i]", "note")
        fig.text(px, 476 - 8, "i = blockIdx.x * blockDim.x + threadIdx.x", "code")
    fig.text(px, GY - 4, "time", "tick")
    fig.arrow((px + 8, GY + 8), (px + 8, GY + N * C - 8), label=None)
fig.caption("A loop takes N steps; a grid of N threads takes one.")
fig.write()
