"""Post 02, section 2: a launch of N = 6 in blocks of 4, eight threads, two of them stopped by the guard.

Run from the series root:  uv run python posts/02-vector-addition/diagrams/src/02-launch-config.py
Writes posts/02-vector-addition/diagrams/02-launch-config.svg.
A Section 1 form of cuda_fig_design.md, reduced to the case the post works by hand: one row of eight thread cells
(orange when the thread owns an element, red-edged and empty of color when it is surplus) under two braces that name
the blocks, and the array C (green) below, one arrow per write. Every count comes from launch() in
snippets/vector_add.py, the same function whose output the post pastes, and is asserted here.
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its tables on import
    from vector_add import launch  # noqa: E402

N, BS = 6, 4
BLOCKS, THREADS, SURPLUS = launch(N, BS)
assert (BLOCKS, THREADS, SURPLUS) == (2, 8, 2)
BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "$N = 6$ in blocks of 4 needs $(6 + 3) / 4 = 2$ blocks, 8 threads" in BODY

fig = Figure(
    "02-launch-config", "Six elements need two blocks; the guard stops two threads",
    f"A launch of N = {N} elements with blockSize = {BS}: blocksPerGrid = (6 + 3) / 4 = {BLOCKS} blocks, "
    f"{THREADS} threads. A row of {THREADS} thread cells, i = 0 to 7, under two braces, block 0 with threadIdx.x 0 "
    f"to 3 and block 1 with threadIdx.x 0 to 3. Threads i = 0 to 5 are orange and each has an arrow down to its cell "
    f"of the green array C, C[0] to C[5]. Threads i = 6 and 7 are outlined in red: they fail i < N and write "
    f"nothing. Thread i = 5 is emphasized. Below, the guarded line of the kernel and thread 1 of block 1 worked as i = 1 * 4 + 1 = 5.",
    subtitle=f"N = {N}, blockSize = {BS}; counts derived by launch() in snippets/vector_add.py.",
    data_w=True)

X0, CW, TY, CY = 160, 80, 216, 344
fig.text(40, 136, f"blocksPerGrid = ({N} + {BS - 1}) / {BS} = {BLOCKS}", "code")
fig.text(40, 160, f"threads = {BLOCKS} * {BS} = {THREADS}, surplus = {SURPLUS}", "code")

threads = fig.strip(X0, TY, THREADS, cell_w=CW, cell_h=48, values=[f"i = {t}" for t in range(THREADS)], font=14,
                    fill=lambda t: "compute-soft" if t < N else None)
for b in range(BLOCKS):
    fig.brace(X0 + b * BS * CW + 4, X0 + (b + 1) * BS * CW - 4, TY - 8, label=f"block {b}: threadIdx.x 0 to {BS - 1}",
              side="above")
threads.outline(N, THREADS - 1, color="hazard", width=2.5)
fig.text(40, TY + 30, "threads", "label")

arr = fig.strip(X0, CY, N, cell_w=CW, cell_h=48, values=[f"C[{j}]" for j in range(N)], font=14,
                fill=lambda j: "memory-soft")
fig.text(40, CY + 30, "array C", "label")
for j in range(N):
    fig.arrow((X0 + j * CW + CW / 2, TY + 48), (X0 + j * CW + CW / 2, CY))
fig.text(X0 + N * CW + 8, TY + 80, "fail i < N:", "label", color="hazard")
fig.text(X0 + N * CW + 8, TY + 100, "no write", "label", color="hazard")
threads.window(5, 5, color="compute")

fig.text(40, 436, "if (i < N) C[i] = A[i] + B[i];", "code")
fig.text(480, 436, "block 1, thread 1: i = 1 * 4 + 1 = 5", "code")
fig.caption("Blocks come whole, so the grid rounds up and the guard absorbs the excess.")
fig.write()
