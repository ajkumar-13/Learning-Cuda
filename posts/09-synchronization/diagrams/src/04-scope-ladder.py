"""Post 09, figure 4: the three scopes, what each orders, and what a fence at each cost on the GTX 1650.

Run from the series root:  uv run python posts/09-synchronization/diagrams/src/04-scope-ladder.py
Writes posts/09-synchronization/diagrams/04-scope-ladder.svg.
cuda_fig_design.md sections 0 and 5: the scopes as a ladder (a table from the narrowest to the widest, with the
fence and the scoped atomic of each, from the Programming Guide's sections "Memory Fence Functions" and "Atomic
Functions"), and a bar chart of part 4 of snippets/sync.cu, nanoseconds per iteration of a store-then-fence loop,
parsed from the run pasted in index.md section 6 (measured on the author's GTX 1650, compute capability 7.5,
CUDA 12.3, October 2026); fences are sync (purple), the no-fence baseline is ink.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure, num  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
block = BODY.split("4. store then fence, 14336 threads x 1000 iterations, fastest of 5 launches", 1)[1].split("```", 1)[0]
NS = {name: int(v) for name, v in re.findall(r"^\s+(no fence|__threadfence\w*\(\))\s+(\d+)\s*$", block, flags=re.M)}
assert NS == {"no fence": 39, "__threadfence_block()": 166, "__threadfence()": 328,
              "__threadfence_system()": 2919}, NS
assert "2.0 times from block to device, 8.9 times from device to system" in BODY
assert round(NS["__threadfence()"] / NS["__threadfence_block()"], 1) == 2.0
assert round(NS["__threadfence_system()"] / NS["__threadfence()"], 1) == 8.9

fig = Figure(
    "04-scope-ladder", "A wider scope orders more threads and costs more",
    "Top, a table of the three scopes from narrowest to widest: block, with __threadfence_block() and "
    "atomicAdd_block, reaches the threads of one block; device, with __threadfence() and atomicAdd, "
    "every thread of the GPU; system, with __threadfence_system() and atomicAdd_system, also host threads and "
    "peer GPUs. Bottom, a bar chart of nanoseconds per iteration of a store-then-fence loop: no fence 39, block "
    "166, device 328, system 2,919.",
    subtitle="Bars: part 4 of sync.cu on the author's GTX 1650 (7.5) at a 1,754 MHz SM clock, CUDA 12.3, October 2026.",
    data_w=True)

COLS = [96, 232, 176, 376]
ROWS = [("scope", "fence", "atomic", "reaches"),
        ("block", "__threadfence_block()", "atomicAdd_block", "the threads of one block"),
        ("device", "__threadfence()", "atomicAdd", "every thread of the GPU"),
        ("system", "__threadfence_system()", "atomicAdd_system", "also host threads and peer GPUs")]
TY, RH = 112, 32
fig.table(40, TY, [[r[0], "", "", r[3]] if i else list(r) for i, r in enumerate(ROWS)], COLS, row_h=RH,
          col_align=["start", "start", "start", "start"], highlight_row={1: "sync", 2: "sync", 3: "sync"})
for i, r in enumerate(ROWS[1:], start=1):
    y = TY + RH * i + 21
    fig.text(40 + COLS[0] + 8, y, r[1], "code")
    fig.text(40 + COLS[0] + COLS[1] + 8, y, r[2], "code")

rows = [("no fence", NS["no fence"], "ink"),
        ("block", NS["__threadfence_block()"], "sync"),
        ("device", NS["__threadfence()"], "sync"),
        ("system", NS["__threadfence_system()"], "sync")]
fig.text(40, 288, "Store, then fence: ns per iteration, 14,336 threads, fastest of 5 launches", "head")
fig.bar_chart(Box(40, 300, 880, 176), rows, 0, 8000, [0, 2000, 4000, 6000, 8000], label_w=104, value_w=72,
              fmt=lambda v: num(v), unit=" ns")
fig.caption("Choose the narrowest scope that reaches every thread that must see the write.")
fig.write()
