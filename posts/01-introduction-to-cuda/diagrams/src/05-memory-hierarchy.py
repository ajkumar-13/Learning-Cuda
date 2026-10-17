"""Post 01, section 6: the memory ladder of the GTX 1650, from registers down to global memory.

Run from the series root:  uv run python posts/01-introduction-to-cuda/diagrams/src/05-memory-hierarchy.py
Writes posts/01-introduction-to-cuda/diagrams/05-memory-hierarchy.svg.
The ladder drawn once in full in the series (cuda_fig_design.md section 2): four green tiers, registers at the top and
global memory at the bottom, each with its scope and its size. The sizes are read from the device_query output pasted
in section 7 of the post (measured on the author's card, CUDA 12.3, October 2026). Latencies are shown only where the
post states them, labeled illustrative (about a cycle for a register, hundreds of cycles for global memory).
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
BODY = (POST / "index.md").read_text(encoding="utf-8")


def query(label, pattern):
    return re.search(re.escape(label) + r"\s*: " + pattern, BODY).group(1)


REGS = int(query("registers / SM", r"(\d+) \(32-bit\)"))
SHARED_KB = int(query("shared mem / block", r"(\d+) KB"))
L2_KB = int(query("L2 cache", r"(\d+) KB"))
GLOBAL_GIB = query("global memory", r"([\d.]+) GiB")
assert (REGS, SHARED_KB, L2_KB, GLOBAL_GIB) == (65536, 48, 1024, "4.00")
assert "a register answers in about a cycle" in BODY and "in hundreds" in BODY
assert "Those latencies are illustrative orders of magnitude" in BODY

TIERS = [  # name, scope, size on the card, illustrative latency (only where the post states one)
    ("Registers", "one thread", f"{num(REGS)} per SM", "about 1 cycle"),
    ("Shared memory", "one block, on chip beside L1", f"{SHARED_KB} KB per block", None),
    ("L2 cache", "the whole GPU", f"{num(L2_KB)} KB", None),
    ("Global memory (DRAM)", "the whole GPU, off chip", f"{GLOBAL_GIB} GiB", "hundreds of cycles"),
]

fig = Figure(
    "05-memory-hierarchy", "Fast and private at the top, slow and GPU-wide at the bottom",
    "A ladder of four green memory tiers that widen downward, with the size of each on the GTX 1650 as device_query "
    f"reports it. Registers: one thread, {num(REGS)} 32-bit registers per SM, about 1 cycle. Shared memory, "
    f"emphasized as the tier the kernel manages by hand: one block, on chip beside the L1 cache, {SHARED_KB} KB per "
    f"block. L2 cache: the whole GPU, {num(L2_KB)} KB. Global memory, off-chip DRAM: the whole GPU, {GLOBAL_GIB} GiB, "
    "hundreds of cycles. An arrow on the left points up, from slower and wider in scope to faster and narrower. The two latencies "
    "are illustrative.",
    subtitle="Sizes measured by device_query on the GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026.",
    data_w=True)

X, TOP, H, GAP = 168, 136, 72, 12
fig.text(920, TOP - 16, "latency, illustrative", "tick", anchor="end")
for k, (name, scope, size, lat) in enumerate(TIERS):
    y = TOP + k * (H + GAP)
    w = 400 + k * 64
    fig.card(X, y, w, H, heading=name, lines=[], color="memory")
    fig.text(X + 16, y + 56, f"{scope}; {size}", "note")
    if name == "Shared memory":
        fig.outline(Box(X, y, w, H), "memory", width=2.5, radius=8)
        fig.text(X + w + 16, y + 32, "managed by the kernel", "label", color="memory")
    if lat:
        fig.text(920, y + 44, lat, "label", anchor="end")
bottom = TOP + 4 * H + 3 * GAP
fig.arrow((96, bottom - 32), (96, TOP + 32))
fig.text(40, TOP + 12, "faster, narrower", "tick")
fig.text(40, bottom - 4, "slower, wider", "tick")
fig.caption("Inputs and outputs live at the bottom; fast kernels bring data up once and reuse it.")
fig.write()
