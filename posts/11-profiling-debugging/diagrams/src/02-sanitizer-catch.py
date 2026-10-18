"""Post 11, section 2: the two surplus threads of missing_guard.cu run past the 24-byte arrays, and memcheck names them.

Run from the series root:  uv run python posts/11-profiling-debugging/diagrams/src/02-sanitizer-catch.py
Writes posts/11-profiling-debugging/diagrams/02-sanitizer-catch.svg.
cuda_fig_design.md sections 1 and 2: threads are compute (orange), the arrays memory (green), the out-of-bounds
accesses hazard (red, the region past the end dashed). N, the block size and the grid are read from
snippets/missing_guard.cu; the threads, blocks and byte offsets memcheck reported are parsed from the report pasted
in index.md section 2 (run on the author's GTX 1650, CUDA 12.3, October 2026) and asserted.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

SRC = (POST / "snippets" / "missing_guard.cu").read_text(encoding="utf-8")
m = re.search(r"const int N = (\d+), threads = (\d+), blocks = \(N \+ threads - 1\) / threads;", SRC)
N, T = int(m.group(1)), int(m.group(2))
B = (N + T - 1) // T
assert (N, T, B) == (6, 4, 2)
BYTES = 4 * N
BODY = (POST / "index.md").read_text(encoding="utf-8")
hits = re.findall(r"by thread \((\d+),0,0\) in block \((\d+),0,0\)\n=+ +Address 0x([0-9a-f]+) is out of bounds\n"
                  r"=+ +and is \d+ bytes after the nearest allocation at 0x([0-9a-f]+) of size (\d+) bytes", BODY)
assert len(hits) == 2, hits
offsets = [int(a, 16) - int(base, 16) for _, _, a, base, _ in hits]
gidx = [int(b) * T + int(t) for t, b, _, _, _ in hits]
assert offsets == [24, 28] and gidx == [6, 7] and all(int(s) == BYTES for *_, s in hits)

fig = Figure(
    "02-sanitizer-catch", "Without the guard, two threads run past the end",
    f"Eight threads in two blocks of four, i from 0 to 7, above an array of {N} floats, {BYTES} bytes, standing for "
    "each of A, B and C. Threads 0 to 5 reach elements 0 to 5. Threads 6 and 7, thread 2 and thread 3 of block 1, "
    "reach past the end into a dashed red region at bytes 24 and 28, which memcheck reports as invalid "
    "__global__ reads by thread (2,0,0) and thread (3,0,0) in block (1,0,0).",
    subtitle=f"missing_guard.cu without its guard: N = {N}, blocks of {T}, a grid of {B} blocks; memcheck on the author's GTX 1650.",
    data_w=True)

X0, CW = 248, 80
TY, AY, CH = 176, 320, 48
fig.text(40, TY + 30, "threads, i", "label")
fig.text(40, AY + 30, "A, B and C", "label")
fig.text(40, AY + 54, f"{N} floats each", "note")
for b in range(B):
    x0, x1 = X0 + b * T * CW, X0 + (b + 1) * T * CW
    fig.brace(x0 + 4, x1 - 4, TY - 8, label=f"block {b}", side="above", kind="bracket")
for i in range(B * T):
    cell = Box(X0 + i * CW + 4, TY, CW - 8, CH)
    surplus = i >= N
    fig.fill(cell, "hazard-soft" if surplus else "compute-soft")
    fig.outline(cell, "hazard" if surplus else "compute-line")
    fig.text(cell.cx, TY + 30, f"i = {i}", "label", anchor="middle", color="hazard" if surplus else None)
    tgt = Box(X0 + i * CW + 4, AY, CW - 8, CH)
    if surplus:
        fig.outline(tgt, "hazard", width=1.5, dash="proj")
        fig.text(tgt.cx, AY + 30, f"[{i}]", "label", anchor="middle", color="hazard")
        fig.arrow((cell.cx, TY + CH + 4), (tgt.cx, AY - 4), color="hazard")
    else:
        fig.fill(tgt, "memory-soft")
        fig.outline(tgt, "memory-line")
        fig.text(tgt.cx, AY + 30, f"[{i}]", "label", anchor="middle")
        fig.arrow((cell.cx, TY + CH + 4), (tgt.cx, AY - 4))
end_x = X0 + N * CW
fig.text(X0 + 4, AY + CH + 24, "byte 0", "note")
fig.text(end_x - 4, AY + CH + 24, f"byte {BYTES}: the end", "note", anchor="end")
fig.text(end_x + 4, AY + CH + 24, "past the end", "note", color="hazard")
fig.text(40, 444, "memcheck:", "label")
fig.text(136, 444, "Invalid __global__ read of size 4 bytes", "code")
fig.text(136, 468, "by thread (2,0,0) and (3,0,0) in block (1,0,0), at bytes 24 and 28", "label")
fig.caption("Run alone, the program printed 6 of 6 and exited 0; memcheck stopped the kernel at the first read.")
fig.write()
