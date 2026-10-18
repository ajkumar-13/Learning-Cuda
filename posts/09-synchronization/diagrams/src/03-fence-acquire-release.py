"""Post 09, figure 3: without fences a reader can see the flag before the data; a fence on each side (release,
acquire) makes a reader that sees the flag see the data.

Run from the series root:  uv run python posts/09-synchronization/diagrams/src/03-fence-acquire-release.py
Writes posts/09-synchronization/diagrams/03-fence-acquire-release.svg.
cuda_fig_design.md section 4 (two thread lanes and the memory they share on one time axis; a fence is purple,
the stale read red). The top timeline is one order the Programming Guide's weakly ordered model allows for a
plain data store followed by a relaxed flag update (section "Memory Fence Functions"); it is illustrative, not a
measurement, and the subtitle says so. The values 42 and the flag 1 are the post's section 4 example.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "data[7] = 42" in BODY and "is not necessarily the order in which the data is observed" in BODY

fig = Figure(
    "03-fence-acquire-release", "Fence before the flag, fence after it",
    "Two timelines of six time slots, each with a writer thread in block 0, the data and flag as block 1 sees "
    "them, and a reader thread in block 1. Top, no fences: the writer stores data = 42 in slot 1 and flag = 1 "
    "in slot 2, but block 1 sees the flag first (data 0, flag 1), so the reader sees flag 1 in slot 3 and reads "
    "the stale data 0 in slot 4, in red; data 42 arrives in slot 5. Bottom: the writer stores data = 42, fences "
    "(release) and stores flag = 1; the reader sees flag 1, fences (acquire) and reads data 42.",
    subtitle="Illustrative: one order the weakly ordered memory model allows; part 3 of sync.cu fences the writer's side.",
    data_w=True)

LX, GX, CW, CH = 40, 216, 112, 32
ROWS = ["writer, block 0", "block 1 sees", "reader, block 1"]


def cell(slot, row, y0, color, s, text_color=None, span=1, outline=None):
    b = Box(GX + CW * slot, y0 + CH * row, CW * span, CH)
    if color:
        fig.fill(b, color)
    fig.outline(b, outline or "rule")
    if s:
        fig.text(b.cx, b.y + 21, s, "label", anchor="middle", color=text_color)
    return b


def timeline(y0, heading, writer, seen, reader):
    fig.text(LX, y0 - 12, heading, "head")
    for r, name in enumerate(ROWS):
        fig.text(LX, y0 + CH * r + 21, name, "label")
    for slot in range(6):
        for row, ops in ((0, writer), (2, reader)):
            kind, s = ops.get(slot, (None, ""))
            color = {"op": "compute-soft", "fence": "sync-soft", "stale": "hazard-soft", "ok": "memory-soft",
                     None: None}[kind]
            cell(slot, row, y0, color, s, "hazard" if kind == "stale" else None)
        d, f, stale = seen[slot]
        cell(slot, 1, y0, "hazard-soft" if stale else "memory-soft", f"data {d}, flag {f}",
             "hazard" if stale else None)


Y1 = 140
timeline(Y1, "No fences",
         {0: ("op", "data = 42"), 1: ("op", "flag = 1")},
         [(0, 0, False), (0, 1, True), (0, 1, True), (0, 1, True), (42, 1, False), (42, 1, False)],
         {2: ("op", "sees flag 1"), 3: ("stale", "reads data 0")})
Y2 = 308
timeline(Y2, "With fences",
         {0: ("op", "data = 42"), 1: ("fence", "fence"), 2: ("op", "flag = 1")},
         [(0, 0, False), (42, 0, False), (42, 1, False), (42, 1, False), (42, 1, False), (42, 1, False)],
         {3: ("op", "sees flag 1"), 4: ("fence", "fence"), 5: ("ok", "reads data 42")})
fig.brace(GX + CW * 1, GX + CW * 3, Y2 - 4, "release", side="above", color="sync")
with fig.data():
    for slot in range(6):
        fig.text(GX + CW * slot + CW / 2, Y2 + 3 * CH + 20, str(slot + 1), "tick", anchor="middle", snap=False)
fig.text(LX, Y2 + 3 * CH + 20, "time slot", "tick")
fig.brace(GX + CW * 3, GX + CW * 6, Y2 + 3 * CH + 28, "acquire", side="below", color="sync")
fig.caption("A reader that sees the flag is sure to see the data only when both sides order their accesses.")
fig.write()
