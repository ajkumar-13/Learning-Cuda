"""Post 09, the hero: a plain counter += 1 loses an update; atomicAdd makes the second thread wait.

Run from the series root:  uv run python posts/09-synchronization/diagrams/src/01-race-condition.py
Writes posts/09-synchronization/diagrams/01-race-condition.svg.
cuda_fig_design.md section 4 (a race is two thread lanes reading and writing one green memory cell on one
time axis, the lost update red; the fixed version repeats the drawing with the purple atomic). The plain
schedule and its values (both threads read 7, both write 8, the counter ends at 8 instead of 9) come from
snippets/race_model.py's two_threads(7), asserted against the post's quoted trace; the atomic schedule
replays the same two threads with each read, add and write made one step.
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
    from race_model import simulate, two_threads  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
EVENTS, FINAL = two_threads(7)
assert EVENTS == [('R', 0), ('R', 1), ('I', 0), ('I', 1), ('W', 0), ('W', 1)] and FINAL == 8
ATOMIC = [('R', 0), ('I', 0), ('W', 0), ('R', 1), ('I', 1), ('W', 1)]
assert simulate(ATOMIC, start=7, threads=2) == 9
assert "both read 7" in BODY and "counter ends at 8, not 9" in BODY


def counter_after(events, start):
    """The shared counter after each micro-op, replayed as the model does."""
    reg, c, out = [0, 0], start, []
    for op, i in events:
        if op == 'R':
            reg[i] = c
        elif op == 'I':
            reg[i] += 1
        else:
            c = reg[i]
        out.append(c)
    return out


PLAIN_C = counter_after(EVENTS, 7)
ATOM_C = counter_after(ATOMIC, 7)
assert PLAIN_C == [7, 7, 7, 7, 8, 8] and ATOM_C == [7, 7, 8, 8, 8, 9]

fig = Figure(
    "01-race-condition", "Two threads add one each; the counter gains one",
    "Two timelines of six time slots, each with rows for thread 0, the shared counter and thread 1. Top, a plain "
    "counter += 1: thread 0 reads 7 in slot 1, thread 1 reads 7 in slot 2, both compute 8, thread 0 writes 8 in slot "
    "5 and thread 1 writes 8 in slot 6; the counter reads 7, 7, 7, 7, 8, 8 and ends at 8, not 9, the second write "
    "in red as the lost update. Bottom, atomicAdd: thread 0's indivisible add takes the counter from 7 to 8 in slots "
    "1 to 3 while thread 1 waits, then thread 1's takes it to 9 in slots 4 to 6.",
    subtitle="Two threads, one increment each, the counter starting at 7; the schedule of snippets/race_model.py.",
    data_w=True)

LX, GX, CW, CH = 40, 160, 120, 32
ROWS = ["thread 0", "counter", "thread 1"]


def cell(x, y, w, h, color, s, text_color=None):
    b = Box(x, y, w, h)
    if color:
        fig.fill(b, color)
    fig.outline(b, "rule")
    if s:
        fig.text(b.cx, b.y + h / 2 + 5, s, "label", anchor="middle", color=text_color)
    return b


def lanes(y0, heading):
    fig.text(LX, y0 - 12, heading, "head")
    for r, name in enumerate(ROWS):
        fig.text(LX, y0 + CH * r + 21, name, "code" if name == "counter" else "label")


# top: the plain read-modify-write
Y1 = 140
lanes(Y1, "Plain read, add, write")
words = {('R', 0): "reads 7", ('R', 1): "reads 7", ('I', 0): "computes 8", ('I', 1): "computes 8",
         ('W', 0): "writes 8", ('W', 1): "writes 8"}
for slot, ev in enumerate(EVENTS):
    op, t = ev
    x = GX + CW * slot
    row = 0 if t == 0 else 2
    lost = ev == ('W', 1)
    cell(x, Y1 + CH * row, CW, CH, "hazard-soft" if lost else "compute-soft", words[ev],
         "hazard" if lost else None)
    cell(x, Y1 + CH * (2 - row), CW, CH, None, "")
    last = slot == len(EVENTS) - 1
    cell(x, Y1 + CH, CW, CH, "hazard-soft" if last else "memory-soft",
         f"{PLAIN_C[slot]}, not 9" if last else str(PLAIN_C[slot]), "hazard" if last else None)

# bottom: atomicAdd
Y2 = 308
lanes(Y2, "atomicAdd: one indivisible step each")
cell(GX, Y2, CW * 3, CH, "sync-soft", "atomicAdd: 7 → 8")
cell(GX + CW * 3, Y2, CW * 3, CH, None, "")
cell(GX, Y2 + 2 * CH, CW * 3, CH, "hazard-soft", "waits its turn")
cell(GX + CW * 3, Y2 + 2 * CH, CW * 3, CH, "sync-soft", "atomicAdd: 8 → 9")
for slot in range(6):
    cell(GX + CW * slot, Y2 + CH, CW, CH, "memory-soft", str(ATOM_C[slot]))

with fig.data():
    for slot in range(6):
        fig.text(GX + CW * slot + CW / 2, Y2 + 3 * CH + 20, str(slot + 1), "tick", anchor="middle", snap=False)
fig.text(LX, Y2 + 3 * CH + 20, "time slot", "tick")
fig.legend(GX, 452, [dict(color="compute-soft", label="a thread's step"),
                     dict(color="memory-soft", label="counter after the slot"),
                     dict(color="sync-soft", label="atomic"),
                     dict(color="hazard-soft", label="lost update or wait")], direction="row")
fig.caption("Two increments, one counted: the second write lands on the first.")
fig.write()
