"""Post 05, hero: atomics on one bin are correct but take turns; privatization shortens the queue on the hot bin.

Run from the series root:  uv run python posts/05-histogram/diagrams/src/01-contention.py
Writes posts/05-histogram/diagrams/01-contention.svg.
Left, cuda_fig_design.md section 4's timeline form: eight threads (compute, orange labels) each issue one atomicAdd on
bins[128]; a turn is one illustrative time slot, the atomic is purple (sync), a thread waiting for its turn is red
(hazard: a contended atomic), and the bin (memory, green) steps from 1 to 8. Right, the counts on the hottest address
for 16,777,216 solid-color values, taken from snippets/histogram_model.py and asserted against the body.
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure, num  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its tables on import
    from histogram_model import N, GRID, rows  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
SOLID = rows["solid"]
assert (N, SOLID["global_hot"], SOLID["shared_hot"], SOLID["merge_hot"]) == (16777216, 16777216, 65536, 256)
assert SOLID["merge_hot"] == GRID and SOLID["equal"]
for line in ("histGlobal: atomics on the hottest bin       66,595  16,777,216",
             "histShared: hottest shared bin, 1 block         323      65,536",
             "histShared: merges on any global bin            256         256"):
    assert line in BODY, line
T = 8                                                    # threads drawn; one illustrative slot per atomic

fig = Figure(
    "01-contention", "Atomics on one bin take turns",
    "Left: eight threads, 0 to 7, each issue one atomicAdd on the same global bin, bins[128], over eight illustrative "
    "time slots. Thread k waits in red for k slots, adds in purple in slot k, and the bin's count steps from 1 to 8. "
    f"Right: a table of the model's counts for {num(N)} solid-color values: {num(SOLID['global_hot'])} atomics on the "
    f"hottest global bin with global atomics; with privatization, {num(SOLID['shared_hot'])} on one block's shared "
    f"bin and {num(SOLID['merge_hot'])} merges on the global bin.",
    subtitle="Left: one illustrative time slot per atomic. Right: counts derived by snippets/histogram_model.py.",
    data_w=True)

X, Y, CW, CH = 136, 136, 48, 28
fig.text(40, Y - 16, "8 threads, one atomicAdd each on bins[128]", "head")
fill = lambda i, j: "hazard-soft" if j < i else ("sync" if j == i else None)  # noqa: E731
g = fig.grid(X, Y, T, T, cell_w=CW, cell_h=CH, fill=fill)
for t in range(T):
    fig.text(40, Y + CH * t + CH // 2 + 5, f"thread {t}", "label")
by = Y + CH * T + 8
b = fig.grid(X, by, 1, T, cell_w=CW, cell_h=CH, values=[[k + 1 for k in range(T)]], font=16,
             fill=lambda i, j: "memory-soft")
fig.text(40, by + CH // 2 + 5, "bins[128]", "code")
with fig.data():
    for j in range(T):
        fig.text(X + CW * j + CW / 2, by + CH + 20, str(j + 1), "tick", anchor="middle", snap=False)
fig.text(40, by + CH + 20, "time slot", "tick")
fig.legend(X, by + CH + 48, [dict(color="sync", label="atomic add"), dict(color="hazard-soft", label="waits its turn"),
                             dict(color="memory-soft", label="count after the slot")], direction="row")

TX = 584
fig.text(TX, Y - 16, f"{num(N)} solid-color values", "head")
fig.table(TX, Y, [["kernel", "hottest address", "atomics"],
                  ["histGlobal", "bins[128]", num(SOLID["global_hot"])],
                  ["histShared", "s[128], per block", num(SOLID["shared_hot"])],
                  ["histShared", "bins[128], merges", num(SOLID["merge_hot"])]],
          [104, 152, 80], row_h=32, col_align=["start", "start", "end"], highlight_row={1: "hazard"})
fig.text(TX, Y + 4 * 32 + 28, "The global queue is one turn per element;", "note")
fig.text(TX, Y + 4 * 32 + 48, "privatized, 256 shorter queues and 256 merges.", "note")
fig.caption("Every increment counts, but on one address the threads are served one at a time.")
fig.write()
