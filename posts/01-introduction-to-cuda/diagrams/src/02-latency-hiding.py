"""Post 01, section 2: one warp leaves the scheduler idle while its load is in flight; four warps fill the wait.

Run from the series root:  uv run python posts/01-introduction-to-cuda/diagrams/src/02-latency-hiding.py
Writes posts/01-introduction-to-cuda/diagrams/02-latency-hiding.svg.
Every cell is derived by snippets/thread_model.py: schedule(n_warps, cycles) says which warp issues on each cycle,
and busy_share() the share of busy cycles. The timings (4 instructions, then a 12-cycle wait) are the model's
illustrative defaults, so the subtitle says "illustrative" (cuda_fig_design.md section 4: compute is orange, a
load in flight is green, the waste, an idle scheduler cycle, is red).
"""
import inspect
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its tables on import
    from thread_model import busy_share, schedule  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
DEF = {k: v.default for k, v in inspect.signature(schedule).parameters.items() if v.default is not inspect._empty}
WORK, WAIT = DEF["work"], DEF["wait"]
CYCLES = 32
assert (WORK, WAIT) == (4, 12)
assert round(busy_share(1) * 100) == 25 and round(busy_share(4) * 100) == 100
for phrase in ("a warp issues 4 instructions and then waits 12 cycles", "busy for 4 cycles out of every 16",
               "Four warps fill the 12-cycle wait exactly"):
    assert phrase in BODY, phrase


def states(n_warps):
    """Per warp and cycle: 'issue', 'wait' (its load in flight) or None (ready, queued); and the issue row."""
    issued = schedule(n_warps, CYCLES)
    rows = [[None] * CYCLES for _ in range(n_warps)]
    for w in range(n_warps):
        count = 0
        for t, who in enumerate(issued):
            if who == w:
                rows[w][t] = "issue"
                count += 1
                if count % WORK == 0:                         # the load goes out after the WORK-th instruction
                    for u in range(t + 1, min(t + 1 + WAIT, CYCLES)):
                        rows[w][u] = "wait"
    return rows, issued


fig = Figure(
    "02-latency-hiding", "Enough ready warps keep the scheduler busy",
    "Two timelines of 32 cycles on one SM scheduler, with illustrative timings from snippets/thread_model.py: a "
    "warp issues 4 instructions in orange and then waits 12 cycles for its load, in green. With one warp, the "
    "scheduler row issues on 8 of the 32 cycles and is idle, in red, on the other 24: 25 percent busy. With four "
    "warps, each issues while the other three wait, and the scheduler row issues on all 32 cycles, numbered by the "
    "warp it serves: 100 percent busy.",
    subtitle=f"Derived by snippets/thread_model.py; illustrative timings: {WORK} instructions, then a {WAIT}-cycle wait.",
    data_w=True)

X, CELL, ROW = 168, 23, 32                     # 32 cycles of 23 = 736, ending at 904
FILL = {"issue": "compute", "wait": "memory-soft", None: None}       # None: ready, queued behind another warp


def panel(top, n_warps):
    rows, issued = states(n_warps)
    busy = sum(w is not None for w in issued)
    fig.text(40, top - 16, f"{n_warps} warp{'s' if n_warps > 1 else ''}: scheduler busy {busy} of {CYCLES} "
             f"cycles, {round(100 * busy / CYCLES)}%", "head")
    with fig.data():
        fig.grid(X, top, n_warps, CYCLES, cell_w=CELL, cell_h=ROW, fill=lambda i, j: FILL[rows[i][j]],
                 lines_on_top=True)
    for w in range(n_warps):
        fig.text(40, top + ROW * w + 21, f"warp {w}", "label")
    s_top = top + ROW * n_warps + 8
    with fig.data():
        fig.grid(X, s_top, 1, CYCLES, cell_w=CELL, cell_h=ROW, lines_on_top=True,
                 fill=lambda i, j: "compute-soft" if issued[j] is not None else "hazard-soft")
        for j, w in enumerate(issued):
            if w is not None:
                fig.text(X + j * CELL + CELL / 2, s_top + 21, str(w), "tick", anchor="middle", snap=False)
    fig.text(40, s_top + 21, "scheduler", "label")
    return s_top + ROW


bottom = panel(136, 1)
bottom = panel(bottom + 48, 4)
with fig.data():
    for c in range(0, CYCLES + 1, 8):
        fig.text(X + c * CELL, bottom + 20, str(c), "tick", anchor="middle", snap=False)
fig.text(40, bottom + 20, "cycle", "tick")
fig.legend(X, bottom + 52, [dict(color="compute", label="warp issues"),
                            dict(color="memory-soft", label="its load in flight"),
                            dict(color="rule", label="ready, queued", mark="outline"),
                            dict(color="hazard-soft", label="scheduler idle")], direction="row")
fig.caption("A stall costs nothing while another warp is ready to issue.")
fig.write()
