"""Post 15, section 3: the commits and waits of matmulPipelined over five tiles, one thread's view.

Run from the series root:  uv run python posts/15-async-copy-pipelining/diagrams/src/03-commit-groups.py
Writes posts/15-async-copy-pipelining/diagrams/03-commit-groups.svg.
cuda_fig_design.md section 4 (a pipeline: the buffers green, the compute orange, a cp.async wait purple), drawn as a
ruled table. The table is simulated here from the kernel's control flow (one commit per issued tile, the wait count
1 while a prefetch was issued and 0 on the last tile) and the Programming Guide's rule for __pipeline_wait_prior(N):
with commits numbered 0 .. L, it waits for every group up to and including L - N. The script asserts that the group
of the tile computed in each iteration is complete after the wait, and that only the newest group may be in flight.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "__pipeline_wait_prior(t + 1 < numTiles ? 1 : 0);" in BODY
assert 'waits for "completion of batches at least up to and including L-N"' in BODY

N_TILES = 5
committed, buffer_of = [], {}
cols = []                                   # per column: (commit, buffer, wait n, in flight, compute buffer, tile)
# prologue
committed.append(0); buffer_of[0] = 0
cols.append(("0", "0", "none", "0", "none", "none"))
for t in range(N_TILES):
    issued = t + 1 < N_TILES
    if issued:
        committed.append(t + 1); buffer_of[t + 1] = (t + 1) & 1
    n = 1 if issued else 0
    last = len(committed) - 1
    complete = [g for g in committed if g <= last - n]
    in_flight = [g for g in committed if g > last - n]
    assert t in complete and buffer_of[t] == (t & 1)
    assert len(in_flight) == (1 if issued else 0)
    cols.append((str(t + 1) if issued else "none", str((t + 1) & 1) if issued else "none", str(n),
                 ", ".join(map(str, in_flight)) or "none", str(t & 1), str(t)))

header = ["", "prologue"] + [f"t = {t}" for t in range(N_TILES)]
labels = ["group committed", "into buffer", "wait_prior count", "group still in flight", "compute from buffer",
          "tile computed"]
rows = [header] + [[labels[i]] + [c[i] for c in cols] for i in range(len(labels))]

fig = Figure(
    "03-commit-groups", "Waiting for all but the newest group keeps one tile ahead",
    "A table of matmulPipelined over five tiles, one thread's view. Columns: the prologue and iterations t = 0 to 4. "
    "Group committed: 0, 1, 2, 3, 4, none; into buffer: 0, 1, 0, 1, 0, none. Wait count: none in the prologue, 1 at "
    "t = 0 to 3, 0 at t = 4. Group still in flight after the wait: 0 after the prologue, then 1, 2, 3, 4, none. "
    "Compute from buffer 0, 1, 0, 1, 0, and the tile computed is t.",
    subtitle="Five tiles; __pipeline_wait_prior(n) completes every group but the newest n (Programming Guide).",
    data_w=True)

col_w = [208] + [112] * (N_TILES + 1)
tb = fig.table(40, 128, rows, col_w, header=True, row_h=40,
               col_align=["start"] + ["middle"] * (N_TILES + 1),
               highlight_row={1: "memory", 2: "memory", 3: "sync", 5: "compute", 6: "compute"})
fig.legend(40, 440, [dict(color="memory-soft", label="copies issued"),
                     dict(color="sync-soft", label="the wait"),
                     dict(color="compute-soft", label="the math")], direction="row")
fig.caption("Each iteration computes a tile whose group is complete while the next tile's group runs on.")
fig.write()
