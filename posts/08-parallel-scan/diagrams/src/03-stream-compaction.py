"""Post 08, section 7: stream compaction is a predicate, an exclusive scan of it, and a scatter.

Run from the series root:  uv run python posts/08-parallel-scan/diagrams/src/03-stream-compaction.py
Writes posts/08-parallel-scan/diagrams/03-stream-compaction.svg.
cuda_fig_design.md: data at rest is memory (green); an element the filter drops is a neutral cell; the scatter,
one write per kept element by its own thread, is compute (orange). Every value is printed by
snippets/scan_model.py (DATA, compact), asserted against the post.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure, num  # noqa: E402
from scan_model import DATA, compact  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
PRED, POS, OUT = compact(DATA)
assert DATA == [3, -1, 7, 0, -2, 4, 1, -5, 6] and PRED == [1, 0, 1, 0, 0, 1, 1, 0, 1]
assert POS == [0, 1, 1, 2, 2, 2, 3, 4, 4] and OUT == [3, 7, 4, 1, 6]
assert "exclusive scan  :  0  1  1  2  2  2  3  4  4" in BODY and "scatter         : [3, 7, 4, 1, 6]" in BODY
n, kept = len(DATA), len(OUT)
KEEP = [i for i in range(n) if PRED[i]]

fig = Figure(
    "03-stream-compaction", "An exclusive scan tells each survivor where to go",
    f"Four rows for {n} values. Input: {' '.join(num(v) for v in DATA)}. Predicate, 1 for a positive value: "
    f"{' '.join(map(str, PRED))}. Exclusive scan of the predicate: {' '.join(map(str, POS))}. Orange arrows "
    f"scatter each kept input to the output index its scan gives, filling {' '.join(map(str, OUT))}: {kept} of "
    f"{n} kept, in order, with no gaps. Dropped elements are drawn neutral.",
    subtitle=f"Keep the positive values of {n} numbers; every value is printed by scan_model.py.",
    height=720, data_w=True)

X, CW, CH = 216, 64, 48
Y_IN, Y_PRED, Y_SCAN, Y_OUT = 160, 272, 384, 560
keepfill = lambda j: "memory-soft" if PRED[j] else "neutral-soft"  # noqa: E731
g_in = fig.strip(X, Y_IN, n, cell_w=CW, cell_h=CH, values=[num(v) for v in DATA], font=18, fill=keepfill)
g_in.col_labels({j: str(j) for j in range(n)}, side="above")
g_pred = fig.strip(X, Y_PRED, n, cell_w=CW, cell_h=CH, values=PRED, font=18, fill=keepfill)
g_scan = fig.strip(X, Y_SCAN, n, cell_w=CW, cell_h=CH, values=POS, font=18, fill=keepfill)
X_OUT = X + (n - kept) * CW // 2
g_out = fig.strip(X_OUT, Y_OUT, kept, cell_w=CW, cell_h=CH, values=OUT, font=18,
                  fill=lambda j: "memory-soft")
g_out.col_labels({j: str(j) for j in range(kept)}, side="below")
g_out.window(0, kept - 1, color="memory")

for label, y in (("input", Y_IN), ("predicate", Y_PRED), ("exclusive scan", Y_SCAN), ("output", Y_OUT)):
    fig.text(40, y + 30, label, "label")
fig.text(40, Y_IN - 8, "index", "tick")
fig.text(40, Y_PRED + 58 + 16, "1 keeps a value > 0", "note")
fig.text(40, Y_SCAN + 58 + 16, "kept before it = its index", "note")

for j in range(n):
    a, b, c = g_in.cell(0, j), g_pred.cell(0, j), g_scan.cell(0, j)
    fig.arrow((a.cx, a.bottom + 4), (b.cx, b.y - 4), width=1)
    fig.arrow((b.cx, b.bottom + 4), (c.cx, c.y - 4), width=1)
for i in KEEP:
    src, dst = g_scan.cell(0, i), g_out.cell(0, POS[i])
    fig.arrow((src.cx, src.bottom + 4), (dst.cx, dst.y - 4), color="compute", width=1.5)

fig.legend(40, 656, [dict(color="memory", label="kept"), dict(color="neutral-soft", label="dropped", mark="swatch"),
                     dict(color="compute", label="scatter: out[scan[i]] = in[i]", mark="line")], direction="row")
fig.caption(f"The scan's total, {POS[-1] + PRED[-1]}, is the output's length; the kept values stay in order.")
fig.write()
