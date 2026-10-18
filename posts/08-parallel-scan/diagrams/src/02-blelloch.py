"""Post 08, section 3: the Blelloch scan's up-sweep builds partial sums, the root is cleared, and the down-sweep
turns them into the exclusive scan.

Run from the series root:  uv run python posts/08-parallel-scan/diagrams/src/02-blelloch.py
Writes posts/08-parallel-scan/diagrams/02-blelloch.svg.
cuda_fig_design.md section 3 (a scan as levels drawn top to bottom): values at rest are memory (green), the
elements a level writes compute (orange), the cleared root purple (the step that turns the reduction into a
scan, as the addendum asks), the finished exclusive scan green and outlined. The levels are re-derived here with
the indices of scan.cu (ai = offset * (2t + 1) - 1, bi = offset * (2t + 2) - 1); the three rows the model prints
(after the up-sweep, root cleared, after the down-sweep) are asserted equal to snippets/scan_model.py's blelloch().
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
from scan_model import EIGHT, blelloch, hillis_steele, serial_exclusive  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
N = len(EIGHT)


def levels(x):
    """Every level of both sweeps: (label, values, {written index: [source indices]})."""
    s, out, offset = list(x), [], 1
    d = N >> 1
    while d > 0:
        old, writes = list(s), {}
        for t in range(d):
            ai, bi = offset * (2 * t + 1) - 1, offset * (2 * t + 2) - 1
            s[bi] = old[bi] + old[ai]
            writes[bi] = [ai, bi]
        out.append((f"offset {offset}", list(s), writes))
        offset *= 2
        d >>= 1
    s[N - 1] = 0
    out.append(("root = 0", list(s), {}))
    d = 1
    while d < N:
        offset >>= 1
        old, writes = list(s), {}
        for t in range(d):
            ai, bi = offset * (2 * t + 1) - 1, offset * (2 * t + 2) - 1
            s[ai], s[bi] = old[bi], old[bi] + old[ai]
            writes[ai], writes[bi] = [bi], [ai, bi]
        out.append((f"offset {offset}", list(s), writes))
        d *= 2
    return out


LV = levels(EIGHT)
UP, CLEARED, DOWN = blelloch(EIGHT)
assert LV[2][1] == UP == [3, 4, 7, 11, 4, 5, 6, 25] and LV[3][1] == CLEARED and LV[-1][1] == DOWN
assert DOWN == serial_exclusive(EIGHT) == [0, 3, 4, 11, 11, 15, 16, 22]
assert "after up-sweep  :  3  4  7 11  4  5  6 25" in BODY and "after down-sweep:  0  3  4 11 11 15 16 22" in BODY
ADDS = sum(len(w) for _, _, w in LV[:3]) + sum(1 for _, _, w in LV[4:] for k, src in w.items() if len(src) == 2)
assert ADDS == 2 * (N - 1) == 14

HS_USEFUL = sum(N - st for st, _ in hillis_steele(EIGHT)[1:])
assert HS_USEFUL == 17
row = lambda k: " ".join(map(str, LV[k][1]))  # noqa: E731
fig = Figure(
    "02-blelloch", "Blelloch: sum up the tree, then pass prefixes down",
    "Two panels on the values 3 1 7 0 4 1 6 3. Up-sweep, top to bottom: at offset 1 the right element of each "
    f"pair adds the left, giving {row(0)}; at offset 2, {row(1)}; at offset 4, {row(2)}, the total last. "
    f"Down-sweep: the root is cleared to 0 (purple), giving {row(3)}; at offsets 4, 2 and 1 each left child takes "
    "its parent's value and each right child the parent plus the old left value, giving "
    f"{row(4)}, then {row(5)}, then the exclusive scan {row(6)}. Orange cells are written at that level, and the total and the finished scan are outlined green; "
    f"{ADDS} additions in all.",
    subtitle="The post's eight values, with the tree indices of scan.cu; the end rows are printed by scan_model.py.",
    height=720, data_w=True)

left, right = fig.row(2, y=104, h=528)
CW, CH = 42, 40
YS = [160 + 120 * k for k in range(4)]


def draw(panel, heading, rows):
    body = fig.panel(panel, heading)
    x = body.x + 72
    strips = []
    for k, (label, values, writes, y) in enumerate(rows):
        def fill(j, k=k, writes=writes, label=label):
            if label == "root = 0" and j == N - 1:
                return "sync-soft"
            return "compute-soft" if j in writes else "memory-soft"
        g = fig.strip(x, y, N, cell_w=CW, cell_h=CH, values=values, font=16, fill=fill)
        strips.append(g)
        fig.text(body.x, y + 26, label, "label")
        if k > 0:
            above = strips[k - 1]
            for dst_i, srcs in writes.items():
                dst = g.cell(0, dst_i)
                for src_i in srcs:
                    src = above.cell(0, src_i)
                    dx = 0 if src_i == dst_i else (-8 if src_i < dst_i else 8)
                    fig.arrow((src.cx + (8 if src_i == dst_i and len(srcs) == 2 else 0), src.bottom + 4),
                              (dst.cx + dx, dst.y - 4), color="compute", width=1)
    return strips


up_rows = [("input", list(EIGHT), {}, YS[0])] + [(lab, v, w, y) for (lab, v, w), y in zip(LV[:3], YS[1:])]
ups = draw(left, "Up-sweep: a reduction tree in place", up_rows)
ups[-1].window(N - 1, N - 1, color="memory")
tot = ups[-1].cell(0, N - 1)
fig.text(tot.cx, tot.bottom + 28, "the total", "label", anchor="middle", color="memory")

down_rows = [(LV[3][0], LV[3][1], {}, YS[0])] + [(lab, v, w, y) for (lab, v, w), y in zip(LV[4:], YS[1:])]
downs = draw(right, "Down-sweep: prefixes passed down", down_rows)
downs[-1].window(0, N - 1, color="memory")
fig.text(downs[-1].box.cx, downs[-1].box.bottom + 28, "the exclusive scan", "label", anchor="middle", color="memory")

fig.legend(left.x, 644, [dict(color="compute", label="written at this level"),
                         dict(color="memory", label="unchanged at this level"),
                         dict(color="memory", label="result", mark="outline"),
                         dict(color="sync", label="root cleared to 0")], direction="row")
fig.caption(f"Each sweep has log₂ 8 = 3 levels and {N - 1} additions, {ADDS} in all, against {HS_USEFUL} for Hillis-Steele.")
fig.write()
