"""Post 08, the hero: the serial scan is a chain of dependent additions; Hillis-Steele reaches the same totals in
log2 N parallel steps.

Run from the series root:  uv run python posts/08-parallel-scan/diagrams/src/01-scan-dependency.py
Writes posts/08-parallel-scan/diagrams/01-scan-dependency.svg.
cuda_fig_design.md section 3 (a scan is a lattice of levels drawn top to bottom): the inputs and the finished
running totals are memory (green), the partial sums written by threads at each Hillis-Steele step compute
(orange), and the dependency chain of the serial loop hazard (red), since it is what serializes the loop. Every
value comes from snippets/scan_model.py (EIGHT, serial_inclusive, hillis_steele), asserted against the post.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Figure  # noqa: E402
from scan_model import EIGHT, hillis_steele, serial_inclusive  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
INC = serial_inclusive(EIGHT)
HS = hillis_steele(EIGHT)
assert EIGHT == [3, 1, 7, 0, 4, 1, 6, 3] and INC == [3, 4, 11, 11, 15, 16, 22, 25]
assert [s for s, _ in HS[1:]] == [1, 2, 4] and HS[-1][1] == INC
assert "inclusive       :  3  4 11 11 15 16 22 25" in BODY and "stride 4, 4 adds :  3  4 11 11 15 16 22 25" in BODY
CHAIN = len(EIGHT) - 1
STEPS = len(HS) - 1
assert (CHAIN, STEPS) == (7, 3)

vals = " ".join(map(str, EIGHT))
fig = Figure(
    "01-scan-dependency", "A running total looks serial; it needs only 3 steps",
    f"Two panels on the eight values {vals}. Left, the serial loop: each running total adds its input to the "
    f"total before it, so the totals 3 4 11 11 15 16 22 25 form a chain of {CHAIN} dependent additions, drawn as "
    f"red arrows numbered 1 to {CHAIN}. Right, Hillis-Steele: at stride 1, 2 and 4 every element adds the "
    f"element that many places back, all at once; the rows read 3 4 8 7 4 5 7 9, then 3 4 11 11 12 12 11 14, "
    f"then the same totals, in {STEPS} parallel steps.",
    subtitle="The post's eight values; every number is printed by scan_model.py.",
    height=720, data_w=True)

left, right = fig.row(2, y=104, h=528)
CW, CH = 44, 40
YS = [160 + 120 * k for k in range(4)]

# left: the serial loop
body = fig.panel(left, f"Serial loop: {CHAIN} dependent steps")
X0 = body.x + 40
inp = fig.strip(X0, YS[0], 8, cell_w=CW, cell_h=CH, values=EIGHT, font=16, fill=lambda j: "memory-soft")
out = fig.strip(X0, YS[3], 8, cell_w=CW, cell_h=CH, values=INC, font=16, fill=lambda j: "memory-soft")
fig.text(body.x, YS[0] + 26, "in", "label")
fig.text(body.x, YS[3] + 26, "out", "label")
fig.text(X0, 316, "for i = 1 to 7:", "code")
fig.text(X0 + 24, 344, "out[i] = out[i - 1] + in[i]", "code")
fig.text(X0, 392, "addition i needs the result of addition i − 1,", "note")
fig.text(X0, 416, "so the 7 additions run one after another", "note")
for j in range(1, 8):
    a, b = out.cell(0, j - 1), out.cell(0, j)
    yb = a.bottom + 24
    fig.arrow((a.cx + 4, a.bottom + 4), (b.cx - 4, b.bottom + 4), via=[(a.cx + 4, yb), (b.cx - 4, yb)],
              color="hazard", width=1.5)
    fig.text((a.cx + b.cx) / 2, yb + 20, str(j), "tick", anchor="middle", color="hazard")

# right: Hillis-Steele
body = fig.panel(right, f"Hillis-Steele: {STEPS} parallel steps")
X1 = body.x + 72
rows = []
for k, y in enumerate(YS):
    values = EIGHT if k == 0 else HS[k][1]
    color = "memory" if k in (0, STEPS) else "compute"
    rows.append(fig.strip(X1, y, 8, cell_w=CW, cell_h=CH, values=values, font=16,
                          fill=lambda j, c=color: c + "-soft"))
    fig.text(body.x, y + 26, "in" if k == 0 else f"stride {HS[k][0]}", "label")
for k in range(1, STEPS + 1):
    stride = HS[k][0]
    above, below = rows[k - 1], rows[k]
    for t in range(8):
        dst, src = below.cell(0, t), above.cell(0, t)
        fig.arrow((src.cx + 8, src.bottom + 4), (dst.cx + 8, dst.y - 4), width=1)
        if t >= stride:
            s2 = above.cell(0, t - stride)
            fig.arrow((s2.cx - 8, s2.bottom + 4), (dst.cx - 8, dst.y - 4), color="compute", width=1)

fig.legend(left.x, 644, [dict(color="hazard", label="an addition that waits for the one before", mark="line"),
                         dict(color="compute", label="element t adds element t − stride", mark="line"),
                         dict(color="arrow", label="element's own value carried down", mark="line")],
           direction="row")
fig.caption(f"Same {CHAIN + 1} totals: {CHAIN} additions in a chain, or {STEPS} steps in which every element adds at once.")
fig.write()
