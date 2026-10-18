"""Post 06, hero: a warp's row read costs one 128-byte line of four sectors; its column write costs 32 lines and sectors.

Run from the series root:  uv run python posts/06-matrix-transpose/diagrams/src/01-coalesced-vs-strided.py
Writes posts/06-matrix-transpose/diagrams/01-coalesced-vs-strided.svg.
cuda_fig_design.md section 3, lanes against addresses: lanes are compute (orange), the words of memory a lane uses are
memory (green), the rest of a sector that is moved but not used is hazard (red, a strided transaction's waste), and a
line's untouched words are left empty. Left, the 32 lanes and the one line their row read touches, with its four
sectors bracketed. Right, 8 of the 32 lanes of the column write, each with its own line, one word used and seven moved
for nothing. Every count comes from snippets/transpose_model.py (cost), asserted against the post.
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure, num  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):          # the model prints its blocks on import
    from transpose_model import LINE, SECTOR, WARP, WORD, cost  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROW, COL = cost["row"], cost["column"]
assert ROW[:3] == (1, 4, 128) and COL[:3] == (32, 32, 1024) and (ROW[3], COL[3]) == (1.0, 0.125)
assert "  a row (coalesced)                 1                4           128 100.0%" in BODY
assert "  a column (strided)               32               32         1,024  12.5%" in BODY
WORDS_PER_LINE, WORDS_PER_SECTOR = LINE // WORD, SECTOR // WORD    # 32 and 8
SHOWN = 8                                                           # lanes drawn on the right

fig = Figure(
    "01-coalesced-vs-strided", "A row costs a warp one line; a column costs 32",
    f"Left: the {WARP} lanes of a warp read a row of A, 32 consecutive floats, which fill one {LINE}-byte line of "
    f"{ROW[1]} sectors of {SECTOR} bytes, every word used: {ROW[0]} line, {ROW[1]} sectors, {ROW[2]} bytes moved, 100% "
    f"used. Right: lanes 0 to {SHOWN - 1} of the same warp write a column of B; each lane's float lands in its own line, "
    f"and its sector moves 7 other words for nothing. For all {WARP} lanes: {COL[0]} lines, {COL[1]} sectors, "
    f"{num(COL[2])} bytes moved for 128 used, 12.5%.",
    subtitle="One warp, 4-byte floats, a 4,096 × 4,096 row-major matrix; counts derived by snippets/transpose_model.py.",
    height=720, data_w=True)

used, wasted = "memory", "hazard-soft"

# left: the row read
LX, CW = 40, 12
fig.text(LX, 136, "Read a row of A: coalesced", "head")
lanes = fig.grid(LX, 168, 1, WARP, cell_w=CW, cell_h=24, fill=lambda i, j: "compute-soft", lines_on_top=True)
fig.text(LX, 160, "lanes 0 to 31", "tick")
line = fig.grid(LX, 264, 1, WORDS_PER_LINE, cell_w=CW, cell_h=24, fill=lambda i, j: used, lines_on_top=True)
fig.arrow((LX + WARP * CW // 2, 196), (LX + WARP * CW // 2, 260), label="one access")
for s in range(ROW[1]):
    x0 = LX + s * WORDS_PER_SECTOR * CW
    fig.brace(x0 + 2, x0 + WORDS_PER_SECTOR * CW - 2, 296, label=f"sector {s}", side="below", kind="bracket")
fig.text(LX, 364, f"{ROW[0]} line of {LINE} bytes, {ROW[1]} sectors of {SECTOR} bytes", "label")
fig.text(LX, 388, f"{ROW[2]} bytes moved, 100% used", "label", color="memory")
fig.text(LX, 448, "Bytes moved for the same 32 floats", "head")
fig.bar_chart(Box(LX, 464, 408, 128), [("row read", ROW[2], used), ("column write", COL[2], "hazard")],
              0, 1024, [0, 256, 512, 768, 1024], label_w=104, value_w=56, fmt=lambda v: num(v), axis_label="bytes")

# right: the column write, one line per lane
RX = 504
fig.text(RX, 136, "Write a column of B: strided", "head")
SX, SW, RH = 600, 8, 40
for r in range(SHOWN):
    y = 168 + RH * r
    fig.grid(RX + 56, y, 1, 1, cell_w=24, cell_h=24, fill=lambda i, j: "compute-soft")
    fig.text(RX, y + 17, f"lane {r}", "label")
    fig.arrow((RX + 84, y + 12), (SX - 4, y + 12))
    fig.grid(SX, y, 1, WORDS_PER_LINE, cell_w=SW, cell_h=24,
             fill=lambda i, j: used if j == 0 else (wasted if j < WORDS_PER_SECTOR else None), lines_on_top=True)
    fig.text(SX + WORDS_PER_LINE * SW + 12, y + 17, f"line {r}", "label")
ny = 168 + RH * SHOWN + 20
fig.text(RX, ny, f"lanes {SHOWN} to {WARP - 1}: {WARP - SHOWN} more lines", "note")
fig.text(RX, ny + 36, f"{COL[0]} lines, {COL[1]} sectors", "label")
fig.text(RX, ny + 60, f"{num(COL[2])} bytes moved, 128 used: 12.5%", "label", color="hazard")

fig.legend(40, 624, [dict(color="compute-soft", label="lane"), dict(color=used, label="word a lane uses"),
                     dict(color=wasted, label="moved in its sector, unused"),
                     dict(color="rule", label="not moved", mark="outline")], direction="row")
fig.caption("The column write moves 8 times the bytes of the row read for the same 32 floats.")
fig.write()
