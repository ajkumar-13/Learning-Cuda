"""Post 16, section 5: the naive and the stable softmax of the row 2, 3, 1000, 1001, value by value.

Run from the series root:  uv run python posts/16-numerical-precision/diagrams/src/04-stable-softmax.py
Writes posts/16-numerical-precision/diagrams/04-stable-softmax.svg.
Every cell is computed here in FP32 with numpy, as precision_model.py's block (d) does, and the results are
asserted against the model's and the GPU's printed rows in index.md. cuda_fig_design.md section 0: inf and nan
are hazard red; the finite probabilities are data at rest, memory green.
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure, MINUS  # noqa: E402

np.seterr(over="ignore", invalid="ignore", under="ignore")
F32 = np.float32
x = np.array([2, 3, 1000, 1001], dtype=F32)
M = x.max()
e_naive = np.exp(x)
p_naive = e_naive / e_naive.sum()
shift = x - M
e_st = np.exp(shift)
S = e_st.sum()
p_st = e_st / S

BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "naive  exp(x) / sum      = [ 0.  0. nan nan]" in BODY
assert "stable exp(x - max) / sum = [0.     0.     0.2689 0.7311]" in BODY
assert "(d) naive  softmax: 0.0000 0.0000 nan nan" in BODY
assert "    stable softmax: 0.0000 0.0000 0.2689 0.7311" in BODY
assert ["%.4f" % v for v in p_st] == ["0.0000", "0.0000", "0.2689", "0.7311"]
assert np.isinf(e_naive.sum()) and np.isnan(p_naive[2:]).all() and float(M) == 1001


def fmt(v):
    v = float(v)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf"
    if v == 0:
        return "0"
    if v.is_integer():
        return f"{int(v):,}".replace("-", MINUS)
    return f"{v:.4g}" if abs(v) >= 1 else f"{v:.4f}"


fig = Figure(
    "04-stable-softmax", "Subtracting the maximum keeps every value finite",
    "Two rows of four-cell strips for the scores 2, 3, 1000, 1001 in FP32. Naive: exp gives 7.389, 20.09, inf, "
    "inf, the sum is inf, and the division gives 0, 0, nan, nan, drawn red. Stable: subtracting the maximum 1001 "
    "gives minus 999, minus 998, minus 1 and 0; exp gives 0, 0, 0.3679 and 1; the sum is 1.3679 and the division "
    "gives 0, 0, 0.2689 and 0.7311, drawn green.",
    subtitle="FP32, computed as precision_model.py does; the GPU's softmax kernel prints the same probabilities.",
    data_w=True)

COLS = [40, 264, 488, 712]
CELL = 48


def strip(col, y, vals, label, hazard=False, out=False):
    def fill(i, j):
        s = fmt(vals[j])
        if s in ("inf", "nan"):
            return "hazard-soft"
        return "memory-soft" if out else None
    g = fig.grid(COLS[col], y, 1, 4, cell=CELL, values=[[fmt(v) for v in vals]], fill=fill, font=13)
    fig.text(COLS[col], y + CELL + 24, label, "note")
    return g


fig.text(40, 144, "Naive: exp(x), then divide by the sum", "head")
a = strip(0, 160, x, "scores x")
b = strip(2, 160, e_naive, "exp(x); the sum is inf")
c = strip(3, 160, p_naive, "exp(x) / sum")
fig.arrow(a.box.anchor("right"), b.box.anchor("left"), label="exp")
fig.arrow(b.box.anchor("right"), c.box.anchor("left"))

fig.text(40, 328, "Stable: subtract the maximum M = 1,001 first", "head")
d = strip(0, 344, x, "scores x")
e = strip(1, 344, shift, "x minus M")
f = strip(2, 344, e_st, f"exp(x minus M); the sum is {float(S):.4f}")
g = strip(3, 344, p_st, "probabilities", out=True)
fig.chain([d.box, e.box, f.box, g.box])
fig.caption("The same factor scales numerator and sum, so the probabilities do not change.")
fig.write()
