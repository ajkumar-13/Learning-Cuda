"""Post 16, hero (section 1): four bit strips, FP32, FP16, BF16 and TF32, each split into sign, exponent and
mantissa, with the largest finite value and the spacing eps each split gives.

Run from the series root:  uv run python posts/16-numerical-precision/diagrams/src/01-float-formats.py
Writes posts/16-numerical-precision/diagrams/01-float-formats.svg.
cuda_fig_design.md section 0: bits are data at rest, so the exponent fields are memory green and the mantissa
fields neutral; BF16's exponent, the field that buys FP32's range, carries the emphasis outline; FP16's small
maximum is the hazard. Every max and eps is derived here from the bit widths with the model's formulas and
asserted against the table precision_model.py prints in index.md.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure, MINUS, TIMES, rich, sup  # noqa: E402

FORMATS = [("FP32", 8, 23), ("FP16", 5, 10), ("BF16", 8, 7), ("TF32", 8, 10)]
BODY = (POST / "index.md").read_text(encoding="utf-8")


def derive(e, m):
    bias = 2 ** (e - 1) - 1
    return (2 - 2.0 ** -m) * 2.0 ** bias, 2.0 ** -m


facts = {n: derive(e, m) for n, e, m in FORMATS}
for n, e, m in FORMATS:                      # the model's printed table, as pasted in the body
    row = re.search(rf"^  {n} +(\d+) +{e} +{m} +(\S+) +\S+ +(\S+) ", BODY, re.MULTILINE)
    assert row, n
    mx, eps = facts[n]
    big = "%.0f" % mx if mx < 1e6 else "%.3e" % mx
    assert row.group(2) == big and row.group(3) == "%.3e" % eps, (n, row.groups())
assert facts["FP16"][0] == 65504


def sci(v, digits=2):
    """3.40 × 10 to the 38, with the true minus."""
    mant, exp = ("%.*e" % (digits, v)).split("e")
    exp = int(exp)
    return rich(mant, f" {TIMES} ", sup("10", (MINUS if exp < 0 else "") + str(abs(exp)), italic=False))


fig = Figure(
    "01-float-formats", "Same idea, four splits of the bits",
    "Four bit strips drawn to scale, 16 units per bit, each a sign bit, an exponent field in green and a mantissa "
    "field. FP32: 1, 8, 23 bits, largest value 3.40 times 10 to the 38, eps 1.19 times 10 to the minus 7. FP16: "
    "1, 5, 10, largest 65,504, drawn red as the format that overflows, eps 9.77 times 10 to the minus 4. BF16: "
    "1, 8, 7, outlined because its 8 exponent bits give FP32's range, largest 3.39 times 10 to the 38, eps 7.81 "
    "times 10 to the minus 3. TF32: 1, 8, 10, 19 bits, largest 3.40 times 10 to the 38, eps 9.77 times 10 to the "
    "minus 4.",
    subtitle="Bits to scale. Largest value and spacing eps derived from the widths by precision_model.py.",
    data_w=True)

X0, BIT, H = 176, 16, 40
fig.text(X0 + 8, 132, "s", "tick", anchor="middle")
fig.text(X0 + 16, 132, "exponent", "tick")
fig.text(X0 + 16 + 8 * BIT, 132, "mantissa", "tick")
fig.text(712, 132, "largest value, eps", "tick")
for k, (name, e, m) in enumerate(FORMATS):
    y = 144 + k * 80
    fig.text(40, y + 26, name, "head")
    fig.text(96, y + 26, f"{1 + e + m} bits", "note")
    s = Box(X0, y, BIT, H)
    ex = Box(X0 + BIT, y, e * BIT, H)
    ma = Box(X0 + BIT + e * BIT, y, m * BIT, H)
    fig.fill(s, "neutral-soft", fit=False)
    fig.fill(ex, "memory-soft", fit=False)
    fig.fill(ma, "surface", fit=False)
    for b in (s, ex, ma):
        fig.outline(b, "rule")
    fig.text(ex.cx, y + 26, str(e), "label", anchor="middle", color="memory-ink")
    fig.text(ma.cx, y + 26, str(m), "label", anchor="middle")
    if name == "BF16":
        fig.outline(ex, "memory", width=2.5)
    mx, eps = facts[name]
    if name == "FP16":
        fig.text(712, y + 12, "largest 65,504", "label", color="hazard-ink")
    else:
        fig.text(712, y + 12, rich("largest ", sci(mx)), "label")
    fig.text(712, y + 44, rich("eps ", sci(eps)), "note")

fig.legend(40, 476, [dict(color="memory", label="exponent bits: the range"),
                     dict(color="ink-muted", label="mantissa bits: the precision", mark="outline")],
           direction="row")
fig.caption("FP16 spends its 16 bits on precision and overflows at 65,504; BF16 spends them on FP32's range.")
fig.write()
