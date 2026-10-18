"""Post 14, hero (section 1): four separate kernels round-trip three intermediates through global memory;
one fused kernel reads x once and writes z once.

Run from the series root:  uv run python posts/14-kernel-fusion/diagrams/src/01-fusion-traffic.py
Writes posts/14-kernel-fusion/diagrams/01-fusion-traffic.svg.
cuda_fig_design.md sections 0 and 2: kernels are compute orange cards, global memory is a green band, every data
movement is an arrow between them; the reads and writes of the intermediates, which exist only to be read back,
are hazard red. The byte totals are derived by snippets/fusion_model.py (imported) for the 4-operation chain over
16,777,216 floats, and asserted against the model's pasted output in index.md.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure  # noqa: E402
import fusion_model as m  # noqa: E402

sep, fus = m.traffic(4)
SEP_MB, FUS_MB, ARR_MB = sep / 1e6, fus / 1e6, m.N * m.F / 1e6
assert (round(SEP_MB, 1), round(FUS_MB, 1), round(ARR_MB, 1)) == (536.9, 134.2, 67.1)
BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "    4          536.9       134.2              4.0x              75%  True" in BODY

fig = Figure(
    "01-fusion-traffic", "Fusion deletes the intermediates' round trips",
    f"Two rows over a green band of global memory. Top: four separate kernels, bias, scale, ReLU and GELU, in a "
    f"row; each reads a whole array from global memory and writes one back. The first reads x and the last writes "
    f"z; the three intermediates between them are written and read back, drawn red. Eight arrays of "
    f"{ARR_MB:.1f} MB move, {SEP_MB:.1f} MB in all. Bottom: one fused kernel keeps the value in a register "
    f"through the four operations, reads x once and writes z once, {FUS_MB:.1f} MB, a quarter of the traffic.",
    subtitle="Derived by fusion_model.py for 16,777,216 floats (67.1 MB per array) and a chain of 4 operations.",
    data_w=True)

X0, CW, GAP = 64, 168, 56                    # four kernel cards across
xs = [X0 + k * (CW + GAP) for k in range(4)]
names = ["bias", "scale", "ReLU", "GELU"]


def band(y, label):
    b = Box(40, y, 880, 32)
    fig.fill(b, "memory-soft", fit=False)
    fig.outline(b, "memory-line")
    fig.text(912, y + 21, label, "label", anchor="end", color="memory-ink")
    return b


# ---- top row: separate kernels ----
fig.text(40, 128, f"Separate: 4 launches, 8 arrays moved, {SEP_MB:.1f} MB", "head")
for x, nm in zip(xs, names):
    fig.card(x, 144, CW, 48, color="compute")
    fig.text(x + CW / 2, 173, nm, "label", anchor="middle")
top = band(248, "global memory")
inter = ["t0", "t1", "t0"]
for k, x in enumerate(xs):
    rx, wx = x + 40, x + CW - 40
    rd = "memory" if k == 0 else "hazard"
    wr = "memory" if k == 3 else "hazard"
    fig.arrow((rx, top.y), (rx, 192), color=rd)
    fig.arrow((wx, 192), (wx, top.y), color=wr)
    fig.text(rx - 8, 228, "x" if k == 0 else inter[k - 1], "code", anchor="end",
             color="memory-ink" if k == 0 else "hazard-ink")
    fig.text(wx + 8, 228, "z" if k == 3 else inter[k], "code",
             color="memory-ink" if k == 3 else "hazard-ink")

# ---- bottom row: one fused kernel ----
fig.text(40, 320, f"Fused: 1 launch, 2 arrays moved, {FUS_MB:.1f} MB", "head")
fx, fw = xs[1], xs[2] + CW - xs[1]
fig.card(fx, 336, fw, 48, color="compute")
fig.text(fx + fw / 2, 365, "bias, scale, ReLU, GELU on v, a register", "label", anchor="middle")
bot = band(440, "global memory")
fig.arrow((fx + 40, bot.y), (fx + 40, 384), color="memory")
fig.arrow((fx + fw - 40, 384), (fx + fw - 40, bot.y), color="memory")
fig.text(fx + 32, 420, "x", "code", anchor="end", color="memory-ink")
fig.text(fx + fw - 32, 420, "z", "code", color="memory-ink")

fig.legend(40, 500, [dict(color="memory", label="input or output", mark="line"),
                     dict(color="hazard", label="intermediate, written only to be read back", mark="line")],
           direction="row")
fig.write()
