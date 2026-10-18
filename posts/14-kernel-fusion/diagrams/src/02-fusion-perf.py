"""Post 14, sections 3 and 4: separate against fused, measured on the GTX 1650.

Run from the series root:  uv run python posts/14-kernel-fusion/diagrams/src/02-fusion-perf.py
Writes posts/14-kernel-fusion/diagrams/02-fusion-perf.svg.
cuda_fig_design.md section 5: bar charts on one scale per panel with the number on each bar. Every bar is parsed from
the output of snippets/fusion.cu pasted in index.md section 3 (measured on the author's GTX 1650, compute capability
7.5, CUDA 12.3, October 2026), its checks asserted "ok", its GB/s recomputed from its time, and its speed-ups recomputed
from its times. Left, the elementwise chain: the separate kernels, which move the intermediates, in hazard red; the
fused kernels in ink. Right, the tiled matrix multiplication: alone and with the ReLU in its epilogue in ink, followed
by a separate ReLU kernel in red.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
block = re.search(r"```text\n(device: NVIDIA GeForce GTX 1650[^\n]*\n.*?)```", BODY, re.S).group(1)
chain = {}
for m in re.finditer(r"^\s+(\d)\s+(separate|fused)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+)x\s+(\S+))?$",
                     block, re.M):
    ops, ver = int(m.group(1)), m.group(2)
    mb, ms, gbs = float(m.group(4)), float(m.group(5)), float(m.group(6))
    assert abs(mb / ms - gbs) < 0.15, (ops, ver)                    # GB/s = MB / ms / 1000 * 1000
    chain[(ops, ver)] = ms
    if ver == "fused":
        assert m.group(8) == "ok"
        assert abs(chain[(ops, "separate")] / ms - float(m.group(7))) < 0.006
assert sorted(chain) == [(o, v) for o in (2, 3, 4) for v in ("fused", "separate")]
mm = {k: (float(a), float(b) if b else None) for k, a, b in re.findall(
    r"^[ \t]+(matmul alone|ReLU kernel alone|matmul, then ReLU kernel|matmul with ReLU in its epilogue)[ \t]+([\d.]+)"
    r"[ \t]*([+-][\d.]+)?", block, re.M)}
assert len(mm) == 4 and re.search(r"matmul with ReLU in its epilogue[ \t]+[\d.]+[ \t]+[+-][\d.]+[ \t]+ok", block)
ADD_SEP, ADD_FUS = mm["matmul, then ReLU kernel"][1], mm["matmul with ReLU in its epilogue"][1]
ALONE = mm["matmul alone"][0]
for v in (f"added 0.113 ms, 1.7% of the multiplication", "added 0.007 ms, 0.1%"):
    assert v in BODY, v
assert (ADD_SEP, ADD_FUS) == (0.113, 0.007) and abs(100 * ADD_SEP / ALONE - 1.7) < 0.05
clk = re.search(r"SM clock: (\d+) MHz before the timed rows, (\d+) MHz after", block)
assert min(int(clk.group(1)), int(clk.group(2))) >= 1200

spd = {o: chain[(o, "separate")] / chain[(o, "fused")] for o in (2, 3, 4)}
fig = Figure(
    "02-fusion-perf", "Fused, the chain costs one pass over memory",
    f"Left: horizontal bars of time in milliseconds for an elementwise chain over 16,777,216 floats. Two operations: "
    f"separate {chain[(2, 'separate')]:.3f}, fused {chain[(2, 'fused')]:.3f}. Three: {chain[(3, 'separate')]:.3f} "
    f"and {chain[(3, 'fused')]:.3f}. Four: {chain[(4, 'separate')]:.3f} and {chain[(4, 'fused')]:.3f}. Separate bars "
    f"are red, fused bars ink. Right: the median time a ReLU adds to the tiled 1,024 cubed matrix multiplication "
    f"({ALONE:.1f} ms alone): {ADD_SEP:.3f} ms as a separate kernel after it, in red, and {ADD_FUS:.3f} ms in its "
    f"epilogue, in ink.",
    subtitle="Measured: GTX 1650, compute capability 7.5, CUDA 12.3, October 2026; chain: fastest of 5 passes.",
    data_w=True)
left, right = fig.row([11, 9])
lb = fig.panel(left, "Elementwise chain, 16,777,216 floats")
rows = []
for o in (2, 3, 4):
    rows += [(f"{o} ops, separate", chain[(o, "separate")], "hazard"), (f"{o} ops, fused", chain[(o, "fused")], "ink")]
fig.bar_chart(lb, rows, 0, 4, [0, 1, 2, 3, 4], label_w=128, value_w=56, axis_label="time (ms)",
              fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.3f}")
rb = fig.panel(right, f"ReLU added to a {ALONE:.1f} ms matmul")
fig.bar_chart(rb, [("separate kernel", ADD_SEP, "hazard"), ("in the epilogue", ADD_FUS, "ink")],
              0, 0.15, [0, 0.05, 0.1, 0.15], label_w=128, value_w=56, axis_label="median added time (ms), 20 passes",
              fmt_tick=lambda v: f"{v:.2f}", fmt_value=lambda v: f"{v:.3f}")
fig.caption(f"Fusing divides the chain's time by {spd[2]:.2f}, {spd[3]:.2f} and {spd[4]:.2f}; in an epilogue a ReLU costs almost nothing.")
fig.write()
