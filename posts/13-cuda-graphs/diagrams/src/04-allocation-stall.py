"""Post 13, sections 5 and 6: how long each allocation call holds the host beside a kernel on another stream, and
what one allocate, launch, free and synchronize step costs three ways.

Run from the series root:  uv run python posts/13-cuda-graphs/diagrams/src/04-allocation-stall.py
Writes posts/13-cuda-graphs/diagrams/04-allocation-stall.svg.
cuda_fig_design.md section 5 (a comparison is a bar_chart on one scale with the number on each bar). Every value is
parsed from the run of snippets/graphs.cu pasted in index.md sections 5 and 6 (the author's GTX 1650, compute
capability 7.5, CUDA 12.3, October 2026) and asserted against the numbers the prose states. Left: host time per
call beside the 20.6 ms kernel, in ms, with that kernel as a reference line; cudaFree, which waits for it, is hazard
(red), cudaMalloc ink and the pool memory (green). Right: microseconds per step; the default pool, which returns its memory at every synchronize, is hazard,
cudaMalloc is ink (the baseline) and the pool that keeps its memory is memory (green).
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")


def value(pattern):
    return float(re.search(pattern, BODY, re.MULTILINE).group(1))


KERNEL = value(r"while a ([\d.]+) ms kernel runs on another stream")
MALLOC = value(r"^  cudaMalloc +: +([\d.]+)$")
FREE = value(r"^  cudaFree +: +([\d.]+)$")
TRIM = value(r"^  poolRoundTrip, pool trimmed +: +([\d.]+)$")
KEEP = value(r"^  poolRoundTrip, pool kept +: +([\d.]+)$")
STEP_M = value(r"^  cudaMalloc \+ cudaFree +([\d.]+) ")
STEP_0 = value(r"^  pool, release threshold 0 \(default\) +([\d.]+) ")
STEP_X = value(r"^  pool, release threshold maximum +([\d.]+) ")
assert (KERNEL, MALLOC, FREE, TRIM, KEEP) == (20.6, 145.2, 20343.4, 15.4, 9.2)
assert (STEP_M, STEP_0, STEP_X) == (519.32, 2004.06, 45.81)
assert "`cudaFree` held the host for 20.3 ms, 99% of the 20.6 ms kernel" in BODY
assert round(FREE / 1000, 1) == 20.3 and round(100 * FREE / 1000 / KERNEL) == 99

fig = Figure(
    "04-allocation-stall", "cudaFree waits for every stream; a kept pool does not",
    "Left: host time per call while a 20.6 ms kernel runs on another stream: cudaMalloc 0.15 ms, cudaFree 20.3 ms, "
    "which is the whole kernel, and the pool round trip 0.009 ms, against a reference line at 20.6 ms. Right: "
    "microseconds per step of allocate, kernel, free and synchronize: 519 with cudaMalloc and cudaFree, 2,004 from "
    "the pool at release threshold 0, the default, and 45.8 from the pool at the maximum threshold.",
    subtitle="graphs.cu on the author's GTX 1650 (7.5), CUDA 12.3, October 2026; 4 MiB, median of 7.",
    data_w=True)

left, right = fig.row(2, h=336)
lb = fig.panel(left, "Host time per call beside the kernel, ms")
fig.bar_chart(lb, [("cudaMalloc", MALLOC / 1000, "ink"), ("cudaFree", FREE / 1000, "hazard"),
                   ("pool round trip", KEEP / 1000, "memory")], 0, 25, [0, 5, 10, 15, 20, 25],
              label_w=128, value_w=56, fmt_tick=lambda v: f"{v:.0f}",
              fmt_value=lambda v: f"{v:.3f}" if v < 0.1 else (f"{v:.2f}" if v < 1 else f"{v:.1f}"),
              axis_label=f"dotted: the {KERNEL:.1f} ms kernel on the other stream",
              ref_lines=[dict(x=KERNEL)])

rb = fig.panel(right, "One step with a synchronize, µs")
fig.bar_chart(rb, [("cudaMalloc", STEP_M, "ink"), ("pool, threshold 0", STEP_0, "hazard"),
                   ("pool, maximum", STEP_X, "memory")], 0, 2400, [0, 800, 1600, 2400],
              label_w=136, value_w=64, fmt_tick=lambda v: f"{v:,.0f}",
              fmt_value=lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.1f}",
              axis_label="allocate, kernel, free, synchronize")
fig.caption("The default pool returns its memory at every synchronize; at the maximum threshold it keeps it.")
fig.write()
