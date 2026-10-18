"""Post 11, section 5: theoretical occupancy (model and runtime) and the measured bandwidth of SAXPY for block sizes
32 to 1,024.

Run from the series root:  uv run python posts/11-profiling-debugging/diagrams/src/03-occupancy-measured.py
Writes posts/11-profiling-debugging/diagrams/03-occupancy-measured.svg.
Every measured value is parsed from the output of snippets/occupancy_demo.cu pasted in index.md section 5 (the
author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026). The model's theoretical occupancy is
re-derived here from post 10's rules for sm_75 (32 warp slots, 16 blocks, 65,536 registers per SM in four quarters
of 16,384 that each hold whole warps, registers allocated per warp in units of 256) and the 10 registers that ptxas printed (section 4), and asserted equal to the
runtime's column. cuda_fig_design.md section 5: the model is drawn hollow, the measured values solid; the one-warp
blocks, whose empty warp slots cost bandwidth, are hazard (red); the other bars are ink, on an axis that ends at
the 160 GB/s peak.
"""
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
REGS = int(re.search(r"ptxas info    : Used (\d+) registers, 376 bytes cmem\[0\]", BODY).group(1))
assert REGS == 10
ROW = re.compile(r"^ *(\d+) +(\d+) +(\d+) +(\d+)% +([\d.]+) +([\d.]+) +(\d+)%$", re.MULTILINE)
rows = [tuple(float(v) for v in r) for r in ROW.findall(BODY)]
sizes = [int(r[0]) for r in rows]
assert sizes == [32, 64, 128, 256, 512, 1024], sizes
theo = [r[3] for r in rows]
gbs = [r[5] for r in rows]
assert gbs == [107.4, 145.3, 145.2, 145.1, 145.2, 145.3], gbs


def model(regs, block, slots=32, max_blocks=16, regs_sm=65536, grain=256):
    """Post 10's occupancy model for sm_75, no shared memory."""
    warps = math.ceil(block / 32)
    per_warp = math.ceil(regs * 32 / grain) * grain
    blocks = min(slots // warps, (regs_sm // 4 // per_warp * 4) // warps, max_blocks)
    return 100.0 * min(blocks * warps, slots) / slots


mod = [model(REGS, b) for b in sizes]
assert mod == theo == [50, 100, 100, 100, 100, 100], (mod, theo)
assert "107.4 GB/s, 26% less" in BODY and round(100 * (1 - gbs[0] / gbs[3])) == 26

fig = Figure(
    "03-occupancy-measured", "Only the one-warp blocks lose occupancy and bandwidth",
    "Left: occupancy in percent against threads per block, 32, 64, 128, 256, 512 and 1,024. Post 10's model, "
    "hollow circles, and the CUDA runtime's theoretical occupancy, squares, agree: 50% at 32 and 100% from 64 up. "
    "Right: measured bandwidth, 107.4 GB/s at 32 threads per block, in red, and 145.3, 145.2, 145.1, 145.2 and "
    "145.3 GB/s from 64 to 1,024, on an axis that ends at the 160 GB/s peak.",
    subtitle="SAXPY, 10 registers per thread, on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026.",
    data_w=True)

left, right = fig.row(2, h=320)
lb = fig.panel(left, "Occupancy, percent")
idx = list(range(len(sizes)))
fig.line_chart(lb, [
    dict(xs=idx, ys=mod, color="ink-muted", shape="circle", hollow=True, size=16, label=None),
    dict(xs=idx, ys=theo, color="ink", shape="square", size=8, label=None),
], x=(-0.4, 5.4, idx), y=(0, 108, [0, 25, 50, 75, 100]), x_label="threads per block",
    fmt_x=lambda v: f"{sizes[int(round(v))]:,}", fmt_y=lambda v: f"{v:.0f}", label_w=16)

rb = fig.panel(right, "Bandwidth by threads per block, GB/s")
bars = [(f"{b:,}", g, "hazard" if b == 32 else "ink") for b, g in zip(sizes, gbs)]
fig.bar_chart(rb, bars, 0, 160, [0, 40, 80, 120, 160], label_w=56, value_w=56,
              axis_label="fastest of 5 launches; the axis ends at the 160 GB/s peak",
              fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.1f}")

fig.legend(left.x, 456, [dict(color="ink-muted", label="post 10's model", mark="circle", hollow=True),
                         dict(color="ink", label="runtime, theoretical", mark="square")], direction="row")
fig.caption("Every block size from 64 up keeps memory busy; only the one-warp blocks lose bandwidth.")
fig.write()
