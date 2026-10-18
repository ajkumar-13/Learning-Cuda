"""Post 10, section 5: the three occupancy limits for the register-heavy launch, and the warp slots it fills.

Run from the series root:  uv run python posts/10-roofline/diagrams/src/02-occupancy-limiters.py
Writes posts/10-roofline/diagrams/02-occupancy-limiters.svg.
Every number is derived by snippets/roofline_model.py (imported): occupancy(96, 0, 256) gives the blocks per SM
allowed by block size, registers and shared memory, the binding limit and the resident warps. Left, a bar chart
(cuda_fig_design.md section 5) with the binding bar in hazard red and the others ink; right, the SM's 32 warp slots
as four rows of eight, one row per quarter of the register file (one per warp scheduler), resident warps compute
orange, the empty slots outlined in hazard red.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure, num  # noqa: E402
import roofline_model as m  # noqa: E402

REGS, SMEM, BLOCK = 96, 0, 256
by_block, by_reg, by_smem, blocks, limiter, resident, occ = m.occupancy(REGS, SMEM, BLOCK)
assert (by_block, by_reg, by_smem, blocks, limiter, resident, occ) == (4, 2, 16, 2, "registers", 16, 0.5)
per_warp = REGS * 32
fit = (m.REGS_PER_SM // m.REG_QUARTERS) // per_warp
assert per_warp == 3072 and fit == 5 and resident // m.REG_QUARTERS == 4
BODY = (POST / "index.md").read_text(encoding="utf-8")
assert "register-heavy              96      0    256        4     2     16       2  registers    50%" in BODY

fig = Figure(
    "02-occupancy-limiters", "Registers bind: two blocks fit, half the warp slots fill",
    f"Left: horizontal bars of the blocks per SM that each limit allows for a launch of {BLOCK}-thread blocks using "
    f"{REGS} registers per thread and no shared memory: block size {by_block}, registers {by_reg}, shared memory "
    f"{by_smem}; the registers bar is red, the binding limit. Right: the SM's 32 warp slots drawn as four rows of "
    f"eight, one row per quarter of the register file; {resident} slots, four per row, hold the warps of the "
    f"{blocks} resident blocks, and the other {m.WARP_SLOTS - resident} are empty: occupancy {occ:.0%}.",
    subtitle=f"Derived by roofline_model.py: {BLOCK} threads per block, {REGS} registers per thread, no shared memory.",
    data_w=True)

left, right = fig.row([5, 4])
lb = fig.panel(left, "Blocks per SM each limit allows")
fig.bar_chart(lb, [("block size", by_block, "ink"), ("registers", by_reg, "hazard"),
                   ("shared memory", by_smem, "ink")],
              0, 16, [0, 4, 8, 12, 16], fmt=lambda v: f"{v:.0f}", label_w=128, value_w=40,
              axis_label="blocks per SM")

rb = fig.panel(right, f"32 warp slots: {resident} filled, {occ:.0%}")
g = fig.grid(rb.x + 72, rb.y + 8, 4, 8, cell=32,
             fill=lambda i, j: "compute" if j < resident // 4 else None)
g.row_labels([f"quarter {k}" for k in range(4)])
g.outline(0, resident // 4, 4, 8 - resident // 4, "hazard", width=1.5)
fig.note(Box(rb.x, g.box.y, g.box.right - rb.x, g.box.h), [f"A quarter's 16,384 registers hold {fit} warps of {num(per_warp)};",
                 "blocks of 8 warps come whole, so 4 per quarter reside."])
fig.legend(rb.x, g.box.bottom + 92, [dict(color="compute", label="resident warp", mark="swatch"),
                                          dict(color="hazard", label="empty slot", mark="outline")],
           direction="row")
fig.caption("The smallest bar binds: registers allow 2 blocks of 8 warps, so 16 of 32 slots fill.")
fig.write()
