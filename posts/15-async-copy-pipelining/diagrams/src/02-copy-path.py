"""Post 15, section 2: the copy of one float from global to shared memory in the two builds of matmulPipelined.

Run from the series root:  uv run python posts/15-async-copy-pipelining/diagrams/src/02-copy-path.py
Writes posts/15-async-copy-pipelining/diagrams/02-copy-path.svg.
cuda_fig_design.md section 2 (memory tiers green, a data movement an arrow labelled with what moves) and section 0
(a wait is sync, purple; a stall is hazard, red). The instruction names are the SASS that cuobjdump printed for the
sm_75 and sm_80 builds (CUDA 12.3, October 2026) and the PTX that nvcc emitted for the 4-byte copies; the script
asserts each against the listing and the table that index.md quotes.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
for s in ("@P3 LDGSTS.E [R5], [R18.64] ;", "DEPBAR.LE SB0, 0x1 ;", "@P2 LDG.E.CONSTANT.SYS R7, [R26] ;",
          "@P2 STS [R6], R7 ;", "`cp.async.ca.shared.global`", "`DEPBAR.LE SB0, n`"):
    assert s in BODY, s

fig = Figure(
    "02-copy-path", "One source line, two copies: via a register or around it",
    "Two panels, each with global memory above shared memory. Left, the sm_75 build: LDG loads the float from global "
    "memory into the register file, STS stores it from the register into shared memory, and the warp waits at the STS "
    "until the load returns. Right, the sm_80 build: one LDGSTS instruction, the PTX cp.async.ca, copies global memory "
    "to shared memory without the register file, which is drawn dashed and unused, and the warp waits only at "
    "DEPBAR.LE SB0, 0x1, the cp.async.wait_group 1 of __pipeline_wait_prior(1).",
    subtitle="__pipeline_memcpy_async in matmulPipelined, read from cuobjdump -sass of each build, CUDA 12.3.",
    data_w=True)

left, right = fig.row(2)
CW, CH = 200, 48


def tier(x, y, name, dashed=False):
    b = Box(x, y, CW, CH)
    if dashed:
        fig.outline(b, "ink-muted", width=1, dash="proj", radius=8)
        fig.text(b.cx, y + 30, name, "note", anchor="middle")
    else:
        fig.fill(b, "memory-soft", radius=8)
        fig.outline(b, "memory", width=1.5, radius=8)
        fig.text(b.cx, y + 30, name, "label", anchor="middle")
    return b


YG, YR, YS = 136, 248, 360

# left: the sm_75 build
lb = fig.panel(left, "sm_75 build: through a register")
g = tier(lb.x, YG, "global memory")
r = tier(lb.x, YR, "register file")
s = tier(lb.x, YS, "shared memory")
fig.arrow(g.anchor("bottom"), r.anchor("top"))
fig.arrow(r.anchor("bottom"), s.anchor("top"))
fig.text(g.cx + 16, YG + CH + 36, "LDG", "code")
fig.text(r.cx + 16, YR + CH + 36, "STS", "code")
fig.text(r.cx + 72, YR + CH + 36, "waits for the loaded value", "label", color="hazard")
fig.text(lb.x, YS + CH + 36, "the warp stalls at every copy", "label", color="hazard")

# right: the sm_80 build
rb = fig.panel(right, "sm_80 build: around the registers")
g2 = tier(rb.x, YG, "global memory")
s2 = tier(rb.x, YS, "shared memory")
r2 = tier(rb.x + CW + 24, YR, "register file: unused", dashed=True)
fig.arrow(g2.anchor("bottom"), s2.anchor("top"), color="memory", width=2.5)
fig.text(g2.cx + 16, YR + 14, "LDGSTS", "code")
fig.text(g2.cx + 16, YR + 38, "cp.async.ca", "code")

fig.text(rb.x, YS + CH + 36, "DEPBAR.LE SB0, 0x1", "code", color="sync")
fig.text(rb.x + 176, YS + CH + 36, "the only wait, before the math", "label", color="sync")

fig.caption("On sm_75 the warp stalls at the store; on sm_80 it stalls only where it asks to.")
fig.write()
