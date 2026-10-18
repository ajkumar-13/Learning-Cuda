"""Post 07, section 3: a constant-memory read costs one request per distinct address in the warp.

Run from the series root:  uv run python posts/07-convolution/diagrams/src/02-constant-broadcast.py
Writes posts/07-convolution/diagrams/02-constant-broadcast.svg.
cuda_fig_design.md sections 1 and 2: a warp is a row of 32 orange lane squares; the constant cache and the image are
memory green; the 32 serialized requests are hazard red. The rule is the Programming Guide's (CUDA 12.3, "Constant
Memory": a request is split into as many requests as there are different addresses in it). The counts are computed
here for warp 0 of a 16 x 16 block on the 1920-wide image, at the center tap (i, j) = (1, 1) of the 3 x 3 filter:
the filter index i * K + j is the same for all 32 lanes, the pixel index row * W + col differs for all 32.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
W, K, TILE, WARP = 1920, 3, 16, 32
I, J = 1, 1
lanes = [(t // TILE, t % TILE) for t in range(WARP)]            # (threadIdx.y, threadIdx.x) of warp 0
mask_addr = {I * K + J for _ in lanes}
pixel_addr = {(ty + I - 1) * W + (tx + J - 1) for ty, tx in lanes}
assert (len(mask_addr), len(pixel_addr)) == (1, 32)
assert "c_mask[i * K + j]" in BODY and "64 KB" in BODY and "8 KB" in BODY

fig = Figure(
    "02-constant-broadcast", "Constant memory serves one address per request",
    "Top: the 32 lanes of a warp all read the filter weight c_mask[4] at the same moment; the constant cache serves "
    "the one address with one request, broadcast to every lane. Bottom: if the warp read its 32 pixels, 32 different "
    "addresses, from constant memory, the read would split into 32 requests served one after another. Constant "
    "memory is 64 KB on the device, with an 8 KB cache working set per SM on compute capability 7.5.",
    subtitle="Warp 0 of a 16 × 16 block, center tap of a 3 × 3 filter; the rule is the CUDA Programming Guide's.",
    height=720, data_w=True)

LX, CW, CH = 112, 20, 32                                       # 32 lanes of 20 = 640 wide


def warp_row(y, label):
    g = fig.grid(LX, y, 1, WARP, cell_w=CW, cell_h=CH, fill=lambda i, j: "compute-soft")
    g.ticks([0, 31], ["lane 0", "lane 31"], side="below")
    fig.text(40, y + 21, label, "label")
    return g


# top: one address, one request
fig.text(40, 128, "c_mask[i * K + j]: one address for the whole warp", "head")
cache = fig.grid(LX, 160, 1, 9, cell_w=48, cell_h=CH, values=[[f"[{k}]" for k in range(9)]], font=14,
                 fill=lambda i, j: "memory-soft")
cache.window(4, 4, color="memory")
fig.text(LX + 9 * 48 + 16, 181, "constant cache: c_mask, 9 weights", "label")
top = warp_row(272, "warp 0")
src = cache.cell(0, 4)
fig.arrow((src.cx, src.bottom), (src.cx, 272), label="1 request, broadcast", side="right", color="memory")
fig.brace(LX, LX + WARP * CW, 264, side="above", color="memory")
fig.text(LX + WARP * CW + 16, 293, "32 lanes", "label")

# bottom: 32 addresses, 32 requests
fig.text(40, 384, "if the pixels row * W + col were in constant memory: 32 addresses", "head")
mem = fig.grid(LX, 416, 1, WARP, cell_w=CW, cell_h=CH, fill=lambda i, j: "memory-soft")
fig.text(LX + WARP * CW + 16, 437, "32 pixels, 2 image rows", "label")
bot = warp_row(528, "warp 0")
with fig.data():
    for k in range(WARP):
        a, b = mem.cell(0, k), bot.cell(0, k)
        fig.edge((a.cx, a.bottom + 4), (b.cx, b.y - 4), color="hazard", width=1)
fig.text(LX, 616, "32 requests, served one after another", "label", color="hazard", bold=True)
fig.text(LX, 644, "Constant memory: 64 KB per device, an 8 KB cache working set per SM (compute capability 7.5).",
         "note")
fig.caption("The filter suits constant memory: every lane reads the same weight at the same time.")
fig.write()
