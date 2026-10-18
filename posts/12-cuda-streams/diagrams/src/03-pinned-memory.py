"""Post 12, section 3: the two roads a copy from host memory takes, staged from pageable memory and direct from
pinned memory, with the measured cost of each.

Run from the series root:  uv run python posts/12-cuda-streams/diagrams/src/03-pinned-memory.py
Writes posts/12-cuda-streams/diagrams/03-pinned-memory.svg.
cuda_fig_design.md section 2: host memory green on the host side (blue labels), the host-device boundary a purple
dashed line, the device's global memory green; the staging copy the CPU makes is hazard red. The mechanism (the
driver copies pageable memory into a pinned staging buffer before the DMA, and an asynchronous copy from pageable
memory may be synchronous) is the CUDA 12.3 Programming Guide's, section 3.2.8.1, and Harris, "How to Optimize Data
Transfers in CUDA C/C++" (2012). The times are parsed from the output of snippets/streams.cu pasted in section 6 of
the post (measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026): the serial and
4-stream vector-addition rows from pageable and from pinned memory.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")


def row(name):
    m = re.search(rf"^{re.escape(name)} +([\d.]+) +([\d.]+) +([\d.]+)x ", BODY, re.M)
    return float(m.group(1)), float(m.group(2))


SER_PAGE, SER_PAGE_GBS = row("serial, pageable")
SER_PIN, SER_PIN_GBS = row("serial, pinned")
ST_PAGE, _ = row("4 streams, pageable")
ST_PIN, _ = row("4 streams, pinned")
assert ST_PAGE > ST_PIN * 2 and SER_PAGE > SER_PIN * 2
for v in (SER_PAGE, SER_PIN, ST_PAGE):
    assert f"{v:.2f} ms" in BODY, v

fig = Figure(
    "03-pinned-memory", "From pageable memory every copy is staged by the CPU",
    "Two rows, host on the left and device on the right of a purple dashed boundary. Top, pageable memory from an "
    "ordinary allocation: the CPU first copies each piece into a pinned staging buffer the driver owns, in red, and "
    "only then does the GPU's DMA engine copy it across; vector addition with its copies took "
    f"{SER_PAGE:.2f} ms serially and {ST_PAGE:.2f} ms on four streams. Bottom, pinned memory from cudaMallocHost: the "
    f"DMA engine reads it directly; {SER_PIN:.2f} ms serially and {ST_PIN:.2f} ms on four streams.",
    subtitle="Vector addition over 16,777,216 floats with its three copies; measured on the author's GTX 1650, October 2026.",
    data_w=True)

BX = 600


def host_card(x, y, w, title, sub):
    fig.card(x, y, w, 64, color="memory")
    fig.text(x + 16, y + 28, title, "label")
    fig.text(x + 16, y + 50, sub, "note")
    return Box(x, y, w, 64)


def gpu_card(y):
    fig.card(704, y, 216, 64, color="memory")
    fig.text(720, y + 28, "global memory", "label")
    fig.text(720, y + 50, "d_A, d_B, d_C", "code")
    return Box(704, y, 216, 64)


fig.edge((BX, 112), (BX, 476), color="sync", width=1.5, dash="proj")
fig.text(BX - 12, 468, "host", "head", color="host", anchor="end")
fig.text(BX + 12, 468, "device, across PCIe", "head", color="compute")

# pageable
Y1 = 136
fig.text(40, Y1, "Pageable: new, malloc, std::vector", "head")
h1 = host_card(40, Y1 + 48, 176, "the array", "pages may move")
s1 = host_card(296, Y1 + 48, 248, "staging buffer", "pinned, owned by the driver")
g1 = gpu_card(Y1 + 48)
fig.arrow((h1.right, h1.cy), (s1.x, s1.cy), color="hazard")
fig.text(h1.right + 8, Y1 + 36, "1. the CPU copies", "label", color="hazard")
fig.arrow((s1.right, s1.cy), (g1.x, g1.cy))
fig.text(BX + 16, Y1 + 36, "2. DMA", "label")
fig.text(40, Y1 + 144, f"serial {SER_PAGE:.2f} ms, four streams {ST_PAGE:.2f} ms: no overlap to gain", "label",
         color="hazard")

# pinned
Y2 = 328
fig.text(40, Y2, "Pinned: cudaMallocHost", "head")
h2 = host_card(40, Y2 + 48, 176, "the array", "page-locked")
g2 = gpu_card(Y2 + 48)
fig.arrow((h2.right, h2.cy), (g2.x, g2.cy))
fig.text(h2.right + 80, Y2 + 36, "DMA, straight from the array", "label")
fig.text(40, Y2 + 144, f"serial {SER_PIN:.2f} ms, four streams {ST_PIN:.2f} ms", "label")
fig.caption("Pinning removes the CPU's copy, and only then can a copy run beside a kernel.")
fig.write()
