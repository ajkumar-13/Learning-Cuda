"""Post 02, section 3: the two memory spaces, the six host steps, and the width of each road.

Run from the series root:  uv run python posts/02-vector-addition/diagrams/src/03-host-device-copy.py
Writes posts/02-vector-addition/diagrams/03-host-device-copy.svg.
A Section 2 form of cuda_fig_design.md: the host side in blue with its arrays in green host memory, the host-device
boundary as a purple dashed line (the figure is about crossing it), the device's global memory in green and the
kernel in orange. The six numbered steps are those of snippets/vector_add.cu. The two bandwidths are parsed from the
vector_add_bench output pasted in section 4 of the post (measured on the author's GTX 1650, compute capability 7.5,
CUDA 12.3, October 2026): the pinned host-to-device copy and the kernel's achieved bandwidth at 10 million elements.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Box, Figure  # noqa: E402

POST = Path(__file__).resolve().parents[2]
BODY = (POST / "index.md").read_text(encoding="utf-8")
CU = (POST / "snippets" / "vector_add.cu").read_text(encoding="utf-8")
m = re.search(r"host to device, 40 MB: pinned ([\d.]+) GB/s, pageable ([\d.]+) GB/s", BODY)
PINNED, PAGEABLE = m.group(1), m.group(2)
KERNEL_GBS = re.search(r"^\s+10000000\s.*?\s([\d.]+)\s+ok$", BODY, re.M).group(1)
for step in ("// 1. allocate", "// 2. copy inputs in", "// 3. launch", "// 4. wait", "// 5. copy the result out",
             "// 6. free"):
    assert step in CU, step
assert f"measured at {PINNED} GB/s from pinned memory" in BODY and f"at a measured {KERNEL_GBS} GB/s" in BODY

fig = Figure(
    "03-host-device-copy", "Two memories, six steps, and a narrow road between them",
    "Left, the host: the host program and three arrays in host memory, h_A, h_B and h_C. Right, the device: "
    "d_A, d_B and d_C in global memory and the vectorAdd kernel, where thread i writes C[i] = A[i] + B[i]. A purple "
    f"dashed line between them is the PCIe 3.0 bus of 16 lanes, measured at {PINNED} GB/s from pinned memory and {PAGEABLE} "
    f"GB/s from pageable memory. The kernel reads and writes global memory at a measured {KERNEL_GBS} GB/s. Numbered "
    "steps: 1 allocate d_A, d_B, d_C; 2 copy the inputs in across the bus; 3 launch from the host program; "
    "4 synchronize; 5 copy the result out; 6 free.",
    subtitle="Bandwidths measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026.",
    data_w=True)

# the host side
fig.text(40, 128, "host: CPU", "head", color="host")
fig.card(40, 144, 288, 72, heading="host program", lines=["runs steps 1 to 6"], color="host")
H = {}
for name, y in (("h_A", 256), ("h_B", 312), ("h_C", 408)):
    fig.card(40, y, 160, 48, color="memory")
    fig.text(56, y + 30, name, "code")
    H[name] = Box(40, y, 160, 48)
fig.text(40, 476, "host memory", "note")

# the bus
BX = 456
fig.edge((BX, 112), (BX, 460), color="sync", width=1.5, dash="proj")
fig.text(BX - 8, 380, "PCIe 3.0, 16 lanes:", "label", color="sync", anchor="end")
fig.text(BX - 8, 400, f"{PINNED} GB/s pinned, {PAGEABLE} pageable", "label", color="sync", anchor="end")

# the device side
fig.text(600, 128, "device: GPU", "head", color="compute")
D = {}
for name, y in (("d_A", 256), ("d_B", 312), ("d_C", 408)):
    fig.card(600, y, 104, 48, color="memory")
    fig.text(616, y + 30, name, "code")
    D[name] = Box(600, y, 104, 48)
fig.text(600, 476, "global memory: 1. allocate, 6. free", "note")
K = Box(776, 256, 144, 200)
fig.card(K.x, K.y, K.w, K.h, heading="vectorAdd", lines=["thread i", "writes C[i]"], color="compute")
fig.text(K.x + 16, K.y + 128, f"{KERNEL_GBS} GB/s", "value", color="compute-ink")
fig.text(K.x + 16, K.y + 152, "to global memory", "note")

# copies across the bus
fig.arrow((H["h_A"].right, H["h_A"].cy), (D["d_A"].x, D["d_A"].cy))
fig.text(216, 272, "2. copy in", "label")
fig.arrow((H["h_B"].right, H["h_B"].cy), (D["d_B"].x, D["d_B"].cy))
fig.arrow((D["d_C"].x, D["d_C"].cy), (H["h_C"].right, H["h_C"].cy))
fig.text(216, 424, "5. copy out", "label")
# the launch, and the wait
fig.arrow((328, 168), (880, K.y), via=[(880, 168)], color="host")
fig.text(344, 160, "3. launch", "label", color="host")
fig.arrow((816, K.y), (328, 200), via=[(816, 200)], color="sync")
fig.text(344, 224, "4. synchronize", "label", color="sync")
# the kernel's own road to global memory
fig.arrow((D["d_A"].right, D["d_A"].cy), (K.x, D["d_A"].cy))
fig.arrow((D["d_B"].right, D["d_B"].cy), (K.x, D["d_B"].cy))
fig.arrow((K.x, D["d_C"].cy), (D["d_C"].right, D["d_C"].cy))
fig.caption("The kernel's road to global memory is over ten times wider than the bus that fills it.")
fig.write()
