"""Post 08, section 6: on the GTX 1650 only CUB's one-pass scan reaches the copy; every three-phase scan stays
under half of it.

Run from the series root:  uv run python posts/08-parallel-scan/diagrams/src/04-scan-perf.py
Writes posts/08-parallel-scan/diagrams/04-scan-perf.svg.
Every bar is parsed from the output of snippets/scan.cu pasted in index.md section 6 (measured on the author's
GTX 1650, compute capability 7.5, CUDA 12.3, October 2026) and asserted against the ratios the body states.
cuda_fig_design.md section 5: a speed comparison is a bar_chart on one scale with the number on each bar; the
hand-written kernels are compute (orange), the plain-index Blelloch, slowed by bank conflicts, hazard (red), and
the library baseline CUB ink. The copy kernel's bandwidth is a reference line (the ceiling), and half of it is the
most a three-phase scan can show, since it reads and writes the array twice (derived, scan_model.py section 6).
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
ROW = re.compile(r"^(copy \(the ceiling\)|Hillis-Steele, shared|Blelloch, plain index|Blelloch, conflict-free|"
                 r"warp shuffle|CUB DeviceScan)\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+ok$", re.MULTILINE)
MS = {m: float(t) for m, t, g in ROW.findall(BODY)}
GBS = {m: float(g) for m, t, g in ROW.findall(BODY)}
assert GBS == {"copy (the ceiling)": 140.2, "Hillis-Steele, shared": 33.8, "Blelloch, plain index": 27.9,
               "Blelloch, conflict-free": 39.6, "warp shuffle": 57.4, "CUB DeviceScan": 139.6}, GBS
assert round(MS["Blelloch, plain index"] / MS["Hillis-Steele, shared"], 2) == 1.21
assert round(MS["Blelloch, plain index"] / MS["Blelloch, conflict-free"], 2) == 1.42
assert round(MS["Hillis-Steele, shared"] / MS["warp shuffle"], 2) == 1.70
assert round(MS["warp shuffle"] / MS["CUB DeviceScan"], 2) == 2.43
assert "$4.808 / 3.965 = 1.21$" in BODY and "$3.965 / 2.337 = 1.70$" in BODY and "2.43 times faster" in BODY
COPY = GBS["copy (the ceiling)"]
LIMIT = round(COPY / 2, 1)
assert LIMIT == 70.1 and "three-phase limit of 70.1" in BODY

fig = Figure(
    "04-scan-perf", "Only the one-pass library scan reaches the copy",
    "A horizontal bar chart of bandwidth, the 8 N bytes a scan must move divided by its time, for a scan of "
    f"16,777,216 floats on the author's GTX 1650. Hillis-Steele in shared memory {GBS['Hillis-Steele, shared']} "
    f"GB/s, Blelloch with plain indices {GBS['Blelloch, plain index']} (red: bank conflicts), Blelloch "
    f"conflict-free {GBS['Blelloch, conflict-free']}, warp shuffle {GBS['warp shuffle']}, CUB "
    f"{GBS['CUB DeviceScan']}. Dotted lines mark the copy kernel's {COPY} GB/s and half of it, {LIMIT}, the most a "
    "three-phase scan can show.",
    subtitle="Measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026; 16,777,216 floats.",
    data_w=True)

body = fig.content
rows = [("Hillis-Steele, shared", GBS["Hillis-Steele, shared"], "compute"),
        ("Blelloch, bank conflicts", GBS["Blelloch, plain index"], "hazard"),
        ("Blelloch, conflict-free", GBS["Blelloch, conflict-free"], "compute"),
        ("warp shuffle", GBS["warp shuffle"], "compute"),
        ("CUB DeviceScan", GBS["CUB DeviceScan"], "ink")]
fig.bar_chart(body, rows, 0, 160, [0, 40, 80, 120, 160], label_w=200, value_w=64,
              axis_label="effective bandwidth, 8N bytes over time, GB/s",
              fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.1f}",
              ref_lines=[dict(x=LIMIT, label=f"three-phase limit {LIMIT}"), dict(x=COPY, label=f"copy {COPY}")])
fig.caption("A three-phase scan reads and writes the array twice, so it can show at most half the copy's rate.")
fig.write()
