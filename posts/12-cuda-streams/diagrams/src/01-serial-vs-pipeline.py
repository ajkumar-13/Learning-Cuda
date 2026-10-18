"""Post 12, the hero: the compute-heavy workload run serially and on four streams, on the GTX 1650's engines.

Run from the series root:  uv run python posts/12-cuda-streams/diagrams/src/01-serial-vs-pipeline.py
Writes posts/12-cuda-streams/diagrams/01-serial-vs-pipeline.svg.
cuda_fig_design.md section 4: a horizontal time axis with one lane per engine, copies green, kernels orange. Every
time is parsed from the output of snippets/streams.cu pasted in section 6 of the post (measured on the author's
GTX 1650, compute capability 7.5, CUDA 12.3, October 2026), for the kernel with 64 extra rounds:
- top, serial: the stages measured alone (the copy of A and of B, each half of the two-array copy in; the kernel;
  the result copy), end to end;
- bottom, four streams: the CUDA event recorded after each operation of the fastest pass (A in, B in, kernel,
  C out). An event gives the time an operation finished, not when it started, so each bar ends at its event and its
  start is inferred as the latest of: the end of the operation before it in its stream, the end of the operation
  before it in its lane, and its own end minus the time the same operation took measured alone (per chunk: a
  quarter of one array's copy, of the kernel, of the result copy). The subtitle says the starts are inferred.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Box, Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
CLOCK = int(re.search(r"SM clock: \d+ MHz before the timed rows, (\d+) MHz after", BODY).group(1))
assert f"about {CLOCK:,} MHz" in BODY
T_IN = float(re.search(r"stages alone, pinned: copy A and B in ([\d.]+) ms", BODY).group(1))
T_OUT = float(re.search(r"copy C out ([\d.]+) ms \(", BODY).group(1))
T_K = float(re.search(r"with 64 extra rounds ([\d.]+) ms", BODY).group(1))
SERIAL = float(re.search(r"^serial, pinned +[\d.]+ +[\d.]+ +[\d.]+x +([\d.]+) ", BODY, re.M).group(1))
block = BODY.split("timeline, 4 streams, extra rounds")[1]
EV = [[float(v) for v in m] for m in re.findall(r"stream \d: +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)", block)[:4]]
assert len(EV) == 4
for row in EV:
    assert row == sorted(row), row
PIPE = max(r[3] for r in EV)
COPIES = T_IN + T_OUT
assert f"{PIPE:.2f} ms" in BODY and f"{SERIAL:.2f}" in BODY and f"{COPIES:.2f} ms" in BODY

# bars of the pipelined run: (lane, label, start, end); lanes 0 copy in (A and B apart), 1 kernel, 2 copy out
ALONE = {"A": T_IN / 2 / 4, "B": T_IN / 2 / 4, "K": T_K / 4, "C": T_OUT / 4}   # per chunk, measured alone
LANE = {"A": 0, "B": 0, "K": 1, "C": 2}
bars, lane_end = [], [0.0, 0.0, 0.0]
for i, row in enumerate(EV):
    prev = 0.0
    for op, end in zip("ABKC", row):
        lane = LANE[op]
        start = max(prev, lane_end[lane], end - ALONE[op])
        assert start < end, (op, i, start, end)
        bars.append((lane, f"{op}{i + 1}" if op != "K" else str(i + 1), start, end))
        lane_end[lane] = prev = end
# the copy lanes, as drawn: how long a copy in and a copy out are drawn at the same time
ins = [(t0, t1) for lane, _, t0, t1 in bars if lane == 0]
outs = [(t0, t1) for lane, _, t0, t1 in bars if lane == 2]
DRAWN_OVERLAP = sum(max(0.0, min(a1, b1) - max(a0, b0)) for a0, a1 in ins for b0, b1 in outs)

fig = Figure(
    "01-serial-vs-pipeline", "Streams hide the kernel behind the copies, not the copies",
    f"Two timelines on one axis in milliseconds, each with lanes for the copy in, the kernel and the copy out of "
    f"16,777,216 floats with 64 extra rounds of work per element. Top, serial: A and B copied in, {T_IN:.2f} ms "
    f"together, then the kernel {T_K:.2f} ms, then C copied out {T_OUT:.2f} ms, one engine busy at a time; the "
    f"measured serial run took {SERIAL:.2f} ms. Bottom, four streams: four chunks, each with bars A, B, the kernel "
    f"and C that end at the CUDA events of the fastest pass, with starts inferred; the kernel of one chunk runs "
    f"while the next chunk is copied in, and the copies in and out mostly follow one another. The last copy out ends at "
    f"{PIPE:.2f} ms, near the {COPIES:.2f} ms the two copies take end to end.",
    subtitle=f"GTX 1650 at about {CLOCK:,} MHz, CUDA 12.3, October 2026; bars end at CUDA events, starts inferred.",
    data_w=True)

LX, X0, X1, LH = 40, 176, 880, 32
SPAN = 30.0
assert SERIAL < SPAN and PIPE < SPAN
U = (X1 - X0) / SPAN
LANES = ["copy in", "kernel", "copy out"]
KIND = ["memory", "compute", "memory"]


def bar(y, lane, t0, t1, text):
    b = Box(X0 + U * t0, y + LH * lane + 4, U * (t1 - t0), LH - 8)
    fig.fill(b, KIND[lane] + "-soft", radius=4)
    fig.outline(b, KIND[lane], radius=4)
    if text and b.w >= 24:
        fig.text(b.cx, b.y + 18, text, "label", anchor="middle", snap=False)


def lanes(y0, heading):
    fig.text(LX, y0 - 12, heading, "head")
    for r, name in enumerate(LANES):
        fig.text(LX, y0 + LH * r + 21, name, "label")
        fig.edge((X0, y0 + LH * (r + 1)), (X1, y0 + LH * (r + 1)), color="border", width=0.75)


Y1 = 136
lanes(Y1, "Serial, pinned memory: one engine busy at a time")
with fig.data():
    bar(Y1, 0, 0, T_IN / 2, "A in")
    bar(Y1, 0, T_IN / 2, T_IN, "B in")
    bar(Y1, 1, T_IN, T_IN + T_K, "kernel")
    bar(Y1, 2, T_IN + T_K, T_IN + T_K + T_OUT, "C out")
    fig.text(X0 + U * max(SERIAL, T_IN + T_K + T_OUT) + 8, Y1 + LH * 2 + 21, f"{SERIAL:.2f} ms", "value", snap=False)

Y2 = 300
lanes(Y2, "Four streams: chunk 1 to 4, each in order")
with fig.data():
    for lane, text, t0, t1 in bars:
        bar(Y2, lane, t0, t1, text)
    fig.text(X0 + U * PIPE + 8, Y2 + LH * 2 + 21, f"{PIPE:.2f} ms", "value", color="compute-ink", snap=False)
    for t in range(0, 31, 5):
        x = X0 + U * t
        fig.edge((x, Y2 + LH * 3), (x, Y2 + LH * 3 + 6), color="ink-muted", width=1)
        fig.text(x, Y2 + LH * 3 + 24, f"{t} ms" if t == 30 else str(t), "tick", anchor="middle", snap=False)
fig.text(LX, Y2 + LH * 3 + 52, f"Four streams end at {PIPE:.2f} ms, near the {COPIES:.2f} ms the copies take end "
         "to end; only the kernel fully hides.", "note")
fig.caption("On this laptop the copies in and out mostly take turns, so the kernel is what hides.")
fig.write()
