"""Post 07, section 6: which fix pays off depends on the filter size, measured on the GTX 1650.

Run from the series root:  uv run python posts/07-convolution/diagrams/src/04-conv-perf.py
Writes posts/07-convolution/diagrams/04-conv-perf.svg.
Every bar is parsed from the output of snippets/convolution.cu pasted in index.md section 6 (measured on the author's
GTX 1650, compute capability 7.5, CUDA 12.3, October 2026) and asserted against the table and the ratios the body
states. cuda_fig_design.md section 5: a speed comparison is a bar_chart on one scale with the number on each bar; the
naive baseline is ink, the three fixes, each a change of where data lives, memory green; the dotted line is each
panel's copy kernel, the practical ceiling. GB/s is the 8 bytes per pixel any filter must move over the time.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
from figkit import Figure  # noqa: E402

BODY = (POST / "index.md").read_text(encoding="utf-8")
OUT = BODY[BODY.index("```text\ndevice:"):]
OUT = OUT[:OUT.index("```", 8)]
ROW = re.compile(r"^(copy \(ceiling\)|naive, global filter|constant filter|tiled \+ halo|separable, 2 passes)"
                 r"\s+([\d.]+)\s+([\d.]+)", re.MULTILINE)
MS, GBS = {}, {}
for k, block in zip((3, 7), OUT.split(" box filter")[1:]):
    assert block.count(" ok") == 5 and "MISMATCH" not in block and "exact copy" in block
    for name, ms, gbs in ROW.findall(block):
        MS[(k, name)], GBS[(k, name)] = float(ms), float(gbs)
assert len(MS) == 10
assert (MS[(3, "tiled + halo")], MS[(7, "tiled + halo")]) == (0.291, 0.617)
assert (MS[(3, "separable, 2 passes")], MS[(7, "separable, 2 passes")]) == (0.266, 0.293)
assert "beats constant memory by 17% (0.740 to 0.617 ms" in BODY and "2.5 times faster than constant memory" in BODY
for k in (3, 7):                                    # GB/s is 2 x 4 x 1920 x 1080 bytes over the time
    for name, ms in ((n, MS[(k, n)]) for n in ("naive, global filter", "separable, 2 passes")):
        assert abs(2 * 4 * 1920 * 1080 / (ms * 1e6) - GBS[(k, name)]) < 0.3
assert round(1 - MS[(7, "tiled + halo")] / MS[(7, "constant filter")], 2) == 0.17
assert round(MS[(3, "tiled + halo")] / MS[(3, "constant filter")], 1) == 1.4
assert round(MS[(7, "constant filter")] / MS[(7, "separable, 2 passes")], 1) == 2.5

BARS = [("naive", "naive, global filter", "ink"), ("constant", "constant filter", "memory"),
        ("tiled + halo", "tiled + halo", "memory"), ("separable", "separable, 2 passes", "memory")]
fig = Figure(
    "04-conv-perf", "The best fix depends on the filter size",
    "Two horizontal bar charts of effective bandwidth, 8 bytes per pixel over the time, on one scale of 0 to 160 GB/s, "
    "for a 1920 by 1080 image on the author's GTX 1650. 3 by 3 filter: naive "
    f"{GBS[(3, 'naive, global filter')]}, constant memory {GBS[(3, 'constant filter')]}, tiled "
    f"{GBS[(3, 'tiled + halo')]}, separable {GBS[(3, 'separable, 2 passes')]} GB/s, against a copy at "
    f"{GBS[(3, 'copy (ceiling)')]}. 7 by 7 filter: naive {GBS[(7, 'naive, global filter')]}, constant memory "
    f"{GBS[(7, 'constant filter')]}, tiled {GBS[(7, 'tiled + halo')]}, separable {GBS[(7, 'separable, 2 passes')]} "
    f"GB/s, against a copy at {GBS[(7, 'copy (ceiling)')]}.",
    subtitle="Measured on the author's GTX 1650, compute capability 7.5, CUDA 12.3, October 2026; 1920 × 1080 image.",
    data_w=True)

panels = fig.row(2, y=128, h=300)
for box, k in zip(panels, (3, 7)):
    body = fig.panel(box, f"{k} × {k} box filter")
    copy = GBS[(k, "copy (ceiling)")]
    fig.bar_chart(body, [(label, GBS[(k, key)], color) for label, key, color in BARS],
                  0, 160, [0, 40, 80, 120, 160], label_w=104, value_w=48, axis_label="effective GB/s",
                  fmt_tick=lambda v: f"{v:.0f}", fmt_value=lambda v: f"{v:.1f}",
                  ref_lines=[dict(x=copy, label=f"copy {copy:.1f}")])
fig.text(panels[0].x, 468, f"tiled: {MS[(3, 'tiled + halo')] / MS[(3, 'constant filter')]:.1f} times the time "
         "of constant memory", "label")
fig.text(panels[1].x, 468, f"separable: {MS[(7, 'constant filter')] / MS[(7, 'separable, 2 passes')]:.1f} times "
         "faster than constant memory", "label")
fig.caption("Caches absorb a 3 × 3 filter's re-reads; at 7 × 7 the tile and the separable form pay.")
fig.write()
