"""Post 03, hero: one output element of C is the dot product of one row of A and one column of B.

Run from the series root:  uv run python posts/03-matrix-multiplication/diagrams/src/01-dot-product.py
Writes posts/03-matrix-multiplication/diagrams/01-dot-product.svg.
cuda_fig_design.md section 3: matrices as grids of cells with their shapes in the notation guide's form, the data
green (memory), the one thread orange (compute). The matrices are drawn small (A: [5, 8], B: [8, 6]) to show the
pattern; the kernel line in the card is copied from snippets/matmul.cu and asserted, and the per-output count
(K multiply-adds, 2K loads) is the one snippets/matmul_model.py prints (128 loads for K = 64).
"""
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
POST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POST / "snippets"))
from figkit import Box, Figure, isub, rich, sigma  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import matmul_model as mm  # noqa: E402

CU = (POST / "snippets" / "matmul.cu").read_text(encoding="utf-8")
LINE = "sum += A[row * K + k] * B[k * N + col];"
assert LINE in CU and "for (int k = 0; k < K; ++k)" in CU and "C[row * N + col] = sum;" in CU
assert mm.naive_loads // (mm.M * mm.N) == 2 * mm.K == 128

M, K, N = 5, 8, 6            # drawn sizes only
R, C = 2, 4                  # the worked thread's (row, col)
CELL = 24

fig = Figure(
    "01-dot-product", "One thread, one output: a row of A meets a column of B",
    f"Three grids in the multiply layout: A, shape M by K, drawn {M} by {K} at the lower left; B, shape K by N, drawn "
    f"{K} by {N} above C; C, shape M by N, drawn {M} by {N} at the lower right. Row {R} of A and column {C} of B are "
    f"shaded green, and dashed leaders run from them to cell ({R}, {C}) of C, shaded orange: the one thread that "
    "computes it. A card on the right shows that thread's loop line from matmul.cu, sum += A[row * K + k] * "
    "B[k * N + col], and the sum C sub ij equals the sum over k of A sub ik times B sub kj: K multiply-adds and "
    "2K loads per output element.",
    subtitle="One thread per element of C; the grid of threads covers C. Shapes drawn small.",
    data_w=True)

AX, AY = 152, 336
BX, BY = AX + K * CELL + 24, 120
gB = fig.grid(BX, BY, K, N, cell=CELL, fill=lambda i, j: "memory-soft" if j == C else None)
gA = fig.grid(AX, AY, M, K, cell=CELL, fill=lambda i, j: "memory-soft" if i == R else None)
gC = fig.grid(BX, AY, M, N, cell=CELL, fill=lambda i, j: "compute" if (i, j) == (R, C) else None)
gA.window(R, 0, 1, K, "memory")
gB.window(0, C, K, 1, "memory")
gC.window(R, C, 1, 1, "compute")

fig.text(AX - 16, AY + 20, "A", "code", anchor="end")
fig.text(AX - 16, AY + 44, "[M, K]", "code", anchor="end")
fig.text(BX - 16, BY + 20, "B", "code", anchor="end")
fig.text(BX - 16, BY + 44, "[K, N]", "code", anchor="end")
fig.text(BX + N * CELL + 16, AY + M * CELL - 28, "C", "code")
fig.text(BX + N * CELL + 16, AY + M * CELL - 4, "[M, N]", "code")
fig.text(AX - 16, AY + R * CELL + 17, f"row {R}", "tick", anchor="end")
fig.text(BX + C * CELL + 12, BY - 8, f"col {C}", "tick", anchor="middle")

cell = gC.cell(R, C)
fig.leader((AX + K * CELL, cell.cy), (cell.x, cell.cy))
fig.leader((cell.cx, BY + K * CELL), (cell.cx, cell.y))

card = fig.card(560, 120, 360, None, heading=f"thread (row, col) = ({R}, {C})",
                lines=["for (int k = 0; k < K; ++k)", "  " + LINE, "C[row * N + col] = sum;"],
                style="code13", color="compute", fit=True)
fig.text(560, card.bottom + 64,
         rich(isub("C", "ij"), " = ", sigma("k"), " ", isub("A", "ik"), " ", isub("B", "kj")), "math")
fig.note(Box(560, card.bottom + 88, 360, 0), ["K multiply-adds and 2K loads from", "global memory per element of C"])
fig.caption("The grid supplies the loop over (row, col); each thread runs only the loop over k.")
fig.write()
