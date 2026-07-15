# Matrix Transpose on GPU

Implement a program that **transposes a matrix** of 32‑bit floating
point numbers on a GPU. The transpose of a matrix switches its rows
and columns.

If the input matrix \(A\) has dimensions \(\text{rows} \times \text{cols}\),
then its transpose \(A^T\) has dimensions \(\text{cols} \times
	ext{rows}\).

All matrices are stored in **row‑major** order.

---

## Implementation Requirements

- Use only native features (no external libraries).
- The `solve` function signature **must remain unchanged**.
- The final result must be stored in the matrix **`output`**.

---

## Examples

### Example 1

**Input:** 2×3 matrix \(A\)

\[
A = \begin{bmatrix}
1.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0
\end{bmatrix}
\]

**Output:** 3×2 matrix \(A^T\)

\[
output = A^T = \begin{bmatrix}
1.0 & 4.0 \\
2.0 & 5.0 \\
3.0 & 6.0
\end{bmatrix}
\]

---

### Example 2

**Input:** 3×1 matrix \(A\)

\[
A = \begin{bmatrix}
1.0 \\
2.0 \\
3.0
\end{bmatrix}
\]

**Output:** 1×3 matrix \(A^T\)

\[
output = A^T = \begin{bmatrix}
1.0 & 2.0 & 3.0
\end{bmatrix}
\]

---

## Constraints

- \(1 \leq \text{rows}, \text{cols} \leq 8192\)
- Input matrix dimensions: \(\text{rows} \times \text{cols}\)
- Output matrix dimensions: \(\text{cols} \times \text{rows}\)
- All elements are 32‑bit floating point numbers (`float`).

