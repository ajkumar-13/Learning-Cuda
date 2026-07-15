# Matrix Addition on GPU

Implement a program that performs **element‑wise addition** of two
matrices containing 32‑bit floating point numbers on a GPU. The
program should take two input matrices of equal dimensions and
produce a single output matrix containing their element‑wise sum.

---

## Implementation Requirements

- External libraries are **not** permitted.
- The `solve` function signature must remain unchanged.
- The final result must be stored in matrix **`C`**.

---

## Examples

### Example 1

**Input**

- \(A = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}\)
- \(B = \begin{bmatrix} 5.0 & 6.0 \\ 7.0 & 8.0 \end{bmatrix}\)

**Output**

- \(C = A + B = \begin{bmatrix} 6.0 & 8.0 \\ 10.0 & 12.0 \end{bmatrix}\)

---

### Example 2

**Input**

- \(A = \begin{bmatrix}
     1.5 & 2.5 & 3.5 \\
     4.5 & 5.5 & 6.5 \\
     7.5 & 8.5 & 9.5
     \end{bmatrix}\)

- \(B = \begin{bmatrix}
     0.5 & 0.5 & 0.5 \\
     0.5 & 0.5 & 0.5 \\
     0.5 & 0.5 & 0.5
     \end{bmatrix}\)

**Output**

- \(C = A + B = \begin{bmatrix}
     2.0 & 3.0 & 4.0 \\
     5.0 & 6.0 & 7.0 \\
     8.0 & 9.0 & 10.0
     \end{bmatrix}\)

---

## Constraints

- Input matrices \(A\) and \(B\) have identical dimensions.
- Let \(N\) be the matrix size parameter (e.g. rows or columns):

     \[
     1 \leq N \leq 4096
     \]

- All elements are 32‑bit floating point numbers (`float`).