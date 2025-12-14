# ReLU Activation on a Vector

Implement a program that performs the **Rectified Linear Unit (ReLU)**
activation function on a vector of 32‑bit floating point numbers. The
ReLU function sets all negative values to zero and leaves positive
values unchanged:

\[
\operatorname{ReLU}(x) = \max(0, x)
\]

---

## Implementation Requirements

- External libraries are **not** permitted (only standard C/CUDA/C++ APIs).
- The `solve` function signature **must remain unchanged**.
- The final result must be stored in the vector **`output`**.

---

## Examples

### Example 1

**Input**

- `input = [-2.0, -1.0, 0.0, 1.0, 2.0]`

**Output**

- `output = [0.0, 0.0, 0.0, 1.0, 2.0]`

Explanation:

- Negative values (−2.0, −1.0) become 0.0.
- Non‑negative values (0.0, 1.0, 2.0) stay the same.

---

### Example 2

**Input**

- `input = [-3.5, 0.0, 4.2]`

**Output**

- `output = [0.0, 0.0, 4.2]`

Explanation:

- −3.5 becomes 0.0, while 0.0 and 4.2 are unchanged.

---

## Constraints

- Let `N` be the length of the input vector.
- The constraints are:

	\[
	1 \leq N \leq 100{,}000{,}000
	\]

Your implementation should correctly apply `ReLU` to all `N` elements
and write the final values into `output`.

