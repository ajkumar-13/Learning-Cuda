# Vector Addition on GPU

Implement a program that performs **element-wise addition** of two vectors containing 32-bit floating point numbers on a GPU. The program should take two input vectors of equal length and produce a single output vector containing their sum.

---

## Requirements

- The program must run the vector addition on the **GPU**.
- **External libraries are not permitted** (only standard CUDA / C++ APIs).
- The `solve` function signature **must remain unchanged**.
- The final result must be stored in **vector `C`**.

---

## Input / Output

- Input:
  - Vector `A` – length `N`, elements are 32-bit floating point numbers (`float`).
  - Vector `B` – length `N`, elements are 32-bit floating point numbers (`float`).
  - `A` and `B` have **identical lengths**.
- Output:
  - Vector `C` – length `N`, where `C[i] = A[i] + B[i]` for
    all indices `i = 0, 1, ..., N-1`.

---

## Examples

### Example 1

**Input**

- `A = [1.0, 2.0, 3.0, 4.0]`  
- `B = [5.0, 6.0, 7.0, 8.0]`

**Output**

- `C = [6.0, 8.0, 10.0, 12.0]`

Explanation:

- `C[0] = 1.0 + 5.0 = 6.0`  
- `C[1] = 2.0 + 6.0 = 8.0`  
- `C[2] = 3.0 + 7.0 = 10.0`  
- `C[3] = 4.0 + 8.0 = 12.0`

---

### Example 2

**Input**

- `A = [1.5, 1.5, 1.5]`  
- `B = [2.3, 2.3, 2.3]`

**Output**

- `C = [3.8, 3.8, 3.8]`

Explanation:

- `C[i] = 1.5 + 2.3 = 3.8` for all valid `i`.

---

## Constraints

- `A` and `B` have the **same length** `N`.
- `1 <= N <= 100000000`.
- Elements are 32-bit floating point values (`float`).
- The algorithm must:
  - Correctly handle large vectors up to the given limit.
  - Use GPU parallelism to perform the addition.