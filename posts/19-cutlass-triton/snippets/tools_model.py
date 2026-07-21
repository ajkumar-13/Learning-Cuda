"""tools_model.py — what the Triton matmul computes, and the abstraction ladder.

CUDA, Triton, and CUTLASS cannot run in this blog's test environment, so this
NumPy model reproduces the *exact* block-tiled accumulation that the Triton
kernel in triton_matmul.py performs (loop over K in BLOCK_K tiles, tl.dot each
tile into a register accumulator) and checks it against a reference product. It
then prints the developer-time-versus-performance ladder reported for a 4096
cubed FP16 GEMM on an A100, the point of the post: higher-level tools reach
90-100% of peak for a fraction of the effort.  Run: python snippets/tools_model.py  (NumPy)
"""
import numpy as np

M = N = K = 256
BLOCK_K = 32
rng = np.random.default_rng(0)
A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
ref = A @ B

# what Triton's `for k in range(0, K, BLOCK_K): acc += tl.dot(a, b)` does:
acc = np.zeros((M, N), dtype=np.float32)
for k in range(0, K, BLOCK_K):
    acc += A[:, k:k + BLOCK_K] @ B[k:k + BLOCK_K, :]    # one tile multiply, accumulated

# tool ladder: (name, dev time, % of cuBLAS peak, lines of code)  — A100, 4096^3 FP16
ladder = [
    ("cuBLAS",            "minutes", 100, "~1"),
    ("CUTLASS (tuned)",   "days",     98, "~50 (templates)"),
    ("Triton (autotuned)","hours",    95, "~30 (Python)"),
    ("Triton (naive)",    "hours",    90, "~30 (Python)"),
    ("raw CUDA (ours)",   "weeks",    60, "thousands"),
]

print("tools model")
print(f"  block-tiled matmul == reference : {bool(np.allclose(acc, ref, atol=1e-4))}")
print("  4096^3 FP16 GEMM on A100 (% of cuBLAS peak):")
print(f"    {'tool':22} {'dev time':9} {'%peak':>6}  code")
for name, t, pct, loc in ladder:
    print(f"    {name:22} {t:9} {pct:>5}%  {loc}")
print("  -> Triton reaches ~95% of cuBLAS for hours of work, not weeks")
