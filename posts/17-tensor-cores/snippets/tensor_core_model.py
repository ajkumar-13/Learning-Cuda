"""tensor_core_model.py — a CPU model of Tensor Core throughput and mixed precision.

CUDA cannot run in this blog's test environment, so this NumPy model shows the
two ideas behind Tensor Cores:

1. Density. A CUDA core does one fused multiply-add per thread, 32 per warp per
   cycle. A Tensor Core does a whole 16x16x16 matrix multiply-accumulate per warp
   instruction: 16*16*16 = 4096 FMAs, 128x more.
2. Mixed precision. Multiplying in FP16 is fast but FP16 has only ~3 digits and a
   max of 65,504. Accumulating a long dot product in FP16 loses accuracy; doing
   the multiply in FP16 but the accumulate in FP32 stays close to full FP32. This
   reproduces both and measures the error against a float64 reference.

Run: python tensor_core_model.py  (needs NumPy)
"""
import numpy as np

cuda_fmas = 32
tc_fmas = 16 * 16 * 16

K = 4096
rng = np.random.default_rng(0)
A = (rng.standard_normal((16, K)) * 0.1).astype(np.float32)
B = (rng.standard_normal((K, 16)) * 0.1).astype(np.float32)
ref = A.astype(np.float64) @ B.astype(np.float64)          # ground truth

A16, B16 = A.astype(np.float16), B.astype(np.float16)

# pure FP16: accumulate the dot products in FP16
acc16 = np.zeros((16, 16), np.float16)
for k in range(K):
    acc16 = (acc16 + A16[:, [k]] * B16[[k], :]).astype(np.float16)

# mixed: FP16 inputs, FP32 accumulate
acc32 = np.zeros((16, 16), np.float32)
for k in range(K):
    acc32 += A16[:, [k]].astype(np.float32) * B16[[k], :].astype(np.float32)

err16 = float(np.max(np.abs(acc16.astype(np.float64) - ref)))
errmix = float(np.max(np.abs(acc32.astype(np.float64) - ref)))

print("tensor core model")
print(f"  FMAs per warp instruction: CUDA core {cuda_fmas}, Tensor Core {tc_fmas} ({tc_fmas // cuda_fmas}x)")
print(f"  FP16 range: max 65504, ~3 decimal digits")
print(f"  dot products of length K={K}, max abs error vs FP64 reference:")
print(f"    pure FP16 accumulate   : {err16:.5f}")
print(f"    FP16 mul + FP32 accum   : {errmix:.6f}  ({err16/errmix:.0f}x more accurate)")
print("  -> mixed precision keeps Tensor Core speed with near-FP32 accuracy")
