"""fusion_model.py — a CPU model of the memory traffic that fusion saves.

CUDA cannot run in this blog's test environment, so this NumPy model runs a chain
of elementwise ops two ways: separately (each op reads the whole tensor from
memory and writes the whole tensor back) and fused (read once, compute the chain
in registers, write once). It checks the results match and counts the global
memory traffic. For a chain of N ops the fused version moves 1/N of the bytes,
so the saving is (N-1)/N.  Run: python fusion_model.py  (needs NumPy)
"""
import numpy as np

M = 1024 * 4096                      # 4M elements = 16 MB in FP32
rng = np.random.default_rng(0)
x = rng.standard_normal(M).astype(np.float32)
bias, scale = np.float32(0.5), np.float32(2.0)

def gelu(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v ** 3)))

ops = ["+ bias", "* scale", "relu", "gelu"]
N = len(ops)

# separate: each op materializes a full intermediate tensor in memory
y = x + bias
y = y * scale
y = np.maximum(y, 0.0)
sep = gelu(y)

# fused: one pass, intermediates stay in registers
fused = gelu(np.maximum((x + bias) * scale, 0.0))

size_mb = M * 4 / 1e6
sep_traffic = N * 2 * size_mb        # each op: read full tensor + write full tensor
fused_traffic = 2 * size_mb          # read once + write once

print("fusion model")
print(f"  tensor              = {M:,} floats = {size_mb:.1f} MB")
print(f"  chain               = {' -> '.join(ops)}  ({N} ops)")
print(f"  fused == separate   : {bool(np.allclose(fused, sep, atol=1e-4))}")
print(f"  separate traffic    : {sep_traffic:.0f} MB  ({N} ops x read+write)")
print(f"  fused traffic       : {fused_traffic:.0f} MB  (read once, write once)")
print(f"  saving              : {(1 - fused_traffic / sep_traffic) * 100:.0f}%  (= (N-1)/N)")
