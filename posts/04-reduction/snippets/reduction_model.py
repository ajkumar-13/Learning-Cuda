"""reduction_model.py — a CPU model of why addressing matters in a reduction.

CUDA cannot run in this blog's test environment, so this model reproduces the
two shared-memory reduction patterns over one block of 256 threads and counts
the *divergent warp-steps*: a (step, warp) pair where a warp has some lanes
active and some idle, which the hardware must serialize.

Interleaved addressing (tid % (2*stride) == 0) scatters the active threads, so
almost every warp is half-idle at every step. Sequential addressing (tid <
stride) keeps whole warps active, so divergence appears only in the final few
steps. It also checks the tree reduction equals the plain sum.

Run: python reduction_model.py  (needs NumPy)
"""
import numpy as np

BLOCK, WARP = 256, 32
rng = np.random.default_rng(0)
x = rng.random(BLOCK, dtype=np.float32)
ref = x.sum()

# tree reduction with sequential addressing, folding in half
a = x.copy()
stride, steps = BLOCK // 2, 0
while stride > 0:
    a[:stride] += a[stride:2 * stride]
    stride //= 2
    steps += 1

def divergent_warp_steps(active_at):
    """Sum over steps of warps that are partially (not fully/none) active."""
    total = 0
    for active in active_at:                       # set of active thread ids
        per_warp = {}
        for t in range(BLOCK):
            per_warp.setdefault(t // WARP, []).append(t in active)
        for flags in per_warp.values():
            if any(flags) and not all(flags):
                total += 1
    return total

interleaved = []
s = 1
while s < BLOCK:
    interleaved.append({t for t in range(BLOCK) if t % (2 * s) == 0})
    s *= 2
sequential = []
s = BLOCK // 2
while s > 0:
    sequential.append({t for t in range(BLOCK) if t < s})
    s //= 2

print("reduction model (block of 256, warp 32)")
print(f"  tree reduce == sum    : {bool(np.isclose(a[0], ref, atol=1e-2))}")
print(f"  parallel steps        : {steps}   (log2 of 256, vs 256 sequential adds)")
print(f"  interleaved divergence: {divergent_warp_steps(interleaved)} divergent warp-steps")
print(f"  sequential divergence : {divergent_warp_steps(sequential)} divergent warp-steps")
print("  -> sequential addressing keeps whole warps active; divergence only at the end")
