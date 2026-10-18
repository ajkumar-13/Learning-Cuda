"""fusion_model.py: a CPU model of what kernel fusion saves, and of the register limit on it.

Part 1 runs the elementwise chain of snippets/fusion.cu (bias, scale, ReLU, GELU) in NumPy two ways,
separately (each operation materializes a whole array, as each separate kernel writes one to global
memory) and fused (one expression per element), checks that the results are equal, and counts the
bytes each version moves to and from global memory for chains of 2, 3 and 4 operations: a separate
operation reads and writes the whole array, the fused kernel reads x once and writes z once, so the
ratio is the chain length and fusing saves (ops - 1) / ops of the traffic.

Part 2 applies post 10's occupancy rule for compute capability 7.5 (32 warp slots, at most 16 blocks
and 65,536 registers per SM in four quarters of 16,384 that each hold whole warps, registers allocated
per warp in units of 256) to 256-thread blocks at the register counts a fused kernel might reach.

Run from the series root:  uv run python posts/14-kernel-fusion/snippets/fusion_model.py
"""
import math

import numpy as np

N = 1 << 24                                   # 16,777,216 floats, as in fusion.cu
F = 4                                         # bytes per float
BIAS, SCALE = np.float32(0.5), np.float32(2.0)
OPS = ["+ bias", "* scale", "relu", "gelu"]


def gelu(v):
    """GELU, the tanh approximation, in float32 as fusion.cu computes it."""
    c, k, h, one = np.float32(0.7978845608), np.float32(0.044715), np.float32(0.5), np.float32(1.0)
    return h * v * (one + np.tanh(c * (v + k * v * v * v)))


def separate(x, ops):
    """One full array per operation, as one kernel per operation writes it to global memory."""
    y = x + BIAS
    y = y * SCALE
    if ops >= 3:
        y = np.maximum(y, np.float32(0.0))
    if ops >= 4:
        y = gelu(y)
    return y


def fused(x, ops):
    """The same chain as one expression per element: what one fused kernel's thread computes."""
    v = (x + BIAS) * SCALE
    if ops >= 3:
        v = np.maximum(v, np.float32(0.0))
    if ops >= 4:
        v = gelu(v)
    return v


def traffic(ops):
    """Bytes moved to and from global memory: separate (each op reads and writes N floats), fused."""
    return ops * 2 * N * F, 2 * N * F


# ---- part 2: post 10's occupancy rule for sm_75 ----
WARP_SLOTS, MAX_BLOCKS, REGS_PER_SM, QUARTERS, GRAIN = 32, 16, 65536, 4, 256


def occupancy(regs, threads=256):
    """(resident blocks per SM, resident warps, theoretical occupancy) for a kernel with no shared memory."""
    warps = math.ceil(threads / 32)
    per_warp = math.ceil(regs * 32 / GRAIN) * GRAIN
    by_regs = (REGS_PER_SM // QUARTERS) // per_warp * QUARTERS // warps
    blocks = min(WARP_SLOTS // warps, by_regs, MAX_BLOCKS)
    return blocks, blocks * warps, blocks * warps / WARP_SLOTS


REG_ROWS = [10, 12, 39, 64, 65, 96, 128, 255]  # fusion.cu's kernels (10, 12, 39 in -Xptxas=-v) and heavier ones


def main():
    x = np.random.default_rng(0).uniform(-4.0, 4.0, N).astype(np.float32)
    mb = N * F / 1e6
    print("1. traffic of the elementwise chain (derived)")
    print(f"  tensor  = {N:,} floats = {mb:.1f} MB")
    print(f"  chain   = {' -> '.join(OPS)}")
    print("  ops  separate (MB)  fused (MB)  separate / fused  saved by fusing  fused == separate")
    for ops in (2, 3, 4):
        sep, fus = traffic(ops)
        same = bool(np.array_equal(separate(x, ops), fused(x, ops)))
        print(f"  {ops:3d}  {sep / 1e6:13.1f}  {fus / 1e6:10.1f}  {sep / fus:15.1f}x  "
              f"{100 * (sep - fus) / sep:14.0f}%  {same}")
    print(f"  one separate bias, scale or ReLU: 1 FLOP per {2 * F} bytes, I = {1 / (2 * F):.3f} FLOP per byte")
    print("2. register pressure on sm_75, 256-thread blocks (derived, post 10's rule)")
    print("  regs/thread  regs/warp  blocks/SM  warps/SM  occupancy")
    for r in REG_ROWS:
        blocks, warps, occ = occupancy(r)
        print(f"  {r:11d}  {math.ceil(r * 32 / GRAIN) * GRAIN:9,d}  {blocks:9d}  {warps:8d}  {occ:9.0%}")
    top = max(r for r in range(1, 256) if occupancy(r)[2] == 1.0)
    print(f"  100% up to {top} registers per thread ({REGS_PER_SM:,} / 1,024 threads)")


if __name__ == "__main__":
    main()
