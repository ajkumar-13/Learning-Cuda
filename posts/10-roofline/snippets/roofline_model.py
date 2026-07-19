"""Roofline + occupancy model for the GTX 1650 (Turing, sm_75).

Pure standard-library Python: no GPU, no NumPy. Every number it prints is
*derived* from counting floating-point operations and bytes, or from the
published sm_75 resource limits, so it reproduces identically on any machine.

Two parts:
  1. Arithmetic intensity and the roofline: where each kernel of the series
     lands, and the performance ceiling that intensity implies.
  2. Occupancy: how many warps stay resident, and which of the three
     per-SM resources runs out first.
"""

import math

# ---- GTX 1650 (Turing TU117, compute capability 7.5) published peaks ----
CORES      = 896                            # 14 SMs x 64 FP32 cores (sm_75)
CLOCK_HZ   = 1.155e9                         # max clock of this Max-Q card (device_query)
PEAK_FLOPS = CORES * CLOCK_HZ * 2          # 2 FLOP per FMA -> ~2.07 TFLOP/s
PEAK_BW    = 160e9                           # bytes/s: 5001 MHz GDDR6 x 128-bit = 160 GB/s
RIDGE      = PEAK_FLOPS / PEAK_BW           # FLOP/byte where the roofs cross
F = 4                                       # bytes per FP32 element


def intensity(flops, bytes_moved):
    return flops / bytes_moved


def ceiling(ai):
    """Attainable FLOP/s at arithmetic intensity `ai`: the lower roof."""
    return min(PEAK_FLOPS, ai * PEAK_BW)


# Per-kernel FLOP and byte counts. Matmuls use N = 1024; the tile width is 32.
N, T = 1024, 32
KERNELS = [
    # name,                 flops,        bytes moved
    ("vector-add  C=A+B",   1,            3 * F),                 # 1 add,  read A,B write C
    ("saxpy  y=a*x+y",      2,            3 * F),                 # mul+add, read x,y write y
    ("reduction  sum",      1,            1 * F),                 # 1 add per element read
    ("matmul naive",        2 * N**3,     F * (2 * N**3 + N**2)), # re-reads A row, B col
    ("matmul tiled 32x32",  2 * N**3,     F * (2 * N**3 // T + N**2)),
]

print("roofline model  (GTX 1650: %.2f TFLOP/s FP32 peak, %d GB/s, ridge %.1f FLOP/byte)"
      % (PEAK_FLOPS / 1e12, PEAK_BW / 1e9, RIDGE))
print()
print("kernel                  AI FLOP/byte   bound     ceiling GFLOP/s   %% of FP32 peak")
print("-" * 80)
for name, flops, bytes_moved in KERNELS:
    ai = intensity(flops, bytes_moved)
    ceil = ceiling(ai)
    bound = "memory " if ai < RIDGE else "compute"
    print("%-22s %8.3f      %s     %10.1f        %5.1f%%"
          % (name, ai, bound, ceil / 1e9, 100 * ceil / PEAK_FLOPS))
print()
print("tiling raised matmul intensity %.3f -> %.1f FLOP/byte (%.0fx), sliding it toward the ridge"
      % (intensity(2 * N**3, F * (2 * N**3 + N**2)),
         intensity(2 * N**3, F * (2 * N**3 // T + N**2)), T))

# ---- Occupancy: the three per-SM limiters on sm_75 ----
WARP_SLOTS   = 32          # 1024 threads / 32
MAX_BLOCKS   = 16
REGS_PER_SM  = 65536
SMEM_PER_SM  = 65536       # bytes (64 KiB)
REG_GRAIN    = 256         # per-warp register allocation granularity


def occupancy(regs_per_thread, smem_per_block, block_threads):
    wb = math.ceil(block_threads / 32)                      # warps per block
    by_warp = WARP_SLOTS // wb
    regs_per_warp = math.ceil(regs_per_thread * 32 / REG_GRAIN) * REG_GRAIN
    by_reg = REGS_PER_SM // (regs_per_warp * wb)
    by_smem = SMEM_PER_SM // smem_per_block if smem_per_block else MAX_BLOCKS
    blocks = min(by_warp, by_reg, by_smem, MAX_BLOCKS)
    limiter = min([("block size", by_warp), ("registers", by_reg),
                   ("shared mem", by_smem)], key=lambda kv: kv[1])[0]
    resident = min(blocks * wb, WARP_SLOTS)
    return by_warp, by_reg, by_smem, blocks, limiter, resident, resident / WARP_SLOTS


OCC = [
    # name,             regs, smem,  block
    ("vector-add lean",   32,     0, 256),
    ("register-heavy",    96,     0, 256),
    ("tiled matmul 32x32",40,  8192, 256),
    ("over-tiled (smem)", 32, 49152, 256),
    ("tiny blocks",       32,     0,  64),
]

print()
print("occupancy model  (sm_75: 32 warp slots, 64K regs, 64 KiB smem, 16 blocks/SM)")
print()
print("kernel               regs  smem   blk  byWarp byReg bySmem  blk/SM  binds on    occ%%")
print("-" * 84)
for name, regs, smem, blk in OCC:
    bw, br, bs, blocks, lim, res, occ = occupancy(regs, smem, blk)
    print("%-20s %4d %6d  %4d   %5d %5d %6d  %5d   %-10s %4.0f%%"
          % (name, regs, smem, blk, bw, br, bs, blocks, lim, 100 * occ))
