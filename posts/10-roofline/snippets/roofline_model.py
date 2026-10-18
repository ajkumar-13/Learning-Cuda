"""Roofline and occupancy model for the GTX 1650 with Max-Q Design (Turing, compute capability 7.5, sm_75).

Run from the series root:  uv run python posts/10-roofline/snippets/roofline_model.py

Standard-library Python: no GPU, no NumPy. Two kinds of number appear in its output, and it says which:
  derived   counted here from FLOP and bytes, or from the card's limits, so it prints the same on any machine;
  measured  timed on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026, by the harness
            of an earlier post and copied below as a constant with the post named; this script times nothing.

Five parts:
  1. the two peaks and the ridge point;
  2. arithmetic intensity I of the series' kernels, counted as the algorithm is written (every read goes to DRAM);
  3. the measured kernels against their roof;
  4. theoretical occupancy from block size, registers and shared memory;
  5. latency hiding as bytes in flight (Little's law), with an illustrative latency.
"""

import math

F = 4                                        # bytes per FP32 element

# ---- 1. the card (post 01's device facts; nvidia-smi -q -d CLOCK gives the maximum SM clock) ----
SMS = 14                                     # streaming multiprocessors
CORES = SMS * 64                             # 896 FP32 lanes
CLOCK_HZ = 1.740e9                           # maximum (boost) SM clock, nvidia-smi; the device query's 1,155 MHz is not it
DEVICE_QUERY_HZ = 1.155e9                    # cudaGetDeviceProperties' clockRate (post 01's query does not print it)
PEAK_FLOPS = CORES * CLOCK_HZ * 2            # one fused multiply-add = 2 FLOP per lane per cycle
PEAK_BW = 160e9                              # theoretical: 2 x 16 bytes x 5,001 MHz (post 01), the beta of the post
RIDGE = PEAK_FLOPS / PEAK_BW                 # FLOP per byte where the two roofs meet


def intensity(flops, bytes_moved):
    """Arithmetic intensity I in FLOP per byte."""
    return flops / bytes_moved


def ceiling(ai):
    """Attainable FLOP/s at intensity `ai`: the lower of the two roofs, min(pi, I x beta)."""
    return min(PEAK_FLOPS, ai * PEAK_BW)


# ---- 2. intensity as the algorithm is written: FLOP and DRAM bytes per unit of work ----
N = 1024                                     # matrix side, M = N = K, as post 03's table row
TILE = 16                                    # post 03's matmulTiled
KERNELS = [
    # name,                    flops,       bytes moved,                          what is counted
    ("vector add  C = A + B",  1,           3 * F,                                "1 add; read A, B, write C"),
    ("saxpy  y = a * x + y",   2,           3 * F,                                "1 multiply-add; read x, y, write y"),
    ("reduction  sum",         1,           1 * F,                                "1 add per element read"),
    ("matmul naive",           2 * N**3,    F * (2 * N**3 + N**2),                "2K loads per output, write C"),
    ("matmul tiled, TILE 16",  2 * N**3,    F * (2 * N**3 // TILE + N**2),        "loads divided by TILE"),
    ("matmul tiled, TILE 32",  2 * N**3,    F * (2 * N**3 // 32 + N**2),          "hypothetical: TILE 32"),
    ("matmul, each once",      2 * N**3,    F * 3 * N**2,                         "read A, B, write C once"),
]


# ---- 3. measured points: copied from the earlier posts' pasted tables, never timed here ----
MEASURED = [
    # name,                    I (derived),               GFLOP/s measured, source
    ("vector add",             intensity(1, 3 * F),       145.0 / 12,  "post 02, 10,000,000 elements, 145.0 GB/s"),
    ("reduction, shuffle",     intensity(1, F),           154.6 / 4,   "post 04, warp shuffle, 154.6 GB/s"),
    ("matmul naive",           intensity(*KERNELS[3][1:3]), 196.0,     "post 03, 1024 cubed"),
    ("matmul tiled, TILE 16",  intensity(*KERNELS[4][1:3]), 305.0,     "post 03, 1024 cubed"),
    ("cuBLAS",                 intensity(*KERNELS[6][1:3]), 2189.0,    "post 03, 1024 cubed; I at most"),
]
TRANSPOSE_GBS = 124.4                        # post 06, the padded transpose: no FLOP at all, I = 0


# ---- 4. occupancy: the sm_75 limits (Programming Guide, compute capability table) and allocation units ----
WARP_SLOTS = 32                              # 1,024 resident threads per SM / 32
MAX_BLOCKS = 16                              # resident blocks per SM
MAX_THREADS_PER_BLOCK = 1024
REGS_PER_SM = 65536
REG_QUARTERS = 4                             # the register file is four quarters, one per warp scheduler
REG_GRAIN = 256                              # registers are allocated per warp, rounded up to 256
SMEM_PER_SM = 65536                          # bytes (64 KiB)
SMEM_GRAIN = 256                             # shared memory is allocated per block, rounded up to 256 bytes


def occupancy(regs_per_thread, smem_per_block, block_threads):
    """Blocks per SM allowed by each limit, the result, the binding limit, resident warps and occupancy."""
    wb = math.ceil(block_threads / 32)                                  # warps per block
    if block_threads > MAX_THREADS_PER_BLOCK:
        return 0, 0, 0, 0, "block size", 0, 0.0
    by_warp = min(WARP_SLOTS // wb, MAX_BLOCKS)
    regs_per_warp = math.ceil(regs_per_thread * 32 / REG_GRAIN) * REG_GRAIN
    warps_by_reg = (REGS_PER_SM // REG_QUARTERS) // regs_per_warp * REG_QUARTERS
    by_reg = min(warps_by_reg // wb, MAX_BLOCKS)
    by_smem = (min(SMEM_PER_SM // (math.ceil(smem_per_block / SMEM_GRAIN) * SMEM_GRAIN), MAX_BLOCKS)
               if smem_per_block else MAX_BLOCKS)
    blocks = min(by_warp, by_reg, by_smem)
    limiter = min([("block size", by_warp), ("registers", by_reg), ("shared mem", by_smem)],
                  key=lambda kv: kv[1])[0]
    resident = blocks * wb
    return by_warp, by_reg, by_smem, blocks, limiter, resident, resident / WARP_SLOTS


OCC = [
    # name,                      regs, smem bytes, block; the series' kernels as nvcc -Xptxas=-v reports them
    ("vector add (02)",            12,      0, 256),
    ("matmul naive (03)",          49,      0, 256),
    ("matmul tiled 16 (03)",       39,   2048, 256),
    ("reduction, shuffle (04)",    15,     32, 256),
    ("transpose, padded (06)",     20,   4224, 256),
    # hypothetical launches, one per limit
    ("register-heavy",             96,      0, 256),
    ("over-tiled (smem)",          32,  49152, 256),
    ("TILE 32, 1,024 threads",     39,   8192, 1024),
    ("tiny blocks",                12,      0,  32),
]

# ---- 5. latency hiding: Little's law, bytes in flight = bandwidth x latency ----
LATENCY_CYCLES = 400                         # illustrative: "400+ cycles for memory" (Volkov, GTC 2010)
WARP_LOAD_BYTES = 32 * F                     # one coalesced load of one float per lane


def main():
    print("1. the two roofs of the GTX 1650 with Max-Q Design (derived)")
    print("  pi   = %d lanes x %.3f GHz x 2 FLOP = %s GFLOP/s FP32 peak"
          % (CORES, CLOCK_HZ / 1e9, format(round(PEAK_FLOPS / 1e9), ",")))
    print("         (at the device query's %.3f GHz it would read %s)"
          % (DEVICE_QUERY_HZ / 1e9, format(round(CORES * DEVICE_QUERY_HZ * 2 / 1e9), ",")))
    print("  beta = %d GB/s theoretical DRAM bandwidth" % (PEAK_BW / 1e9))
    print("  ridge point pi / beta = %.1f FLOP per byte" % RIDGE)
    print()
    print("2. arithmetic intensity, every read of the algorithm as written going to DRAM (derived)")
    print("  kernel                    I FLOP/byte  bound     ceiling GFLOP/s  % of pi")
    for name, flops, bytes_moved, _ in KERNELS:
        ai = intensity(flops, bytes_moved)
        cl = ceiling(ai)
        print("  %-24s %10.3f    %-7s  %13.1f  %6.1f%%"
              % (name, ai, "memory" if ai < RIDGE else "compute", cl / 1e9, 100 * cl / PEAK_FLOPS))
    naive, tiled = (intensity(*KERNELS[k][1:3]) for k in (3, 4))
    print("  tiling with TILE %d: %.3f -> %.3f FLOP per byte, %.1f times (%.2f -> %.2f counting loads only)"
          % (TILE, naive, tiled, tiled / naive, 2 / (2 * F), 2 * TILE / (2 * F)))
    print()
    print("3. measured on the author's GTX 1650 by earlier posts, against the roof (measured / derived)")
    print("  kernel                  I FLOP/byte  measured GFLOP/s  roof GFLOP/s  of roof  source")
    for name, ai, gf, src in MEASURED:
        roof = ceiling(ai) / 1e9
        print("  %-22s %10.3f  %16.1f  %12.1f  %6.0f%%  %s" % (name, ai, gf, roof, 100 * gf / roof, src))
    nv = MEASURED[2][2]
    print("  naive matmul runs %.1f times its roof: its DRAM intensity is at least %.3f FLOP per byte,"
          % (nv / (ceiling(naive) / 1e9), nv * 1e9 / PEAK_BW))
    print("    because the caches serve repeats the count sends to DRAM")
    print("  cuBLAS needs at least %.1f FLOP per byte to reach %s GFLOP/s at %d GB/s"
          % (MEASURED[4][2] * 1e9 / PEAK_BW, format(round(MEASURED[4][2]), ","), PEAK_BW / 1e9))
    print("  transpose, padded: I = 0, %.1f GB/s = %.0f%% of beta" % (TRANSPOSE_GBS, 100 * TRANSPOSE_GBS * 1e9 / PEAK_BW))
    print()
    print("4. theoretical occupancy on sm_75 (derived): %d warp slots, %d blocks, %s registers in %d quarters,"
          % (WARP_SLOTS, MAX_BLOCKS, format(REGS_PER_SM, ","), REG_QUARTERS))
    print("   %d KiB shared memory per SM; registers per warp rounded up to %d, shared memory per block to %d bytes"
          % (SMEM_PER_SM // 1024, REG_GRAIN, SMEM_GRAIN))
    print("  kernel                    regs   smem  block  byBlock byReg bySmem  blk/SM  binds on    occ")
    for name, regs, smem, blk in OCC:
        bw, br, bs, blocks, lim, res, occ = occupancy(regs, smem, blk)
        print("  %-24s %5d %6d  %5d    %5d %5d %6d  %6d  %-10s %4.0f%%"
              % (name, regs, smem, blk, bw, br, bs, blocks, lim, 100 * occ))
    print()
    print("5. latency hiding (illustrative latency of %d cycles at %.3f GHz)" % (LATENCY_CYCLES, CLOCK_HZ / 1e9))
    lat = LATENCY_CYCLES / CLOCK_HZ
    flight = PEAK_BW * lat
    per_sm = flight / SMS
    print("  bytes in flight to sustain beta = %d GB/s x %.1f ns = %s bytes, %s per SM"
          % (PEAK_BW / 1e9, lat * 1e9, format(round(flight), ","), format(round(per_sm), ",")))
    for ilp in (1, 2, 4):
        warps = math.ceil(per_sm / (ilp * WARP_LOAD_BYTES))
        print("  %d independent %d-byte load(s) per warp: %2d warps per SM, occupancy %3.0f%%"
              % (ilp, WARP_LOAD_BYTES, warps, 100 * warps / WARP_SLOTS))


if __name__ == "__main__":
    main()
