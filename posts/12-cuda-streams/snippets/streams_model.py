"""streams_model.py — a CPU model of the copy/compute pipeline.

CUDA cannot run in this blog's test environment, so this model simulates the
three-stage pipeline (host-to-device copy, compute, device-to-host copy) for N
chunks and reports the makespan and per-engine utilization, serial versus
overlapped. It assumes the three stages take equal time per chunk, the regime
where streams help most.

Serial runs every stage of every chunk end to end: 3*N stage-times. A 3-stage
pipeline overlaps them, finishing in (N + 2) stage-times once filled and drained.
Run: python streams_model.py
"""

def simulate(n, stage=1.0):
    # schedule each chunk on three engines; an engine is busy `stage` per chunk,
    # a chunk's stage k can't start until its stage k-1 done and that engine free.
    end = {"h2d": [0.0] * n, "comp": [0.0] * n, "d2h": [0.0] * n}
    free = {"h2d": 0.0, "comp": 0.0, "d2h": 0.0}
    order = ["h2d", "comp", "d2h"]
    for i in range(n):
        prev = 0.0
        for eng in order:
            start = max(prev, free[eng])
            end[eng][i] = start + stage
            free[eng] = end[eng][i]
            prev = end[eng][i]
    makespan = end["d2h"][n - 1]
    util = {e: (n * stage) / makespan for e in order}
    return makespan, util

N = 4
stage = 1.0
serial = 3 * N * stage
overlapped, util = simulate(N, stage)

print("streams model (3 equal stages: H2D, compute, D2H)")
print(f"  chunks (streams)   = {N}")
print(f"  serial makespan    = {serial:.0f} stage-times  (every stage end to end)")
print(f"  pipelined makespan = {overlapped:.0f} stage-times  (= N + 2 once filled)")
print(f"  speedup            = {serial / overlapped:.2f}x")
print(f"  engine utilization = H2D {util['h2d']*100:.0f}%, compute {util['comp']*100:.0f}%, D2H {util['d2h']*100:.0f}%")
big, ubig = simulate(64, stage)
print(f"  with 64 chunks     = {3*64*stage/big:.2f}x speedup, compute {ubig['comp']*100:.0f}% utilized (-> 3x, 100%)")
