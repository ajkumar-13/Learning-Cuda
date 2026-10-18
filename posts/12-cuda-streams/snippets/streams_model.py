"""streams_model.py: a CPU model of the copy and compute pipeline of post 12.

A workload of three stages per chunk (host-to-device copy, compute, device-to-host copy) is scheduled on the
GPU's engines: a stage of chunk i starts when the stage before it in chunk i has finished and its engine is free,
and each engine takes its stages in the order the host issued them, as Harris's version 1 loop issues them.

Part 1 is the textbook case: three equal stages and a copy engine for each direction. Serial runs every stage
of every chunk end to end, 3 * N stage-times; the pipeline finishes in N + 2 stage-times once filled and drained.

Part 2 feeds the model the stage times that snippets/streams.cu measured on the author's GTX 1650 (compute
capability 7.5), CUDA 12.3, on 2026-10-18 (constants below), for its two kernels, and predicts the pipelined time
for 1 to 32 chunks under two assumptions about the bus: the two copy directions run at full speed at the same
time (two copy engines, a full-duplex link), or one copy runs at a time (the copies share one engine). The
per-chunk times are the measured whole-array times divided by the number of chunks; the model charges nothing
for issuing an operation.

Run from the series root: uv run python posts/12-cuda-streams/snippets/streams_model.py
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


def pipeline(n, t_in, t_kernel, t_out, shared_copy):
    """Makespan of n chunks with whole-array stage times t_in, t_kernel, t_out (ms).

    shared_copy=False: a copy engine for each direction, each taking its copies in issue order.
    shared_copy=True: one copy at a time; whenever the copy engine is free it takes, of the copies that are ready
    (a copy in is ready at once, a copy out when its chunk's kernel has finished), the one issued first.
    Returns the makespan and the (start, end) of every stage of every chunk."""
    a, k, b = t_in / n, t_kernel / n, t_out / n
    spans = [dict() for _ in range(n)]
    kernel_free = 0.0

    def run_kernel(i):
        nonlocal kernel_free
        start = max(spans[i]["in"][1], kernel_free)
        kernel_free = start + k
        spans[i]["kernel"] = (start, kernel_free)

    if not shared_copy:
        in_free = out_free = 0.0
        for i in range(n):
            spans[i]["in"] = (in_free, in_free + a)
            in_free += a
            run_kernel(i)
            start = max(spans[i]["kernel"][1], out_free)
            out_free = start + b
            spans[i]["out"] = (start, out_free)
    else:
        t, next_in, next_out = 0.0, 0, 0               # copies in and out each leave in chunk order
        while next_out < n:
            out_ready = spans[next_out]["kernel"][1] if next_out < next_in else None
            # issue order is in 0, out 0, in 1, out 1, ...: a ready copy out goes before the next copy in
            if out_ready is not None and (out_ready <= t or next_in == n):
                start = max(t, out_ready)
                spans[next_out]["out"] = (start, start + b)
                t, next_out = start + b, next_out + 1
            else:
                spans[next_in]["in"] = (t, t + a)
                t, next_in = t + a, next_in + 1
                run_kernel(next_in - 1)
    return max(s["out"][1] for s in spans), spans


# Measured by snippets/streams.cu on 2026-10-18, pinned memory, fastest of 5 passes (ms): both input copies
# (2 x 64 MiB), each kernel alone over 2^24 floats, the result copy (64 MiB), and the serial pinned rows.
MEASURED = {
    "vector add": dict(t_in=11.01, t_kernel=1.37, t_out=6.79, serial=18.95),
    "64 extra rounds": dict(t_in=11.01, t_kernel=9.44, t_out=6.79, serial=26.62),
}

if __name__ == "__main__":
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

    print()
    print("measured stage times on the GTX 1650 (ms), and the pipelined time the model predicts")
    for name, m in MEASURED.items():
        total = m["t_in"] + m["t_kernel"] + m["t_out"]
        print(f"  {name}: copy in {m['t_in']:.2f}, kernel {m['t_kernel']:.2f}, copy out {m['t_out']:.2f}; "
              f"stages end to end {total:.2f}")
        print("    chunks   both directions at once   one copy at a time")
        for n in (1, 2, 4, 8, 16, 32):
            duplex, _ = pipeline(n, m["t_in"], m["t_kernel"], m["t_out"], shared_copy=False)
            single, _ = pipeline(n, m["t_in"], m["t_kernel"], m["t_out"], shared_copy=True)
            print(f"    {n:6d}   {duplex:8.2f} ms  {total / duplex:5.2f}x      {single:8.2f} ms  {total / single:5.2f}x")
