"""pipeline_model.py — a CPU model of software-pipelined tile loading.

CUDA cannot run in this blog's test environment, so this model computes the
real makespan of a tiled loop two ways and reports the speedup. It is the
verified artifact for this post: the .cu kernel ships as a compilable reference,
but the latency-hiding numbers quoted in the post come from here.

The loop has K iterations. Iteration k does two things:
  - load[k]:    fetch tile k from global memory  (takes load_t time units)
  - compute[k]: do the math on tile k            (takes comp_t time units)

(a) Synchronous (post 03's tiled matmul): load[k] and compute[k] cannot
    overlap, because compute[k] reads the very tile load[k] just wrote and a
    __syncthreads() sits between them. Makespan is the full sum:
        T_sync = sum_k (load_t + comp_t) = K * (load_t + comp_t)

(b) Double-buffered / pipelined (cp.async): load[k+1] is prefetched into the
    other buffer WHILE compute[k] runs, so each steady-state step costs only the
    slower of the two. With a one-tile prologue and a one-tile epilogue:
        T_pipe = load_0  +  sum_{k=0..K-2} max(load_{k+1}, comp_k)  +  comp_{last}

Both are exact integer/float arithmetic. No timing, no invented bandwidth.
Run: python pipeline_model.py
"""


def makespans(loads, comps):
    """Return (synchronous, pipelined) makespan for per-tile load/compute times."""
    assert len(loads) == len(comps)
    k = len(loads)

    # (a) synchronous: every load and every compute serialized, end to end.
    t_sync = sum(loads) + sum(comps)

    # (b) pipelined: prologue loads tile 0; in steady state, prefetch of tile
    # k+1 overlaps compute of tile k, so the step costs max(load[k+1], comp[k]);
    # epilogue computes the last tile with nothing left to prefetch.
    t_pipe = loads[0]
    for kk in range(k - 1):
        t_pipe += max(loads[kk + 1], comps[kk])
    t_pipe += comps[k - 1]
    return t_sync, t_pipe


def report(name, load_t, comp_t, k=8):
    loads = [load_t] * k
    comps = [comp_t] * k
    t_sync, t_pipe = makespans(loads, comps)
    speedup = t_sync / t_pipe
    # Steady-state hidden time: how much of each load disappears behind compute.
    hidden = min(load_t, comp_t)
    print(f"  {name}")
    print(f"    per tile          : load={load_t}, compute={comp_t}  (K={k} tiles)")
    print(f"    synchronous        = {t_sync}  units   (sum of all loads + computes)")
    print(f"    pipelined          = {t_pipe}  units   (load_0 + overlap + compute_last)")
    print(f"    speedup            = {speedup:.2f}x")
    print(f"    hidden per step    = {hidden} units of load tucked behind compute")


if __name__ == "__main__":
    print("software-pipelining model (double-buffered tile loop)\n")

    # Case 1: load == compute. Each step costs max(L, C) = one unit instead of
    # two, so the loop nearly halves -> ~2x in the large-K limit.
    report("balanced  (load == compute)", load_t=10, comp_t=10)
    print()

    # Case 2: compute-bound. The load is shorter than the compute, so the whole
    # load hides: pipelined makespan is dominated by compute, ~no load cost.
    report("compute-bound (load <  compute)", load_t=4, comp_t=10)
    print()

    # Case 3: memory-bound. The load is longer than the compute, so compute
    # hides instead; you still win, but the load latency now sets the pace.
    report("memory-bound  (load >  compute)", load_t=10, comp_t=4)
    print()

    # The asymptotic rule, stated exactly for the balanced case as K grows.
    for k in (8, 64, 1024):
        ts, tp = makespans([10] * k, [10] * k)
        print(f"  balanced, K={k:>4}: speedup = {ts / tp:.4f}x  "
              f"(-> 2.0000x as K -> infinity)")
