"""pipeline_model.py: a CPU model of a software-pipelined tile loop.

The loop walks n tiles. Tile t needs two things:
  - load[t]:    fetch tile t from global memory  (load_t time units)
  - compute[t]: the multiply-adds on tile t      (comp_t time units)

(a) Synchronous (post 03's tiled matmul): load[t] and compute[t] cannot overlap, because compute[t]
    reads the very tile load[t] just wrote and a __syncthreads() sits between them:
        T_sync = sum over t of (load_t + comp_t)
(b) Double-buffered (pipelined): load[t + 1] goes into the other buffer while compute[t] runs, so each
    steady-state step costs only the slower of the two. With a one-tile prologue and a one-tile epilogue:
        T_pipe = load_0 + sum over t = 0 .. n - 2 of max(load_{t+1}, comp_t) + comp_{n-1}

Both are exact arithmetic in abstract time units: the model assumes the overlap is perfect and costs
nothing, and measures nothing. Post 15's async_pipeline.cu measures a real kernel; the last lines below
read one of its measured speed-ups back through the model.
Run: uv run python posts/15-async-copy-pipelining/snippets/pipeline_model.py   (from the series root)
"""


def makespans(loads, comps):
    """Return (synchronous, pipelined) makespan for per-tile load and compute times."""
    assert len(loads) == len(comps) and loads
    n = len(loads)
    t_sync = sum(loads) + sum(comps)              # every load and every compute in series
    t_pipe = loads[0]                             # prologue: tile 0's load, nothing to compute yet
    for t in range(n - 1):                        # steady state: load t + 1 alongside compute t
        t_pipe += max(loads[t + 1], comps[t])
    t_pipe += comps[n - 1]                        # epilogue: the last compute, nothing left to load
    return t_sync, t_pipe


def row(name, load_t, comp_t, tiles):
    t_sync, t_pipe = makespans([load_t] * tiles, [comp_t] * tiles)
    print(f"  {name:<14}{load_t:>5}{comp_t:>9}{tiles:>7}{t_sync:>13}{t_pipe:>11}{t_sync / t_pipe:>9.2f}x")


def compute_share(speedup, tiles):
    """The compute time, as a share of the load time, at which a loop of `tiles` equal tiles whose
    load is the longer stage gains `speedup`: T_sync / T_pipe = n (L + C) / (n L + C), solved for C / L."""
    return tiles * (speedup - 1) / (tiles - speedup)


if __name__ == "__main__":
    print("software-pipelining model (double-buffered tile loop), times in model units")
    print(f"  {'case':<14}{'load':>5}{'compute':>9}{'tiles':>7}{'synchronous':>13}{'pipelined':>11}{'speedup':>10}")
    row("balanced", 10, 10, 8)
    row("compute-bound", 4, 10, 8)
    row("memory-bound", 10, 4, 8)
    # The limit: for equal tiles the speedup tends to (L + C) / max(L, C), 2 when the stages are balanced.
    for n in (64, 1024):
        row("balanced", 10, 10, n)

    # async_pipeline.cu at M = N = K = 1024 walks 1024 / 16 = 64 tiles. Its register-staged pipeline ran
    # 1.61 times faster than the synchronous kernel with one block per SM (post 15, section 5: the
    # author's GTX 1650, October 2026). Read through the model, assuming perfect overlap:
    measured, tiles = 1.61, 1024 // 16
    share = compute_share(measured, tiles)
    t_sync, t_pipe = makespans([1.0] * tiles, [share] * tiles)
    assert abs(t_sync / t_pipe - measured) < 1e-9
    print(f"  a measured {measured:.2f}x over {tiles} tiles: the model's speedup when compute takes "
          f"{share:.2f} of a load's time")
    print(f"  at that share many tiles tend to {(1 + share):.2f}x; the fit assumes a perfect overlap, so it cannot test one")
