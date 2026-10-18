"""Launch-overhead and allocation model for CUDA Graphs and stream-ordered allocation (post 13).

Standard-library Python, no GPU. The model is fed with costs measured by snippets/graphs.cu; every number it
prints is derived from those inputs, so it prints the same on any machine. The inputs are one run of graphs.cu on
the author's machine (below); a reader who runs graphs.cu on another machine pastes that run's numbers here.

Two parts:
  1. Launch overhead. A kernel of device time W costs L + W in a stream of direct launches and l + W inside a
     replayed graph (l < L), plus the one-time cost of building the graph. The model fits L and l to the measured
     sweep (the mean of the measured time minus W), checks the fit against each row, then derives the speedup
     for any W, its limit as W -> 0, the W where it falls to 1.1x, and the number of kernels after which a
     graph has paid for its build.
  2. Allocation. The per-step cost of allocate, launch, free and synchronize with cudaMalloc and with the
     stream-ordered pool, and what each costs over a loop of steps.
"""

# ---- Inputs: measured by snippets/graphs.cu on 2026-10-18 ----
# Machine: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), CUDA 12.3, driver 545.84,
# Windows 10, Intel Core i7-9750H. Median of 7 passes; microseconds per kernel; graph of 20 kernels.
# W (us): (direct issue, direct done, graph issue, graph done)
SWEEP = {
    0: (1.71, 1.80, 0.28, 0.77),
    2: (1.75, 3.52, 0.28, 2.81),
    4: (1.76, 5.54, 0.29, 4.79),
    8: (1.76, 9.33, 0.29, 8.79),
    16: (1.78, 17.43, 0.28, 16.83),
    32: (1.90, 33.59, 0.32, 32.86),
}
NODES = 20            # kernels per graph
BUILD = 59.9          # us: capture 20 kernels and instantiate, once
# Allocation, 4 MiB, us per step of allocate, kernel, free, cudaStreamSynchronize.
STEP_MALLOC = 519.32  # cudaMalloc + cudaFree
STEP_POOL_0 = 2004.06  # cudaMallocAsync + cudaFreeAsync, release threshold 0 (the default)
STEP_POOL_MAX = 45.81  # cudaMallocAsync + cudaFreeAsync, release threshold maximum
FREE_BUSY = 20343.4   # us: cudaFree while a kernel runs on another stream
BUSY_KERNEL = 20600.0  # us: that kernel
ASYNC_BUSY = 9.2      # us: poolRoundTrip (allocate, launch, free) while the same kernel runs, pool kept


def fit():
    """L and l: the mean per-kernel cost above W over the measured rows, direct and in a graph."""
    ws = sorted(SWEEP)
    big_l = sum(SWEEP[w][1] - w for w in ws) / len(ws)
    small_l = sum(SWEEP[w][3] - w for w in ws) / len(ws)
    return big_l, small_l


L, ELL = fit()


def direct(n, w):
    """n kernels of device time w, launched one by one."""
    return n * (L + w)


def graph(n, w):
    """The same n kernels replayed from a graph built once."""
    return BUILD + n * (ELL + w)


def speedup(n, w):
    return direct(n, w) / graph(n, w)


def main():
    print("inputs: graphs.cu on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, driver 545.84,")
    print("        Intel Core i7-9750H, Windows 10, 2026-10-18; graph of %d kernels; build %.1f us" % (NODES, BUILD))
    print()
    print("fit: per-kernel cost above W, direct L = %.2f us, in a graph l = %.2f us" % (L, ELL))
    print()
    print("  W (us)   measured direct   model   measured graph   model   model speedup")
    for w in sorted(SWEEP):
        _, d, _, g = SWEEP[w]
        print("  %6d   %15.2f   %5.2f   %14.2f   %5.2f   %12.2fx"
              % (w, d, L + w, g, ELL + w, (L + w) / (ELL + w)))
    print()

    n = 10000
    print("derived: %d kernels, direct against one graph built once and replayed" % n)
    print("  W (us)   direct (ms)   graph (ms)   speedup")
    for w in [0, 2, 5, 10, 100]:
        print("  %6d   %11.2f   %10.2f   %6.2fx" % (w, direct(n, w) / 1000, graph(n, w) / 1000, speedup(n, w)))
    base = ELL + BUILD / n
    w_11 = (L - 1.1 * base) / 0.1                       # solve (L + W) / (base + W) = 1.1
    print("  speedup as W -> 0                   : %.2fx" % (L / base))
    print("  W where the speedup falls to 1.1x   : %.1f us" % w_11)
    print("  kernels until the graph repays its build: %.0f (%.1f replays of %d)"
          % (BUILD / (L - ELL), BUILD / (L - ELL) / NODES, NODES))
    issue_direct = sorted(v[0] for v in SWEEP.values())[len(SWEEP) // 2]
    issue_graph = sorted(v[2] for v in SWEEP.values())[len(SWEEP) // 2]
    print("  host issue time per kernel (median) : %.2f us direct, %.2f us from a graph (%.1fx less)"
          % (issue_direct, issue_graph, issue_direct / issue_graph))
    print()

    steps = 1000
    print("allocation: %d steps of allocate, kernel, free, synchronize (4 MiB)" % steps)
    print("  strategy                               per step (us)   total (ms)")
    for name, cost in [("cudaMalloc + cudaFree", STEP_MALLOC),
                       ("pool, release threshold 0 (default)", STEP_POOL_0),
                       ("pool, release threshold maximum", STEP_POOL_MAX)]:
        print("  %-37s  %13.2f   %10.1f" % (name, cost, cost * steps / 1000))
    print("  kept pool against cudaMalloc        : %.1fx less per step, %.1f ms saved"
          % (STEP_MALLOC / STEP_POOL_MAX, (STEP_MALLOC - STEP_POOL_MAX) * steps / 1000))
    print("  default pool against cudaMalloc     : %.1fx more per step" % (STEP_POOL_0 / STEP_MALLOC))
    print("  cudaFree beside a %.1f ms kernel     : blocked %.1f ms, %.0f%% of the kernel"
          % (BUSY_KERNEL / 1000, FREE_BUSY / 1000, 100 * FREE_BUSY / BUSY_KERNEL))
    print("  pool pair beside the same kernel    : blocked %.1f us" % ASYNC_BUSY)


if __name__ == "__main__":
    main()
