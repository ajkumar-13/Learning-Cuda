"""Launch-overhead and allocation model for CUDA Graphs and stream-ordered
allocation (Post 13).

Pure standard-library Python: no GPU, no NumPy, no wall-clock. Every number is
*derived* from a fixed analytic cost model, so it reproduces identically on any
machine. The costs below (host launch overhead, graph build, allocation stalls)
are ILLUSTRATIVE constants, chosen to sit in the right order of magnitude for a
discrete GPU with a recent driver. They are a model, not a measurement; the real
speedup depends on the GPU, the driver, and the kernel.

Two parts:
  1. Launch overhead. N separate kernel launches, each paying the full host
     launch cost L, versus one graph built once and replayed N times, paying L
     only at build time. Swept over kernel size W to expose the crossover.
  2. Allocation. cudaMalloc/cudaFree every iteration (device-synchronizing, so
     it blocks) versus a stream-ordered memory pool (cudaMallocAsync) that hands
     back a reused block at amortized ~zero cost.
"""

# ---- Illustrative cost model, in microseconds. A model, not a measurement. ----
L_LAUNCH   = 5.0     # host cost to launch one kernel (order of a few microseconds)
REPLAY_OH  = 0.5     # host cost to replay a whole instantiated graph, per replay
BUILD_ONCE = 40.0    # one-time stream capture + cudaGraphInstantiate
N          = 1000    # iterations / launches


def direct(n, w):
    """N separate launches: every launch pays the full host overhead L."""
    return n * (L_LAUNCH + w)


def graph(n, w):
    """Build once, then replay: the launch overhead is paid once, not N times."""
    return BUILD_ONCE + n * (REPLAY_OH + w)


W_SWEEP = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 500.0]

print("launch-overhead model  (illustrative: L=%.1f us/launch, replay=%.1f us, build=%.0f us, N=%d)"
      % (L_LAUNCH, REPLAY_OH, BUILD_ONCE, N))
print()
print("  W (us)   direct (us)   graph (us)   speedup   verdict")
print("  " + "-" * 60)
for w in W_SWEEP:
    d, g = direct(N, w), graph(N, w)
    s = d / g
    verdict = "graphs win big" if s >= 2 else ("graphs help" if s >= 1.1 else "negligible")
    print("  %6.1f   %10.1f   %10.1f   %6.2fx   %s" % (w, d, g, s, verdict))
print()

# Closed form: speedup(W) = (L + W) / (REPLAY_OH + BUILD/N + W).
base = REPLAY_OH + BUILD_ONCE / N
amax = (L_LAUNCH + 0.0) / base                       # limit as W -> 0
w_thresh = (L_LAUNCH - 1.1 * base) / (1.1 - 1.0)      # solve speedup(W) = 1.1
print("  max speedup as W -> 0            : %.2fx  (tiny kernels, pure host overhead)" % amax)
print("  W where speedup falls to 1.1x    : %.1f us  (bigger kernels: graphs stop mattering)" % w_thresh)

# ---- Allocation model, in microseconds. Illustrative constants. ----
MALLOC_BLOCK = 20.0   # cudaMalloc: driver work + a device-wide sync, so it blocks
FREE_BLOCK   = 15.0   # cudaFree: also synchronizes the device
POOL_SETUP   = 20.0   # first cudaMallocAsync grows the pool once (one real malloc)
POOL_OP      = 0.3    # later pool alloc/free: stream-ordered, hands back a reused block


def malloc_every_iter(n):
    return n * (MALLOC_BLOCK + FREE_BLOCK)


def pool_reuse(n):
    return POOL_SETUP + n * POOL_OP


print()
print("allocation model  (illustrative: malloc=%.0f us, free=%.0f us, pool op=%.1f us, N=%d)"
      % (MALLOC_BLOCK, FREE_BLOCK, POOL_OP, N))
print()
print("  strategy                        total (us)   per-iter (us)   note")
print("  " + "-" * 74)
m, p = malloc_every_iter(N), pool_reuse(N)
print("  cudaMalloc/cudaFree each iter   %10.1f   %12.2f   blocks: device-wide sync" % (m, m / N))
print("  cudaMallocAsync pool (reuse)    %10.1f   %12.2f   stream-ordered, amortized ~0" % (p, p / N))
print()
print("  pool speedup on allocation      : %.1fx" % (m / p))
print("  host time returned to other work: %.1f ms over %d iters" % ((m - p) / 1000.0, N))
