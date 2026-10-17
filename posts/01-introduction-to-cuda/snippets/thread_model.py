"""thread_model.py: a CPU model of the execution ideas of post 01 (no GPU needed).

It reproduces, in plain Python, four things the hardware does, and prints only counts it derives:

1. A launch of N = 1000 elements in blocks of 256 threads: the number of blocks (a ceiling division),
   the surplus threads, and that the guarded launch covers every element exactly once.
2. The global index i = blockIdx.x * blockDim.x + threadIdx.x for a few threads, with the warp and
   lane each one lands in and whether it passes the guard i < N.
3. Warp divergence in one 256-thread block: a warp runs one instruction for all 32 lanes at a time, so
   a warp whose lanes take two different paths runs both paths, one pass each.
4. Latency hiding, with illustrative timings: a warp issues 4 instructions and then waits 12 cycles
   for memory; the scheduler issues from any ready warp, so enough warps keep it busy.

Run from the series root:  uv run python posts/01-introduction-to-cuda/snippets/thread_model.py
"""
WARP = 32
N, BLOCK = 1000, 256


def global_index(block_idx, thread_idx, block_dim=BLOCK):
    return block_idx * block_dim + thread_idx


def passes(branch, warp_starts):
    """Passes a set of warps needs: one per distinct path its 32 lanes take."""
    total = 0
    for start in warp_starts:
        paths = {branch(start + lane) for lane in range(WARP)}
        total += len(paths)          # 1 when the whole warp agrees, 2 when it splits
    return total


def schedule(n_warps, cycles, work=4, wait=12):
    """Which warp issues on each cycle (None = idle): a warp issues `work` instructions in a row,
    then waits `wait` cycles for its load; the scheduler switches to the next ready warp."""
    ready_at = [0] * n_warps         # cycle at which each warp may issue again
    left = [work] * n_warps          # instructions left before its next load
    current, issued = 0, []
    for t in range(cycles):
        order = [(current + k) % n_warps for k in range(n_warps)]
        w = next((w for w in order if ready_at[w] <= t), None)
        issued.append(w)
        if w is None:
            continue
        current = w
        left[w] -= 1
        if left[w] == 0:             # the load goes out; the warp sleeps until it returns
            left[w], ready_at[w] = work, t + 1 + wait
    return issued


def busy_share(n_warps, cycles=1600):
    issued = schedule(n_warps, cycles)
    return sum(w is not None for w in issued) / cycles


grid = (N + BLOCK - 1) // BLOCK                 # ceil(N / BLOCK)
threads = grid * BLOCK
covered = sorted(i for b in range(grid) for t in range(BLOCK) if (i := global_index(b, t)) < N)

print("thread model (derived on the CPU)")
print()
print(f"1. launch: N = {N}, blockDim.x = {BLOCK}")
print(f"  gridDim.x   = ceil({N} / {BLOCK}) = {grid} blocks")
print(f"  threads     = {grid} x {BLOCK} = {threads}")
print(f"  surplus     = {threads} - {N} = {threads - N}  (fail the guard i < N)")
print(f"  every element covered exactly once: {covered == list(range(N))}")
print()
print("2. i = blockIdx.x * blockDim.x + threadIdx.x")
print("  blockIdx.x  threadIdx.x     i  warp  lane  i < N")
for b, t in [(0, 0), (1, 5), (3, 5), (3, 231), (3, 232), (3, 255)]:
    i = global_index(b, t)
    print(f"  {b:>10}  {t:>11}  {i:>4}  {t // WARP:>4}  {t % WARP:>4}  {str(i < N):>5}")
print()
warps = range(0, BLOCK, WARP)                   # the first lane of each warp of block 0
print(f"3. divergence in one block of {BLOCK} threads ({BLOCK // WARP} warps of {WARP}), passes:")
print(f"  no branch                  = {passes(lambda i: 0, warps)}")
print(f"  branch on i % 2            = {passes(lambda i: i % 2, warps)}  (every warp splits)")
print(f"  branch on (i / 32) % 2     = {passes(lambda i: (i // WARP) % 2, warps)}  (whole warps agree)")
print()
print("4. latency hiding (illustrative: 4 instructions, then a 12-cycle wait)")
print("  warps  scheduler busy")
for w in (1, 2, 4, 8):
    print(f"  {w:>5}  {busy_share(w):>13.0%}")
