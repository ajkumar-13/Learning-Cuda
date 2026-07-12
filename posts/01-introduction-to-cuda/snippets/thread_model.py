"""thread_model.py — a CPU model of two ideas from this post.

CUDA cannot run in this blog's test environment, so this NumPy-free model
reproduces, in plain Python, two things the hardware does:

1. The global-index mapping i = blockIdx.x * blockDim.x + threadIdx.x, and the
   fact that a launch covers every element exactly once (with surplus threads
   guarded by i < N).
2. Warp divergence: threads run in lockstep groups of 32, so when threads in the
   same warp take different branches the hardware runs both paths in series.

It prints only real, derived counts. Run: python snippets/thread_model.py  (from the post directory)
"""
WARP = 32
grid_dim, block_dim, N = 4, 32, 120     # 4 blocks x 32 threads = 128 threads
total = grid_dim * block_dim

# 1. global index + coverage
idx = [b * block_dim + t for b in range(grid_dim) for t in range(block_dim)]
in_range = [i for i in idx if i < N]
covered_once = sorted(in_range) == list(range(N))

# 2. divergence: passes per warp = number of distinct branch outcomes in it
def total_passes(branch):
    passes = 0
    for w in range(total // WARP):
        outcomes = {branch(w * WARP + lane) for lane in range(WARP)}
        passes += len(outcomes)         # 1 if the whole warp agrees, else 2
    return passes

ideal = total // WARP                                  # one pass per warp
checker = total_passes(lambda i: i % 2 == 0)           # alternating lanes
warp_uniform = total_passes(lambda i: (i // WARP) % 2) # whole-warp branch

print("thread model")
print(f"  launch            = {grid_dim} blocks x {block_dim} threads = {total} threads")
print(f"  N                 = {N}  (surplus {total - N} guarded by i < N)")
print(f"  covers 0..N-1 once= {covered_once}")
print(f"  warps             = {total // WARP}  (32 threads each)")
print("  divergence (passes over all warps, lower is better):")
print(f"    no branch              = {ideal}")
print(f"    branch on (i % 2)      = {checker}   -> every warp splits, ~2x cost")
print(f"    branch on whole warp   = {warp_uniform}   -> no divergence")
