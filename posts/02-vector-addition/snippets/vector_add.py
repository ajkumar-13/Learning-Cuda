"""vector_add.py — a CPU model of the vector-add kernel.

CUDA cannot run in this blog's test environment, so this NumPy model reproduces
the *exact* logic of the kernel in vector_add.cu: it simulates a launch of
`blocksPerGrid x blockSize` threads, runs each through the global-index mapping
`i = blockIdx.x * blockDim.x + threadIdx.x`, applies the `if (i < N)` guard, and
checks that every element of C is written exactly once and equals A + B.

It prints only real, derived numbers (sizes, the surplus-thread count, and the
arithmetic intensity). It invents no bandwidth figures.

Run: python vector_add.py   (needs only NumPy)
"""
import numpy as np

N = 1_000_003                       # deliberately not a multiple of blockSize
block_size = 256
blocks_per_grid = (N + block_size - 1) // block_size   # ceiling division
total_threads = blocks_per_grid * block_size

A = np.arange(N, dtype=np.float32)
B = np.full(N, 0.5, dtype=np.float32)
C = np.empty(N, dtype=np.float32)
written = np.zeros(N, dtype=np.int32)               # how many times each i is set

# Each launched thread's identity, exactly as the kernel sees it.
block_idx = np.repeat(np.arange(blocks_per_grid), block_size)   # blockIdx.x
thread_idx = np.tile(np.arange(block_size), blocks_per_grid)    # threadIdx.x
i = block_idx * block_size + thread_idx             # blockIdx*blockDim + threadIdx
in_range = i < N                                    # the boundary check
C[i[in_range]] = A[i[in_range]] + B[i[in_range]]
np.add.at(written, i[in_range], 1)

bytes_per_elem = 3 * 4                               # read A, read B, write C
flops_per_elem = 1                                  # one add
intensity = flops_per_elem / bytes_per_elem

print("vector_add CPU model")
print(f"  N                 = {N:,} (not a multiple of {block_size})")
print(f"  launch            = {blocks_per_grid:,} blocks x {block_size} "
      f"threads = {total_threads:,} threads")
print(f"  surplus threads   = {total_threads - N:,} (guarded out by i < N)")
written_once = bool(np.all(written == 1))
matches = bool(np.allclose(C, A + B))
print(f"  every i written 1x= {written_once}")
print(f"  matches A + B     = {matches}")
print(f"  bytes per element = {bytes_per_elem} (read A, read B, write C)")
print(f"  arithmetic int.   = {intensity:.4f} FLOP/byte  -> memory bound")

# Fail loudly (non-zero exit) if the model's invariants are ever violated.
assert written_once and matches, "CPU model self-check failed"
