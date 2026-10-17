"""vector_add.py: a CPU model of the vector-addition kernel of post 02 (no GPU needed).

It reproduces in NumPy the logic of snippets/vector_add.cu and prints only numbers it derives:

1. A launch of blocksPerGrid x blockSize threads over N = 1,000,003 elements (not a multiple of 256):
   each thread computes i = blockIdx.x * blockDim.x + threadIdx.x, the guard i < N stops the surplus
   threads, and the model checks that every element of C is written exactly once and equals A + B.
2. The small launch the post draws: N = 6 elements in blocks of 4 threads.
3. The bytes each element moves (read A, read B, write C) and the arithmetic intensity, and the
   shortest time a kernel could take at the card's theoretical peak bandwidth of 160 GB/s
   (post 01's device query), for each array size the benchmark measures.

Run from the series root:  uv run python posts/02-vector-addition/snippets/vector_add.py
"""
import numpy as np


def launch(n, block_size):
    """(blocks, threads, surplus) of a 1D launch that covers n elements."""
    blocks = (n + block_size - 1) // block_size    # ceiling division
    return blocks, blocks * block_size, blocks * block_size - n


N = 1_000_003                       # deliberately not a multiple of block_size
block_size = 256
blocks_per_grid, total_threads, surplus = launch(N, block_size)

A = np.arange(N, dtype=np.float32)
B = np.full(N, 0.5, dtype=np.float32)
C = np.empty(N, dtype=np.float32)
written = np.zeros(N, dtype=np.int32)               # how many times each i is set

# Each launched thread's identity, exactly as the kernel sees it.
block_idx = np.repeat(np.arange(blocks_per_grid), block_size)   # blockIdx.x
thread_idx = np.tile(np.arange(block_size), blocks_per_grid)    # threadIdx.x
i = block_idx * block_size + thread_idx             # blockIdx.x * blockDim.x + threadIdx.x
in_range = i < N                                    # the guard
C[i[in_range]] = A[i[in_range]] + B[i[in_range]]
np.add.at(written, i[in_range], 1)
written_once = bool(np.all(written == 1))
matches = bool(np.array_equal(C, A + B))

print("1. launch over N = 1,000,003")
print(f"  N                 = {N:,} (not a multiple of {block_size})")
print(f"  launch            = {blocks_per_grid:,} blocks x {block_size} threads = {total_threads:,} threads")
print(f"  surplus threads   = {surplus:,} (fail the guard i < N)")
print(f"  every i written 1x= {written_once}")
print(f"  matches A + B     = {matches}")

SMALL_N, SMALL_BLOCK = 6, 4
small_blocks, small_threads, small_surplus = launch(SMALL_N, SMALL_BLOCK)
print()
print(f"2. the small launch: N = {SMALL_N}, blockSize = {SMALL_BLOCK}")
print(f"  blocksPerGrid     = ({SMALL_N} + {SMALL_BLOCK - 1}) / {SMALL_BLOCK} = {small_blocks}")
print(f"  threads           = {small_threads}, surplus = {small_surplus}")
for t in range(small_threads):
    b, tx = divmod(t, SMALL_BLOCK)
    j = b * SMALL_BLOCK + tx
    print(f"  block {b} thread {tx}: i = {b} * {SMALL_BLOCK} + {tx} = {j}  "
          + ("writes C[%d]" % j if j < SMALL_N else "fails i < N"))

BYTES_PER_ELEM = 3 * 4              # read A, read B, write C, 4 bytes each
FLOPS_PER_ELEM = 1                  # one add
intensity = FLOPS_PER_ELEM / BYTES_PER_ELEM
PEAK_GBS = 160                      # theoretical peak of the GTX 1650 (post 01's device query)
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

print()
print("3. traffic and the bandwidth floor")
print(f"  bytes per element = {BYTES_PER_ELEM} (read A, read B, write C)")
print(f"  arithmetic int.   = {FLOPS_PER_ELEM}/{BYTES_PER_ELEM} = {intensity:.4f} FLOP/byte  -> memory bound")
print(f"  fastest possible kernel at {PEAK_GBS} GB/s:")
for n in SIZES:
    floor_us = n * BYTES_PER_ELEM / (PEAK_GBS * 1e9) * 1e6
    print(f"  {n:>10,} elements: {n * BYTES_PER_ELEM / 1e6:>7.3f} MB moved, at least {floor_us:>7.2f} us")

# Fail loudly (non-zero exit) if the model's invariants are ever violated.
assert written_once and matches, "CPU model self-check failed"
assert (blocks_per_grid, surplus, small_blocks, small_surplus) == (3907, 189, 2, 2)
