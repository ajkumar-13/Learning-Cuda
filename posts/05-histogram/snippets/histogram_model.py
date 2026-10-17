"""histogram_model.py: a CPU model of the two histogram kernels of post 05, and of the race between them.

It counts what the GPU timing cannot show: how many increments land on the single most contended address
under each strategy, for the two inputs that histogram.cu times (N = 16,777,216 values into 256 bins, a
launch of 256 blocks of 256 threads with a grid-stride loop):

- random: values drawn uniformly from 0 to 255 (NumPy, seed 0; histogram.cu uses its own generator, so
  the counts agree in size, not digit for digit);
- solid: every value is 128, a solid-color image, the worst case.

histGlobal sends every increment to global memory, so the hottest global bin receives every element with
its value. histShared counts each block's elements in a private shared-memory histogram, so the hottest
address inside a block receives at most that block's 65,536 elements, and then merges with one global
atomic per bin per block, so the hottest global bin receives 256 merges whatever the data. The model also
checks that the privatized count equals the reference histogram, and replays the lost update of a plain
`bins[v]++` for two threads. Run from the series root:

    uv run python posts/05-histogram/snippets/histogram_model.py
"""
import numpy as np

N = 16 * 1024 * 1024
BINS, BLOCK, GRID = 256, 256, 256
THREADS = BLOCK * GRID                      # 65,536 threads; thread t handles i = t, t + 65,536, ...


def lost_update(start=5):
    """Two threads increment one counter with a plain read, add, write: serially, then interleaved."""
    serial = start
    for _ in range(2):
        old = serial
        serial = old + 1                    # each thread reads the other's result
    read_a = read_b = start                 # both threads read the old count before either writes
    interleaved = read_a + 1                # thread a writes 6
    interleaved = read_b + 1                # thread b also writes 6: thread a's increment is lost
    return serial, interleaved


def privatized(data):
    """histShared on the CPU: one private histogram per block, then one merge per bin per block."""
    t = np.arange(N) % THREADS              # the grid-stride loop: element i belongs to thread i % 65,536
    block = t // BLOCK
    per_block = np.zeros((GRID, BINS), dtype=np.int64)
    np.add.at(per_block, (block, data), 1)  # phase 2: each block counts into its own bins
    return per_block, per_block.sum(axis=0) # phase 3: the merge adds every block's bins


inputs = {
    "random": np.random.default_rng(0).integers(0, BINS, N),
    "solid": np.full(N, 128, dtype=np.int64),
}

rows = {}
for name, data in inputs.items():
    ref = np.bincount(data, minlength=BINS)
    per_block, merged = privatized(data)
    rows[name] = {
        "equal": bool(np.array_equal(merged, ref)),
        "global_hot": int(ref.max()),                # histGlobal: increments on the hottest global bin
        "shared_hot": int(per_block.max()),          # histShared: increments on the hottest shared bin of a block
        "merge_hot": GRID,                           # histShared: merges on any one global bin
        "merge_all": GRID * BINS,                    # histShared: every global atomic of the merge
    }

serial, interleaved = lost_update()
print("1. the race: two threads run bins[v]++ on a count of 5")
print(f"  one after the other        : {serial}")
print(f"  both read, then both write : {interleaved}  (one increment lost)")
print()
print(f"2. N = {N:,} values into {BINS} bins, <<<{GRID}, {BLOCK}>>>, {N // THREADS} elements per thread")
print(f"  {'':<40} {'random':>10} {'solid':>11}")
print(f"  {'privatized count equals the reference':<40} {rows['random']['equal']!s:>10} {rows['solid']['equal']!s:>11}")
for key, label in [("global_hot", "histGlobal: atomics on the hottest bin"),
                   ("shared_hot", "histShared: hottest shared bin, 1 block"),
                   ("merge_hot", "histShared: merges on any global bin"),
                   ("merge_all", "histShared: merge atomics in all")]:
    print(f"  {label:<40} {rows['random'][key]:>10,} {rows['solid'][key]:>11,}")
print()
print("3. the private histogram's size in shared memory (4-byte int bins; 48 KB per block on the GTX 1650)")
for bins in (256, 4096, 65536):
    kb = bins * 4 // 1024
    print(f"  {bins:>6,} bins = {kb:>3} KB  {'fits' if kb <= 48 else 'does not fit'}")
