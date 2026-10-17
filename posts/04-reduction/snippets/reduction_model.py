"""reduction_model.py: a CPU model of the reductions of post 04 (no GPU needed).

It reproduces, in plain Python, what the kernels of reduction.cu do, and prints only numbers it derives:

1. A tree reduction of eight values with sequential addressing: the values after each step, the active
   threads per step, and the depth log2(8) = 3.
2. The depth of the tree for the post's N = 16,777,216 floats and for one block of 256 threads, against
   the N - 1 additions every method performs.
3. Interleaved against sequential addressing in one block of 256 threads: at each step, how many of the
   8 warps are divergent (some lanes active, some idle), and the total of these divergent warp-steps;
   and a check that the tree gives the same sum as a plain loop.
4. A warp-shuffle reduction: each lane adds the value of the lane `off` places above it (a lane whose
   source is past the end of the warp is handed its own value, as __shfl_down_sync does, so it adds
   that), for off = 4, 2, 1 over 8 lanes and
   off = 16, 8, 4, 2, 1 over a full warp of 32; lane 0 ends with the sum.
5. How many atomicAdd calls each kernel of reduction.cu issues per launch.

Run from the series root:  uv run python posts/04-reduction/snippets/reduction_model.py
"""
import random

WARP, BLOCK = 32, 256
N = 16 * 1024 * 1024                   # the input of reduction.cu: 16,777,216 floats
GRID_BLOCKS = 1024                     # the grid-stride kernels' launch
EIGHT = [3, 1, 7, 0, 4, 1, 6, 3]       # the eight values of the hero figure


def tree_sequential(values):
    """Levels of a sequential-addressing tree: at stride s, thread t < s adds element t + s into t."""
    a = list(values)
    levels, stride = [list(a)], len(a) // 2
    while stride > 0:
        for t in range(stride):        # thread t of the active first half
            a[t] += a[t + stride]
        levels.append(a[:stride])
        stride //= 2
    return levels


def active_interleaved(block=BLOCK):
    """Active thread ids at each step of `if (tid % (2 * stride) == 0)`, stride = 1, 2, 4, ..."""
    steps, s = [], 1
    while s < block:
        steps.append({t for t in range(block) if t % (2 * s) == 0})
        s *= 2
    return steps


def active_sequential(block=BLOCK):
    """Active thread ids at each step of `if (tid < stride)`, stride = block / 2, ..., 1."""
    steps, s = [], block // 2
    while s > 0:
        steps.append({t for t in range(block) if t < s})
        s //= 2
    return steps


def warp_states(active, block=BLOCK):
    """Per warp of one step: 'full' (all 32 lanes active), 'none', or 'split' (divergent)."""
    states = []
    for w in range(block // WARP):
        n = sum(1 for lane in range(WARP) if w * WARP + lane in active)
        states.append("full" if n == WARP else "none" if n == 0 else "split")
    return states


def divergent_warp_steps(steps, block=BLOCK):
    """Sum over the steps of the warps that are split: some lanes active, some idle."""
    return sum(warp_states(a, block).count("split") for a in steps)


def tree_sum(values):
    """The sum a block's sequential-addressing tree produces (len(values) a power of two)."""
    return tree_sequential(values)[-1][0]


def shuffle_down(values, offsets):
    """Lane values after each shuffle-and-add step: lane l adds the value of lane l + off, or its own
    value when l + off is past the last lane (what __shfl_down_sync returns there)."""
    v, levels = list(values), [list(values)]
    for off in offsets:
        v = [v[l] + (v[l + off] if l + off < len(v) else v[l]) for l in range(len(v))]
        levels.append(list(v))
    return levels


def atomics_per_launch():
    """atomicAdd calls of one launch of each kernel of reduction.cu."""
    blocks_all = (N + BLOCK - 1) // BLOCK
    return {"naive atomic": N, "interleaved": blocks_all, "sequential": blocks_all,
            "seq. + stride": GRID_BLOCKS, "warp shuffle": GRID_BLOCKS}


def main():
    levels = tree_sequential(EIGHT)
    print("1. tree reduction of 8 values, sequential addressing")
    print(f"  {'start':<18}: {' '.join(str(x) for x in levels[0])}")
    stride = len(EIGHT) // 2
    for lv in levels[1:]:
        adds = f"{stride} add" + ("s" if stride > 1 else "")
        print(f"  {f'stride {stride}, {adds}':<18}: {' '.join(str(x) for x in lv)}")
        stride //= 2
    print(f"  sum {levels[-1][0]} == plain sum {sum(EIGHT)}: {levels[-1][0] == sum(EIGHT)}, "
          f"in {len(levels) - 1} steps (log2 8)")

    print("2. depth of the tree against the additions")
    print(f"  N = {N}: {N.bit_length() - 1} steps, {N - 1} additions")
    print(f"  one block of {BLOCK} threads: {BLOCK.bit_length() - 1} steps, {BLOCK - 1} additions")

    inter, seq = active_interleaved(), active_sequential()
    print(f"3. one block of {BLOCK} threads ({BLOCK // WARP} warps of {WARP}): divergent warps per step")
    print("  step  interleaved: active  divergent   sequential: active  divergent")
    for k, (a, b) in enumerate(zip(inter, seq), start=1):
        print(f"  {k:>4}  {len(a):>19}  {warp_states(a).count('split'):>9}   "
              f"{len(b):>18}  {warp_states(b).count('split'):>9}")
    print(f"  total divergent warp-steps: interleaved {divergent_warp_steps(inter)}, "
          f"sequential {divergent_warp_steps(seq)}")
    rng = random.Random(0)
    x = [rng.random() for _ in range(BLOCK)]
    print(f"  tree of {BLOCK} random floats == plain sum: {abs(tree_sum(x) - sum(x)) < 1e-9}")

    print("4. warp shuffle: lane l adds lane l + off")
    lanes8 = shuffle_down(EIGHT, [4, 2, 1])
    print(f"  8 lanes, off 4 2 1  : lane 0 = {lanes8[-1][0]} (sum {sum(EIGHT)})")
    lanes32 = shuffle_down(list(range(WARP)), [16, 8, 4, 2, 1])
    print(f"  32 lanes, off 16..1 : lane 0 = {lanes32[-1][0]} (sum {sum(range(WARP))}), 5 steps")

    print(f"5. atomicAdd calls per launch, N = {N}")
    for name, n in atomics_per_launch().items():
        print(f"  {name:<14}: {n}")
    print(f"  grid-stride kernels: {N // (GRID_BLOCKS * BLOCK)} elements per thread")


if __name__ == "__main__":
    main()
