"""scan_model.py: a CPU model of the scans of post 08 (no GPU needed).

It reproduces, in plain Python, what the kernels of scan.cu do, and prints only numbers it derives:

1. The serial scan of the post's eight values: the inclusive and exclusive running totals.
2. Hillis-Steele on the eight values: the array after each doubling stride, checked against the serial scan.
3. Blelloch on the eight values: the array after the up-sweep, after clearing the root, and after the
   down-sweep, checked against the serial exclusive scan.
4. Work and depth: the additions and the parallel steps of the serial scan, Hillis-Steele and Blelloch,
   for N = 2^20 and for the N = 2^24 floats that scan.cu times.
5. Bank conflicts in one Blelloch tile of 2,048 floats (1,024 threads, 32 banks of 4 bytes, as on compute
   capability 7.5): for each level of the up-sweep, the worst warp's conflict degree (the number of
   shared-memory passes one warp-wide access takes) with plain indices and with index + (index >> 5),
   and the total passes of both sweeps.
6. The three-phase multi-block scan of scan.cu: the blocks at each level and the bytes it moves.
7. Stream compaction of nine values: predicate, exclusive scan of the predicate, scatter.

Run from the series root:  uv run python posts/08-parallel-scan/snippets/scan_model.py
"""
WARP, BANKS = 32, 32
EIGHT = [3, 1, 7, 0, 4, 1, 6, 3]                 # the eight values of scan.cu's one-block demo
DATA = [3, -1, 7, 0, -2, 4, 1, -5, 6]            # the compaction example: keep the positives
N_TIMED = 16 * 1024 * 1024                        # the floats scan.cu times
TILE = 2048                                       # one Blelloch block: 1,024 threads, two elements each


def serial_inclusive(x):
    out, run = [], 0
    for v in x:
        run += v                                  # out[i] = out[i - 1] + in[i]: the chain
        out.append(run)
    return out


def serial_exclusive(x):
    return [r - v for r, v in zip(serial_inclusive(x), x)]


def hillis_steele(x):
    """Levels of the Hillis-Steele inclusive scan: at stride s, element t adds element t - s (t >= s)."""
    a, levels, stride = list(x), [list(x)], 1
    while stride < len(a):
        a = [a[t] + (a[t - stride] if t >= stride else 0) for t in range(len(a))]   # read old, write new
        levels.append((stride, list(a)))
        stride *= 2
    return levels


def blelloch(x):
    """(after the up-sweep, after clearing the root, after the down-sweep) of the Blelloch exclusive scan,
    with the tree indices of scan.cu: ai = offset * (2t + 1) - 1, bi = offset * (2t + 2) - 1."""
    s, n, offset = list(x), len(x), 1
    d = n >> 1
    while d > 0:                                  # up-sweep: a reduction tree, in place
        for t in range(d):
            ai, bi = offset * (2 * t + 1) - 1, offset * (2 * t + 2) - 1
            s[bi] += s[ai]
        offset *= 2
        d >>= 1
    up = list(s)
    s[n - 1] = 0                                  # clear the root
    cleared = list(s)
    d = 1
    while d < n:                                  # down-sweep
        offset >>= 1
        for t in range(d):
            ai, bi = offset * (2 * t + 1) - 1, offset * (2 * t + 2) - 1
            s[ai], s[bi] = s[bi], s[bi] + s[ai]   # left <- parent, right <- parent + saved left
        d *= 2
    return up, cleared, s


def log2(n):
    assert n & (n - 1) == 0, "a power of two"
    return n.bit_length() - 1


def work(n):
    """Additions and parallel steps for an array of n elements (n a power of two)."""
    k = log2(n)
    return {
        "serial": (n - 1, n - 1),
        # every thread of the kernel adds at every step (t < stride adds 0), and n - stride of those are useful
        "hs_lane": (n * k, k),
        "hs_useful": (sum(n - (1 << j) for j in range(k)), k),
        "blelloch": (2 * (n - 1), 2 * k),         # n - 1 adds up, n - 1 down (plus n - 1 swaps)
    }


def pad(k):
    return k + (k >> 5)


def degree(addresses):
    """Shared-memory passes for one warp-wide access: the most distinct words that fall in one bank."""
    per_bank = {}
    for a in set(addresses):
        per_bank.setdefault(a % BANKS, set()).add(a)
    return max(len(v) for v in per_bank.values()) if per_bank else 0


def sweep_conflicts(n=TILE, index=lambda k: k):
    """Per up-sweep level: (offset, active threads, worst warp's degree for s[bi]); and the total passes of
    every warp-wide shared access of both sweeps (up: read ai, read bi, write bi; down: read ai, read bi,
    write ai, write bi), against the passes a conflict-free access would take."""
    levels, total, ideal = [], 0, 0

    def accesses(d, offset, which):
        for w in range(0, d, WARP):               # the warps that have active threads
            ts = range(w, min(w + WARP, d))
            ai = [index(offset * (2 * t + 1) - 1) for t in ts]
            bi = [index(offset * (2 * t + 2) - 1) for t in ts]
            yield from ([{"ai": ai, "bi": bi}[c] for c in which])

    offset, d = 1, n >> 1
    while d > 0:
        worst = 0
        for acc in accesses(d, offset, ["ai", "bi", "bi"]):
            total += degree(acc); ideal += 1
        for w in range(0, d, WARP):
            ts = range(w, min(w + WARP, d))
            worst = max(worst, degree([index(offset * (2 * t + 2) - 1) for t in ts]))
        levels.append((offset, d, worst))
        offset *= 2
        d >>= 1
    d = 1
    while d < n:
        offset >>= 1
        for acc in accesses(d, offset, ["ai", "bi", "ai", "bi"]):
            total += degree(acc); ideal += 1
        d *= 2
    return levels, total, ideal


def three_phase(n, per):
    """Block counts at each level of scan.cu's scan of n elements with `per` elements per block."""
    counts = []
    while True:
        blocks = -(-n // per)
        counts.append(blocks)
        if blocks == 1:
            return counts
        n = blocks


def compact(data, keep=lambda v: v > 0):
    pred = [1 if keep(v) else 0 for v in data]
    pos = serial_exclusive(pred)                  # each kept element's output index
    out = [None] * sum(pred)
    for i, v in enumerate(data):                  # scatter: every element independently
        if pred[i]:
            out[pos[i]] = v
    return pred, pos, out


def fmt(xs):
    return " ".join(f"{v:>2}" for v in xs)


def main():
    inc, exc = serial_inclusive(EIGHT), serial_exclusive(EIGHT)
    print("1. the serial scan: out[i] = out[i - 1] + in[i], a chain of 7 additions")
    print(f"  input           : {fmt(EIGHT)}")
    print(f"  inclusive       : {fmt(inc)}")
    print(f"  exclusive       : {fmt(exc)}")

    print("2. Hillis-Steele: at stride s, element t adds element t - s")
    hs = hillis_steele(EIGHT)
    for stride, row in hs[1:]:
        useful = len(EIGHT) - stride
        print(f"  stride {stride}, {useful} adds : {fmt(row)}")
    print(f"  equals the inclusive scan: {hs[-1][1] == inc}, in {len(hs) - 1} steps")

    print("3. Blelloch: up-sweep, clear the root, down-sweep")
    up, cleared, down = blelloch(EIGHT)
    print(f"  after up-sweep  : {fmt(up)}   (the last element is the total, {up[-1]})")
    print(f"  root cleared    : {fmt(cleared)}")
    print(f"  after down-sweep: {fmt(down)}")
    print(f"  equals the exclusive scan: {down == exc}, in {2 * log2(len(EIGHT))} steps")

    print("4. work (additions) and depth (parallel steps)")
    print("  N                    serial adds    Hillis-Steele adds   useful of those   Blelloch adds   "
          "depth S / HS / B")
    for n in (1 << 20, N_TIMED):
        w = work(n)
        print(f"  {n:>10,} (2^{log2(n)})  {w['serial'][0]:>13,}  {w['hs_lane'][0]:>20,}  "
              f"{w['hs_useful'][0]:>16,}  {w['blelloch'][0]:>14,}   {w['serial'][1]:,} / {w['hs_lane'][1]} / "
              f"{w['blelloch'][1]}")
    w = work(1 << 20)
    print(f"  at N = 2^20 Hillis-Steele adds {w['hs_lane'][0] / w['blelloch'][0]:.1f} times as often as "
          f"Blelloch ({w['hs_useful'][0] / w['blelloch'][0]:.1f} times counting only the useful adds)")

    print(f"5. bank conflicts in one Blelloch tile of {TILE:,} floats ({TILE // 2:,} threads, {BANKS} banks)")
    plain, total_p, ideal = sweep_conflicts()
    padded, total_c, ideal_c = sweep_conflicts(index=pad)
    assert ideal == ideal_c
    print("  up-sweep offset  active threads  worst degree, plain  worst degree, index + (index >> 5)")
    for (off, d, wp), (_, _, wc) in zip(plain, padded):
        print(f"  {off:>15,}  {d:>14,}  {wp:>19}  {wc:>34}")
    print(f"  shared passes of both sweeps: plain {total_p:,}, padded {total_c:,}, "
          f"conflict-free ideal {ideal:,}")
    print(f"  padded tile: the highest index, {TILE - 1:,}, lands at {pad(TILE - 1):,}; scan.cu reserves "
          f"{TILE:,} + {TILE >> 5} = {TILE + (TILE >> 5):,} floats")

    print(f"6. the three-phase scan of N = {N_TIMED:,} floats in scan.cu")
    for name, per in (("Hillis-Steele and warp shuffle", 1024), ("Blelloch", 2048)):
        print(f"  {name}, {per:,} per block: blocks per level {' -> '.join(f'{c:,}' for c in three_phase(N_TIMED, per))}")
    print(f"  bytes: phase 1 reads and writes {8 * N_TIMED:,}, phase 3 again {8 * N_TIMED:,}: "
          f"{16 * N_TIMED:,} in all, twice the {8 * N_TIMED:,} of a single pass")

    pred, pos, out = compact(DATA)
    print("7. stream compaction: keep the positive values")
    print(f"  input           : {fmt(DATA)}")
    print(f"  predicate       : {fmt(pred)}")
    print(f"  exclusive scan  : {fmt(pos)}")
    print(f"  scatter         : {out}  (kept {len(out)} of {len(DATA)}; the scan's total is {pos[-1] + pred[-1]})")


if __name__ == "__main__":
    main()
