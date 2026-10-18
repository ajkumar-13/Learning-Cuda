"""race_model.py: a CPU model of the lost-update race and the atomic fix.

A plain `counter += 1` on a shared location is three micro-ops: READ the shared value into a private
register, INC the register, WRITE it back. A schedule is an ordering of those micro-ops across all
threads, and the ordering decides how many updates survive. This model replays fixed schedules (no
clocks, no randomness), so every count it prints is derived and the same on every run; it needs only
the standard library and no GPU. Run from the series root:

    uv run python posts/09-synchronization/snippets/race_model.py
"""
WARP = 32
T, K = 1024, 100                 # T threads, each doing K increments
intended = T * K                 # every increment counted exactly once


def simulate(events, start=0, threads=T):
    """Replay a stream of (op, thread) micro-ops against one shared counter.
    op is 'R' (load shared -> private register), 'I' (register += 1), 'W' (register -> shared)."""
    counter = start
    reg = [0] * threads
    for op, i in events:
        if op == 'R':
            reg[i] = counter     # read the shared counter into a private register
        elif op == 'I':
            reg[i] += 1          # modify the private copy
        else:                    # 'W'
            counter = reg[i]     # write the private copy back
    return counter


def serial(T, K):
    """A lucky, race-free order: each increment's R, I, W are contiguous."""
    for i in range(T):
        for _ in range(K):
            yield 'R', i; yield 'I', i; yield 'W', i


def lockstep(T, K, group):
    """Threads run in lockstep groups of `group`: within a group every lane READs, then every lane
    INCs, then every lane WRITEs, so the group adds a single 1 per step. Groups run one after another."""
    for base in range(0, T, group):
        lanes = range(base, min(base + group, T))
        for _ in range(K):
            for i in lanes: yield 'R', i
            for i in lanes: yield 'I', i
            for i in lanes: yield 'W', i


def atomic(T, K, group):
    """atomicAdd on the full-lockstep arrival order: the lanes arrive together, but the hardware
    performs each lane's R, I, W as one indivisible step, one lane after another."""
    for base in range(0, T, group):
        lanes = range(base, min(base + group, T))
        for _ in range(K):
            for i in lanes:
                yield 'R', i; yield 'I', i; yield 'W', i


def two_threads(start=7):
    """The figure's race: threads 0 and 1 both READ before either WRITEs."""
    events = [('R', 0), ('R', 1), ('I', 0), ('I', 1), ('W', 0), ('W', 1)]
    return events, simulate(events, start=start, threads=2)


def main():
    final_serial = simulate(serial(T, K))
    final_warp = simulate(lockstep(T, K, WARP))     # 32 lanes collide, warps take turns
    final_full = simulate(lockstep(T, K, T))        # every thread collides on every step
    final_atom = simulate(atomic(T, K, T))          # the same arrival, made indivisible
    events, final_two = two_threads(7)

    def row(name, final):
        lost = intended - final
        return "    %-28s %9s %9s        %5.3f" % (name, f"{final:,}", f"{lost:,}", lost / intended)

    print("race model")
    print("  threads (T)               = %d" % T)
    print("  increments per thread (K) = %d" % K)
    print("  intended final counter    = %s   (T x K, every increment counted)" % f"{intended:,}")
    print("  warp size                 = %d" % WARP)
    print()
    print("  trace  (2 threads, 1 increment each, counter starts at 7):")
    names = {'R': "READ ", 'I': "INC  ", 'W': "WRITE"}
    reg, counter = [0, 0], 7
    for op, i in events:
        if op == 'R':
            reg[i] = counter
            print("    %s: thread %d reads counter = %d" % (names[op], i, counter))
        elif op == 'I':
            reg[i] += 1
            print("    %s: thread %d computes %d" % (names[op], i, reg[i]))
        else:
            counter = reg[i]
            print("    %s: thread %d writes %d -> counter = %d" % (names[op], i, reg[i], counter))
    print("    final %d, intended %d: 1 of 2 updates lost" % (final_two, 7 + 2))
    print()
    print("  final counter by schedule (higher is more correct):")
    print("    " + "schedule".ljust(28) + " " + "final".rjust(9) + " " + "lost".rjust(9) + "  fraction lost")
    print(row("serial (race-free order)", final_serial))
    print(row("warp-lockstep (32 collide)", final_warp))
    print(row("full-lockstep (all collide)", final_full))
    print(row("atomicAdd (indivisible)", final_atom))


if __name__ == "__main__":
    main()
