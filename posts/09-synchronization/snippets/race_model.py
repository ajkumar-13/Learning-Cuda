"""race_model.py — a CPU model of the lost-update race and the atomic fix.

CUDA cannot run in this blog's test environment, so this NumPy-free model
reproduces, in plain Python, what happens when many logical threads each run
`counter += 1` on one shared location. A single increment is really three
micro-ops: READ the shared value into a private register, INC the register,
WRITE it back. A schedule is an ordering of those micro-ops across all threads,
and the ordering decides how many updates survive.

It prints only real, derived counts, and it is fully deterministic (no clocks,
no randomness): the schedules below are fixed constructions. Run:
python snippets/race_model.py  (from the post directory)
"""
WARP = 32
T, K = 1024, 100                 # T threads, each doing K increments
intended = T * K                 # every increment counted exactly once


def simulate(events):
    """Replay a stream of (op, thread) micro-ops against one shared counter.
    op is 'R' (load shared -> private reg), 'I' (reg += 1), 'W' (reg -> shared)."""
    counter = 0
    reg = [0] * T
    for op, i in events:
        if op == 'R':
            reg[i] = counter     # read the shared counter into a private register
        elif op == 'I':
            reg[i] += 1          # modify the private copy
        else:                    # 'W'
            counter = reg[i]     # write the private copy back to shared memory
    return counter


def serial(T, K):
    """A lucky, race-free order: each increment's R,I,W are contiguous."""
    for i in range(T):
        for _ in range(K):
            yield 'R', i; yield 'I', i; yield 'W', i


def lockstep(T, K, group):
    """Threads run in lockstep groups of `group`: within a group every lane
    READs, then every lane INCs, then every lane WRITEs (SIMT), so the group
    collapses to a single +1 per step. Groups run one after another."""
    for base in range(0, T, group):
        lanes = range(base, min(base + group, T))
        for _ in range(K):
            for i in lanes: yield 'R', i
            for i in lanes: yield 'I', i
            for i in lanes: yield 'W', i


def atomic(T, K):
    """atomicAdd makes each read-modify-write indivisible, so contenders are
    serialized by the hardware and no update is ever lost, for any arrival."""
    counter = 0
    for _ in range(T * K):
        counter += 1             # one atomicAdd = one uninterruptible R,I,W
    return counter


# The headline schedules.
final_serial = simulate(serial(T, K))
final_warp = simulate(lockstep(T, K, WARP))     # 32 lanes collide, warps serialize
final_full = simulate(lockstep(T, K, T))        # every thread collides every step
final_atom = atomic(T, K)

# A tiny, fully interleaved trace (4 threads, 1 increment each) for the picture.
tr_counter, trace = 0, []
tr_reg = [0] * 4
for i in range(4): tr_reg[i] = tr_counter
trace.append(("READ ", "all 4 threads read counter = %d" % tr_counter))
for i in range(4): tr_reg[i] += 1
trace.append(("INC  ", "all compute %d + 1 = %d" % (tr_counter, tr_counter + 1)))
tr_counter = tr_reg[0]
trace.append(("WRITE", "all write %d -> counter = %d     (3 of 4 updates lost)" % (tr_counter, tr_counter)))


def row(name, final):
    lost = intended - final
    return "    %-28s %9s %9s        %5.3f" % (name, f"{final:,}", f"{lost:,}", lost / intended)


print("race model")
print("  threads (T)               = %d" % T)
print("  increments per thread (K) = %d" % K)
print("  intended final counter    = %s   (T x K, every increment counted)" % f"{intended:,}")
print("  warp size                 = %d" % WARP)
print()
print("  trace  (4 threads, 1 increment each, fully interleaved):")
for op, msg in trace:
    print("    %s: %s" % (op, msg))
print()
print("  final counter by schedule (higher is more correct):")
print("    " + "schedule".ljust(28) + " " + "final".rjust(9) + " " + "lost".rjust(9) + "  fraction lost")
print(row("serial (race-free order)", final_serial))
print(row("warp-lockstep (32 collide)", final_warp))
print(row("full-lockstep (all collide)", final_full))
print(row("atomicAdd (indivisible)", final_atom))
