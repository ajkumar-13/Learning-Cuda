# 08 · Parallel scan — breaking the dependency chain of a prefix sum

> **TL;DR.** A scan (prefix sum) turns `[3,1,7,0]` into the running totals `[3,4,11,11]`, and it looks impossible to parallelize because `out[i] = out[i-1] + in[i]` makes every element wait for the one before it, a serial chain of length `N`. The escape is to **trade work for depth**: do more additions but in `log N` parallel steps instead of `N` sequential ones. Two algorithms do this, **Hillis-Steele** (simple, but `O(N log N)` work) and **Blelloch** (an up-sweep then a down-sweep, `O(N)` work), and a conflict-free addressing trick removes the shared-memory bank conflicts. Scan is the engine behind **stream compaction**, radix sort, and more.
>
> **After reading this you will be able to:**
> - Explain why a prefix sum looks serial and how trading work for depth breaks the chain.
> - Contrast Hillis-Steele and Blelloch by work and depth, and pick by array size.
> - Walk the Blelloch up-sweep and down-sweep and say what each phase computes.
> - Use an exclusive scan to implement stream compaction (predicate, scan, scatter).

![An input array and its inclusive scan running total, with a chain of arrows showing each output depends on the previous one.](diagrams/01-scan-dependency.svg)
*Each output is the running total up to that point; the dependency chain is what makes it look serial.*

---

## 1. The motivation: a chain that looks unbreakable

After map (independent) and reduce (converge to one), scan is the third pillar: **N inputs to N outputs**, each a running total. The CPU version is a one-line loop, and that loop is the problem: `out[i]` reads `out[i-1]`, so the work is a dependency chain of length `N`. How do you use a thousand threads when thread 999 needs thread 0's result first?

The answer is to give up *work-optimality* for *depth*. A serial scan does `N-1` additions in `N-1` steps; a parallel scan does as many or more additions, but arranges them as a tree of depth `log N`. The trade is summarized in two numbers:

| | Work (total adds) | Depth (parallel steps) |
|---|---|---|
| serial | `O(N)` | `N` steps |
| Hillis-Steele | `O(N log N)` | `~log N` steps |
| Blelloch | `O(N)` | `~2 log N` steps (up-sweep + down-sweep) |

## 2. Two algorithms

**Hillis-Steele** is the obvious parallel scan: at step `s` (a doubling stride), every element adds the element `s` positions back. After `log N` steps each element holds its running total. It is short and has minimal depth, but it does `N log N` additions, much more than the `N` a serial scan does, so it wastes work and bandwidth on large arrays. It is, however, the right choice *within a warp*, where `__shfl_up_sync` (a warp shuffle that lets a thread read a register from a lane below it, with no shared memory) makes it cheap.

**Blelloch** is work-efficient. It runs the array through two tree passes in place. The **up-sweep** is exactly the reduction tree from [Post 04](../04-reduction/index.md): it builds partial sums from the leaves to the root, leaving the total sum at the end. Then the root is cleared to the identity, and the **down-sweep** walks back down, at each node passing the parent's value to the left child and the parent-plus-saved-value to the right, which distributes the prefixes.

![Three rows: input, the array after the up-sweep with the total sum at the end, and the green exclusive scan after the down-sweep.](diagrams/02-blelloch.svg)
*Up-sweep builds the sums (a reduction); down-sweep distributes them into the exclusive prefixes.*

```cuda
for (int d = n >> 1; d > 0; d >>= 1) {          // up-sweep: build partial sums
    __syncthreads();
    if (t < d) { int ai = off*(2*t+1)-1, bi = off*(2*t+2)-1; s[bi] += s[ai]; }
    off *= 2;
}
if (t == 0) s[n - 1] = 0;                        // clear root: exclusive scan
for (int d = 1; d < n; d *= 2) {                 // down-sweep: distribute
    off >>= 1; __syncthreads();
    if (t < d) { int ai=off*(2*t+1)-1, bi=off*(2*t+2)-1;
                 float tmp = s[ai]; s[ai] = s[bi]; s[bi] += tmp; }
}
```

A CPU model confirms both algorithms against a reference and shows the work gap: at `N = 2²⁰`, Hillis-Steele does about **10×** the additions of Blelloch.

The text-output block below comes from `python snippets/scan_model.py`:

```text
scan model
  Hillis-Steele inclusive == cumsum : True
  Blelloch exclusive == reference   : True
  work for N=2^20: Hillis-Steele 20,971,520 adds vs Blelloch 2,097,150 (10.0x)
  stream compaction: [3, -1, 7, 0, -2, 4, 1, -5, 6]
                  -> [3, 7, 4, 1, 6]  (kept 5 positives, in parallel)
```

Build and run the kernel (the CUDA scan) and the CPU model:

```bash
nvcc -O3 -arch=sm_75 snippets/scan.cu -o scan && ./scan
python snippets/scan_model.py
```

## 3. Bank conflicts, and arrays larger than a block

The Blelloch strides hit the same shared-memory hazard as [Post 06](../06-matrix-transpose/index.md): at large strides, the accessed indices share a bank and serialize, up to a 32-way conflict. The fix is **conflict-free addressing**, adding a small offset (`index >> 5`) so that every 32 elements skip one slot. This roughly halves the time (1.8 ms -> 0.9 ms on the full 16M-float scan, Section 5). The shipped `snippets/scan.cu` is the *naive* Blelloch (plain indices, for clarity); the conflict-free version is the same kernel with this `index >> 5` offset added to every index, sketched here rather than coded.

A block-level scan only handles as many elements as fit in shared memory (~1–2k). For millions, use a **three-phase hierarchical scan**: (1) scan each block and save its total, (2) scan the array of block totals (recursively if needed), (3) add each block's scanned offset back to its elements. The same pattern that made reduction and histogram scale.

## 4. The payoff: stream compaction

Scan's importance is that it computes *positions* in parallel. **Stream compaction** removes unwanted elements and closes the gaps: build a 0/1 **predicate** of what to keep, take its **exclusive scan**, and that scan is exactly each kept element's destination index. A **scatter** then writes each survivor to its place.

![Four rows: input, keep/drop predicate, its green exclusive scan giving positions, and the compacted output via scatter arrows.](diagrams/03-stream-compaction.svg)
*The exclusive scan of the predicate is each survivor's output index; the scatter places them with no gaps.*

This is how a GPU implements a parallel filter, the backbone of radix sort, particle culling, ray compaction, and GPU database queries. The whole thing is `O(N)` work and `O(log N)` depth.

## 5. The numbers

Each lever (a work-efficient algorithm, then conflict avoidance) compounds, landing near NVIDIA's CUB library:

| Implementation | Time (16M floats) | Note |
|---|---|---|
| Hillis-Steele | 3.2 ms | `N log N` work |
| Blelloch (naive) | 1.8 ms | the shipped snippet (bank conflicts) |
| Blelloch (conflict-free) | 0.9 ms | adds the `index >> 5` offset (Section 3) |
| CUB | 0.6 ms | tuned library |

![A time bar chart: Hillis-Steele 3.2, Blelloch naive 1.8, Blelloch conflict-free 0.9 in green, CUB 0.6 in blue.](diagrams/04-scan-perf.svg)
*Work-efficiency takes 3.2 ms to 1.8; bank-conflict avoidance takes it to 0.9, within ~1.5× of CUB.*

These times were measured separately on an NVIDIA RTX 3080 (compute capability 8.6) scanning 16M floats; the shipped `snippets/scan.cu` checks correctness on a single small block rather than producing this table.

---

## Common pitfalls

- **Off-by-one in the tree indices.** The node indices are `offset*(2*tid+1) - 1` and `offset*(2*tid+2) - 1`; dropping the `-1` (0-based indexing) silently scrambles the result.
- **Missing a `__syncthreads()` between sweep steps.** Each step reads what the previous wrote across threads; without the barrier it is a race.
- **Confusing exclusive and inclusive.** Blelloch yields an *exclusive* scan (starts at the identity); add the input back (`inclusive[i] = exclusive[i] + in[i]`) if you need inclusive.
- **Assuming a power-of-two length.** Pad virtually to the next power of two by loading the identity (0 for sum) for out-of-range indices; do not resize the host array.
- **`cudaMalloc` inside the recursion.** The hierarchical scan should allocate one workspace up front; allocating per recursive call stalls the CPU on a synchronous driver call every level.

---

## Further reading

- Harris, M., Sengupta, S., & Owens, J., *"Parallel Prefix Sum (Scan) with CUDA"* (GPU Gems 3, ch. 39, 2007). The canonical Blelloch implementation with conflict-free addressing (technical, foundational).
- Blelloch, G., *"Prefix Sums and Their Applications"* (CMU, 1990). The algorithm and its many uses, including compaction and sorting (technical, historical).
- Hillis, W. D. & Steele, G. L., *"Data Parallel Algorithms"* (CACM, 1986). The original log-stride scan (historical).
- NVIDIA, *"CUB: DeviceScan"* documentation (current). The production scan to benchmark against and usually to use (reference).

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 04, Reduction](../04-reduction/index.md)**: the tree that the Blelloch up-sweep reuses, and where the work-vs-depth idea began.
- **[Post 09, Synchronization and the memory model](../09-synchronization/index.md)**: the rules that quietly made the last few patterns correct (barriers, atomics, and CUDA's memory model), gathered into one place before you lean on them harder.
