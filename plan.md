# Learning CUDA: From First Kernel to FlashAttention

Standard: 2.2.0 · Plan version: 1.0.0 · Status: draft · Date: 2026-10-03

Learning CUDA is a nineteen-post, from-scratch course in GPU programming. It starts with why a GPU exists and what a kernel is, writes and benchmarks the first kernels (vector addition, tiled matrix multiplication), builds the handful of parallel patterns almost every GPU algorithm is made from (reduction, histogram, transpose, convolution, scan) and the synchronization rules that make them correct, then draws the performance model that bounds a kernel (roofline, occupancy) and the tools that measure it (Nsight, compute-sanitizer). The last two parts move from one kernel to the whole machine (streams, graphs, fusion, asynchronous copy and software pipelining) and to the numerics, hardware, and algorithms that run today's models (floating-point formats, Tensor Cores, FlashAttention), finishing at the production tools (CUTLASS, Triton) that most code actually uses. Every post ships a CPU model in Python or NumPy that reproduces the kernel's logic and prints the numbers the post quotes, and the real `.cu` kernel beside it. A reader who finishes it can write a correct, coalesced, shared-memory-tiled kernel from a blank file, place it on a roofline and say which resource caps it, profile it, overlap its copies and launches, read a FlashAttention kernel line by line, and choose between raw CUDA, CUTLASS, Triton, and a library call for a given job.

# 1. Purpose and authority

This plan owns the curriculum of the series: which posts exist, in which order, what each one promises, what it requires and recommends, what it covers, which figures and labs carry it, and which primary sources anchor it. Layout, identifiers, the sidecar schema, the body template, the question bank, diagrams, tooling, and workflow are owned by series-standard/STANDARD.md at the version `series.yaml` pins, not by this document. `series.yaml` is the machine form of this lock: its `parts[].posts` lists the reading order, its ids are allocated from `next_post_ordinal` and never reused, and a dropped id goes to `retired`. The three stay consistent through the plan stage of `series-standard/tools/validate.py`, which fails when a heading, title, level, or dependency here disagrees with the manifest or a sidecar.

# 2. Audience, entry, and exit levels

## Reader profiles

- The programmer who knows a little C++ and has never written a kernel: the primary reader, addressed by every post. The README states the entry bar as basic programming and high-school algebra; no post assumes a GPU on the reader's desk, because each post's CPU model reproduces its numbers without one.
- The framework user who calls `torch.matmul` and `F.scaled_dot_product_attention` and wants to know what runs underneath: this reader can start at cuda-003 and should not skip Part II, because every later post leans on shared memory, coalescing, and the barrier rules.
- The performance engineer who already writes kernels and wants the systems and modern-hardware half: Parts III to V, entered at cuda-010 after a skim of cuda-003 and cuda-006 for the series' notation.

## Entry requirement

- C or C++ at the level of pointers, arrays, and `for` loops; Python at the level of running a script and reading NumPy. The compute series owns both; cuda-001 recommends cmp-002 for the hardware picture of a CPU that the GPU is contrasted with.
- Linear algebra at the level of a matrix product written as a sum, $C_{ij} = \sum_k A_{ik} B_{kj}$; cuda-003 restates it.
- A machine with an NVIDIA GPU and the CUDA Toolkit 12.x is needed only for the `.cu` kernels (SETUP.md, RUNNING.md). cuda-015 needs an Ampere or newer card (`sm_80`), cuda-017 a Volta or newer one (compute capability 7.0 or above), cuda-019 a Linux machine with Triton installed.

## Exit levels

- After cuda-003 (end of Part I, beginner): the reader can write a kernel from one thread's point of view, compute the global index, guard the surplus threads, move data across the host and device boundary, explain why vector addition is memory bound, and write a shared-memory tiled matmul with correctly placed barriers.
- After cuda-009 (end of Part II, intermediate): the reader owns the five patterns (reduce, histogram, transpose, stencil, scan) as working kernels, can diagnose coalescing, bank conflicts, atomic contention, and warp divergence from a kernel's indexing, and can state what `__syncthreads`, `__syncwarp`, atomics, fences, and scopes each guarantee.
- After cuda-011 (end of Part III): the reader can count a kernel's arithmetic intensity, place it on the roofline of a named card, derive its theoretical occupancy, and measure the achieved values with Nsight Compute and prove correctness with compute-sanitizer.
- After cuda-015 (end of Part IV): the reader can pipeline copies and compute across streams, capture and replay a graph, fuse an elementwise chain into a producer kernel, and write a double-buffered `cp.async` tile loop.
- After cuda-018 (advanced): the reader can read a float as sign, exponent, and mantissa, apply mixed precision and stable softmax, write a WMMA matmul, and derive the online-softmax update that makes FlashAttention memory-linear.
- After cuda-019: the reader can place raw CUDA, CUTLASS, Triton, cuBLAS, and PyTorch on one ladder and choose the rung for a given kernel.

## Gates

`series.yaml` declares no formal gates (`gates: []`). The de facto gates are the correctness checks each `.cu` and model print: no reader should pass cuda-003 without the tiled kernel matching the naive one, pass cuda-009 without the race model reproducing its lost-update table, or pass cuda-018 without the CPU model's FlashAttention output matching standard attention to tolerance.

# 3. Pedagogical rules

- One rhythm per post: picture, intuition, math, code. Every post opens with a TL;DR and an "After reading this you will be able to" list, then a hero diagram, numbered sections, Common pitfalls, Further reading, and What to read next.
- A kernel is described from one thread's point of view ("thread `i` does"), never as a loop; on the GPU the grid is the loop. The global index is always `blockIdx.x * blockDim.x + threadIdx.x`, introduced once per post.
- Real numbers only. Every figure quoted in prose is either measured on a named card (GTX 1650, RTX 3080, A100) or derived by a committed CPU model whose true output is pasted into the post. Where a table was measured separately and the shipped `.cu` only checks correctness, the post says so in the sentence after the table.
- Two companions per post. A `.py` model (standard library or NumPy, no GPU) reproduces the kernel's logic and prints the numbers; the `.cu` kernel builds with one `nvcc` line whose `-arch` the file header states. Readers without a GPU follow every post through the model.
- Naive first, then the fix, then the ceiling. Each kernel post writes the obvious kernel, shows where it leaks (strided writes, contended atomics, re-read tiles), applies one fix per section, and compares against the library (cuBLAS, cuDNN, CUB) to show the remaining gap honestly.
- Correctness before speed. Races, barriers, atomics, and fences are stated as rules when first used (cuda-004, cuda-005, cuda-008) and gathered into one post (cuda-009) before the performance half begins; cuda-011 proves correctness with compute-sanitizer before it profiles.
- Model before measurement. The roofline and occupancy are derived by counting (cuda-010) before the profiler reports them (cuda-011); the numbers section of every later post is read against that model.
- Hardware facts are dated to a card. Compute capability, `sm_XX`, warp size 32, 128-byte transactions, 32 shared-memory banks, and the Tensor Core and `cp.async` requirements are tied to the card they were measured on; the reader's card is expected to differ.
- No framework until the end. PyTorch, cuBLAS, cuDNN, and CUB appear as baselines in the numbers tables; CUTLASS and Triton are introduced only in cuda-019 after the reader has written by hand what they generate.
- One notation. `notation_guide.md` fixes `A`, `B`, `C` as code-span operand names, `TILE` and `s_A`, `s_B` for tiles, $I$ for arithmetic intensity and $\beta$ for peak bandwidth; prose, math, and code use the same identifier.

# 4. Curriculum lock

19 posts in 5 parts. Post numbers are permanent; a post inserted later takes the next ordinal from `series.yaml` and the directories after it renumber (STANDARD.md Section 2.2).

| Part | Title | Posts | Ids |
|---|---|---|---|
| I | Part I - First Kernels | 01-03 | cuda-001 to cuda-003 |
| II | Part II - Parallel Patterns | 04-09 | cuda-004 to cuda-009 |
| III | Part III - Understanding Performance | 10-11 | cuda-010 to cuda-011 |
| IV | Part IV - Systems & Execution | 12-15 | cuda-012 to cuda-015 |
| V | Part V - Tensor & Attention | 16-19 | cuda-016 to cuda-019 |

Retired ids: none.

# 5. Per-post specifications

One specification per shipped post, in `series.yaml` order. The sidecars of this series carry no `level`, `prerequisites`, or `recommended` keys, so the level and dependencies below are the lock of record until the migration pass copies them into the sidecars (Section 9).

## Part I - First Kernels

### cuda-001 - Why GPUs exist — the CUDA execution model from the ground up

- **Meta:** Part I; beginner; core; proof none; code included.
- **Central question:** Why does a GPU beat a CPU on data-parallel work, and how does a kernel's thread find its data?
- **Promise:** The reader can explain the GPU's bet in terms of chip area and latency hiding, map thread, block, and grid onto warp and streaming multiprocessor, compute any thread's global index by hand, say what surplus threads do, and predict when warp divergence halves throughput.
- **Coverage:** latency machine against throughput machine, host and device, the three-word hierarchy, the global index $i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$, SIMT and warps of 32, divergence, the memory hierarchy from registers to global memory, nvcc, PTX, cubin, and compute capability.
- **Dependencies:** requires none; recommends cmp-002; supports cuda-002, cuda-009.
- **Visuals:** hero `01-cpu-vs-gpu.svg` (a few large cores and cache against a dense grid of small cores); the thread, block, grid hierarchy; warp lockstep and a divergent branch; the memory hierarchy by latency.
- **Lab/gate:** Build and run `snippets/device_query.cu` to print the card's name, compute capability, SM count, and memory, and run `snippets/thread_model.py` to reproduce the global-index table and the divergence cost for a 256-thread block.
- **Anchors:** NVIDIA CUDA C++ Programming Guide; Harris, "An Even Easier Introduction to CUDA" (2017); NVIDIA CUDA C++ Best Practices Guide; Patterson and Hennessy, Computer Organization and Design.

### cuda-002 - Your first CUDA kernel — vector addition and the memory wall

- **Meta:** Part I; beginner; core; proof idea; code included.
- **Central question:** How does deleting the loop and launching a grid turn `C[i] = A[i] + B[i]` into a kernel, and why is that kernel memory bound?
- **Promise:** The reader can write the vector-addition kernel, pick a block size that is a multiple of 32, move data with `cudaMalloc`, `cudaMemcpy`, and `cudaFree`, guard surplus threads with `if (i < N)`, and predict from arithmetic intensity that the kernel is memory bound.
- **Coverage:** the grid is the loop, the launch syntax, the global index, the copy, launch, copy cycle, the `CUDA_CHECK` macro, honest benchmarking (CUDA events, warm-up, pinned memory, checksums), the memory wall at $I = 1/12$ FLOP per byte, one-off work against a pipeline.
- **Dependencies:** requires cuda-001; supports cuda-003, cuda-011, cuda-012.
- **Visuals:** hero `01-grid-is-the-loop.svg` (one CPU worker stepping through indices against a grid of threads each owning one); the two memory spaces and the copy cycle; the benchmark table by array size.
- **Lab/gate:** Run `snippets/vector_add.py` and reproduce the launch of 3,907 blocks of 256 threads over $N = 1{,}000{,}003$ with 189 surplus threads and $I = 0.0833$ FLOP per byte; build `snippets/vector_add.cu` and confirm the checksum.
- **Anchors:** Harris, "An Even Easier Introduction to CUDA" (2017); Harris, "Write Flexible Kernels with Grid-Stride Loops"; Harris, "How to Implement Performance Metrics in CUDA C/C++" (2012); Williams, Waterman and Patterson, "Roofline" (2009).

### cuda-003 - Matrix multiplication — from a memory-bound dot product to shared-memory tiling

- **Meta:** Part I; beginner; core; proof idea; code included.
- **Central question:** Why is the one-thread-per-output matmul memory bound, and how does a shared-memory tile cut the global traffic by the tile width?
- **Promise:** The reader can map a 2D output onto a 2D grid and write the `row` and `col` index, count loads per element to show the naive kernel is memory bound, write a tiled kernel with `__shared__` and `__syncthreads()` and say what each barrier protects, and read a naive against tiled against cuBLAS benchmark.
- **Coverage:** $C_{ij} = \sum_k A_{ik} B_{kj}$, 2D indexing with 16 by 16 blocks, the naive kernel and its re-reads, the `TILE` loop with two barriers, the measured 1.9 times gain and the 19.4 times cuBLAS gap, register tiling, `float4` loads, and double buffering as what cuBLAS adds.
- **Dependencies:** requires cuda-002; supports cuda-004, cuda-006, cuda-010, cuda-014, cuda-015, cuda-016, cuda-017, cuda-019.
- **Visuals:** hero `01-dot-product.svg` (a row of `A` and a column of `B` meeting at one element of `C`); the 2D grid over the output; the tile loop through shared memory; `04-matmul-perf.svg` (naive 135, tiled 258, cuBLAS 2616 GFLOPS).
- **Lab/gate:** Run `snippets/matmul_model.py` to reproduce the loads-per-element count and confirm the tiled result equals the naive one; build `snippets/matmul.cu` and read the GTX 1650 table (15.90 ms naive, 8.31 ms tiled, 0.82 ms cuBLAS at $1024^3$).
- **Anchors:** NVIDIA CUDA C++ Programming Guide, shared memory; Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (2022); NVIDIA CUTLASS; NVIDIA cuBLAS documentation.

## Part II - Parallel Patterns

### cuda-004 - Reduction — summing millions of numbers without a serial bottleneck

- **Meta:** Part II; intermediate; core; proof idea; code included.
- **Central question:** How do a thousand threads produce one sum without racing or serializing through an atomic?
- **Promise:** The reader can distinguish a race from a serial atomic sum, implement a shared-memory tree reduction with sequential addressing, reduce a warp with `__shfl_down_sync`, compose warp, block, and grid levels with a grid-stride loop, and read a bandwidth benchmark as a fraction of peak.
- **Coverage:** the race and the atomic trap, the tree of depth $\log N$, interleaved against sequential addressing and divergence counts, warp shuffle, `blockReduce` and one atomic per block, the GTX 1650 table (0.08, 108, 132 GB/s against a 160 GB/s peak).
- **Dependencies:** requires cuda-003; supports cuda-005, cuda-008, cuda-011, cuda-016.
- **Visuals:** hero `01-tree-reduction.svg` (eight values folding to one); `02-addressing.svg` (47 against 5 divergent warp-steps); the warp shuffle lanes; `04-reduction-perf.svg`.
- **Lab/gate:** Run `snippets/reduction_model.py` to reproduce the divergent warp-step counts for a 256-thread block and confirm the tree equals a plain sum; build `snippets/reduction.cu` and verify the warp-shuffle total against the host sum.
- **Anchors:** Harris, "Optimizing Parallel Reduction in CUDA" (2007); Luitjens, "Faster Parallel Reductions on Kepler" (2014); NVIDIA Programming Guide, warp shuffle functions; NVIDIA, "Using CUDA Warp-Level Primitives" (2018).

### cuda-005 - Histogram — beating atomic contention with privatization

- **Meta:** Part II; intermediate; core; proof idea; code included.
- **Central question:** When thousands of threads increment the same bins, how do you keep the count correct without letting contention make it serial?
- **Promise:** The reader can explain the lost-update race and why `atomicAdd` is correct but contention bound, predict from the data distribution when a global-atomic histogram collapses, write a privatized histogram with a per-block shared array and a three-phase init, count, merge, and place its two barriers.
- **Coverage:** counting as contention, the naive global-atomic kernel and its collapse on solid-color input (45 GB/s to 0.8 GB/s), privatization in shared memory, the merge, the measured robustness (180 and 120 GB/s), beyond 256 bins with multiple passes and warp aggregation.
- **Dependencies:** requires cuda-004; supports cuda-006, cuda-009, cuda-010.
- **Visuals:** hero `01-contention.svg` (eight threads queueing on one bin); the three-phase privatized kernel; `03-histogram-perf.svg` (global against shared on random and solid data).
- **Lab/gate:** Run `snippets/histogram_model.py` to reproduce the worst case (16,777,216 increments on one address against 65,536 shared increments per block, privatized equal to reference); build `snippets/histogram.cu` and confirm the count.
- **Anchors:** NVIDIA Programming Guide, atomic functions; NVIDIA Best Practices Guide, shared memory; Gomez-Luna et al., "An Optimized Approach to Histogram Computation on GPU" (2013); Adinetz, "Optimized Filtering with Warp-Aggregated Atomics" (2014).

### cuda-006 - Matrix transpose — coalescing, shared memory, and bank conflicts

- **Meta:** Part II; intermediate; core; proof idea; code included.
- **Central question:** Why does a kernel with no arithmetic run at 17 percent of copy bandwidth, and how do a shared-memory tile and one column of padding recover it?
- **Promise:** The reader can explain coalesced against strided access in 128-byte transactions, diagnose the naive transpose's strided writes, stage a 32 by 32 tile through shared memory so both global accesses coalesce, and recognise and remove a 32-way bank conflict with padding.
- **Coverage:** the 128-byte line and the 32-way penalty, the naive kernel, the tile that decouples read from write pattern, shared-memory banks and the `[32][33]` fix, the RTX 3080 table (60, 220, 310 GB/s against a 350 GB/s copy).
- **Dependencies:** requires cuda-003; recommends cuda-005; supports cuda-007, cuda-008, cuda-010.
- **Visuals:** hero `01-coalesced-vs-strided.svg` (one transaction against 32 cache lines); the tile load and transposed store; the bank-conflict column and the padded tile; `04-transpose-perf.svg`.
- **Lab/gate:** Run `snippets/transpose_model.py` to reproduce the transaction counts for a row and a column read and the bank-conflict count with and without padding; build `snippets/transpose.cu` and confirm correctness.
- **Anchors:** Harris, "An Efficient Matrix Transpose in CUDA C/C++" (2013); NVIDIA Programming Guide, device memory accesses; NVIDIA Best Practices Guide, shared memory and bank conflicts; Volkov, "Better Performance at Lower Occupancy" (2010).

### cuda-007 - Convolution — constant memory, shared-memory halos, and separable filters

- **Meta:** Part II; intermediate; core; proof idea; code included.
- **Central question:** When every output pixel reads an overlapping window of the input, where should the filter and the pixels live so each value is loaded once?
- **Promise:** The reader can write a 2D stencil and see why neighbours cause redundant reads, put a filter in constant memory and explain the warp broadcast, size and load an 18 by 18 shared tile with a one-pixel halo for a 16 by 16 output and compute the reuse factor, and recognise a separable filter's $K^2$ to $2K$ saving.
- **Coverage:** the stencil sum $\text{out}[y][x] = \sum_{i,j} \text{in}[y+i][x+j] \cdot \text{mask}[i+r][j+r]$, cross-correlation against true convolution, `__constant__` and `cudaMemcpyToSymbol`, the halo tile, the RTX 3080 table (2.1, 1.4, 0.9 ms against 0.3 ms cuDNN), separable passes.
- **Dependencies:** requires cuda-006; supports cuda-014.
- **Visuals:** hero `01-stencil.svg` (a 3 by 3 window over a 5 by 5 input producing one pixel); constant-memory broadcast; the tile with its halo ring; `04-conv-perf.svg`.
- **Lab/gate:** Run `snippets/convolution_model.py` to reproduce 2,304 naive against 324 tiled global reads (7.1 times reuse) and the 9 against 6 multiply-adds per pixel for a separable filter; build `snippets/convolution.cu` and confirm `convTiled` matches the reference.
- **Anchors:** NVIDIA Programming Guide, constant memory; Kirk and Hwu, Programming Massively Parallel Processors, chapter 7; Podlozhnyuk, "Image Convolution with CUDA" (2007); NVIDIA cuDNN Developer Guide.

### cuda-008 - Parallel scan — breaking the dependency chain of a prefix sum

- **Meta:** Part II; intermediate; core; proof guided; code included.
- **Central question:** How can a prefix sum, where every output waits on the one before it, finish in $\log N$ parallel steps?
- **Promise:** The reader can explain the work-for-depth trade, contrast Hillis-Steele ($O(N \log N)$ work) with Blelloch ($O(N)$ work, up-sweep then down-sweep) and pick by array size, walk both sweeps by hand, and implement stream compaction as predicate, exclusive scan, scatter.
- **Coverage:** the serial chain, the work and depth table, Hillis-Steele within a warp via `__shfl_up_sync`, the Blelloch up-sweep as the cuda-004 tree and the down-sweep, conflict-free addressing, multi-block scans, compaction, the RTX 3080 table (3.2, 1.8, 0.9 ms against 0.6 ms CUB).
- **Dependencies:** requires cuda-004; recommends cuda-006; supports cuda-009.
- **Visuals:** hero `01-scan-dependency.svg` (the running total and its chain of arrows); `02-blelloch.svg` (input, after up-sweep, after down-sweep); `03-stream-compaction.svg` (predicate, scan, scatter); `04-scan-perf.svg`.
- **Lab/gate:** Run `snippets/scan_model.py` to reproduce the Blelloch sweeps on a small array and the compaction positions, and count the additions each algorithm performs; build `snippets/scan.cu` and confirm the single-block scan.
- **Anchors:** Harris, Sengupta and Owens, "Parallel Prefix Sum (Scan) with CUDA" (GPU Gems 3, 2007); Blelloch, "Prefix Sums and Their Applications" (1990); Hillis and Steele, "Data Parallel Algorithms" (1986); NVIDIA CUB DeviceScan.

### cuda-009 - When threads share data — barriers, atomics, and the memory model

- **Meta:** Part II; intermediate; core; proof idea; code included.
- **Central question:** What exactly do `__syncthreads`, `__syncwarp`, atomics, fences, and scopes each guarantee, and what does none of them guarantee?
- **Promise:** The reader can explain and simulate the lost-update race, place a block barrier correctly and say why one inside divergent code deadlocks, use `atomicAdd` and `atomicCAS` and separate atomicity from ordering, reason about `__threadfence` and block, device, and system scopes with an acquire and release intuition, and say why only a new launch is a global barrier.
- **Coverage:** read, modify, write and the lost update, the race model's schedules (serial, warp-lockstep, full-lockstep), `__syncthreads` rules, independent thread scheduling and the active mask, atomics as correctness not ordering, fences and scopes, no grid-wide barrier inside a kernel.
- **Dependencies:** requires cuda-005, cuda-008; recommends cuda-001; supports cuda-010.
- **Visuals:** hero `01-race-condition.svg` (two threads read 7, both write 8, one increment lost); the barrier inside divergent control flow; the acquire and release picture of a fence; the scope ladder.
- **Lab/gate:** Run `snippets/race_model.py` and reproduce the table for $T = 1024$ threads and $K = 100$ increments (102,400 intended; 3,200 under warp lockstep; 100 under full lockstep; 102,400 with the atomic); build `snippets/sync.cu` and confirm the atomic count.
- **Anchors:** NVIDIA Programming Guide, memory fence functions and atomic functions; NVIDIA Programming Guide, independent thread scheduling; Lin and Grover, "Using CUDA Warp-Level Primitives" (2018); Sorensen et al., "Portable Inter-Workgroup Barrier Synchronisation for GPUs" (2016); Boehm and Adve, "Foundations of the C++ Concurrency Memory Model" (2008).

## Part III - Understanding Performance

### cuda-010 - How fast can a kernel be? — roofline, arithmetic intensity, and occupancy

- **Meta:** Part III; intermediate; core; proof guided; code included.
- **Central question:** What is the fastest a kernel could run on this card, and which resource stops it?
- **Promise:** The reader can count a kernel's arithmetic intensity $I$ and place it on the roofline, say whether it is memory or compute bound and which lever moves it, derive theoretical occupancy from registers, shared memory, and block size, and explain why higher occupancy stops helping and a low-occupancy kernel can still be fast.
- **Coverage:** $I$ in FLOP per byte for vector add ($1/12$), reduction, naive matmul ($0.25$), and tiled matmul, the two roofs and the ridge point of the GTX 1650, the three occupancy limiters, occupancy against instruction-level parallelism, what the roofline does not see (launch overhead, latency-bound and small kernels).
- **Dependencies:** requires cuda-003, cuda-006; recommends cuda-005, cuda-009; supports cuda-011, cuda-013, cuda-014, cuda-015.
- **Visuals:** hero `01-roofline.svg` (log-log roofline with vector add on the slope and tiled matmul toward the ridge); the occupancy limiter table; the diminishing-returns curve of latency hiding.
- **Lab/gate:** Run `snippets/roofline_model.py` to reproduce the intensity and ceiling of each series kernel on the GTX 1650 limits, then compute by hand the occupancy of one of the reader's own kernels from its register and shared-memory counts.
- **Anchors:** Williams, Waterman and Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009); NVIDIA Best Practices Guide; NVIDIA CUDA Occupancy Calculator and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; Volkov, "Better Performance at Lower Occupancy" (2010).

### cuda-011 - Profiling and debugging — Nsight, compute-sanitizer, and the occupancy you can measure

- **Meta:** Part III; intermediate; core; proof none; code included.
- **Central question:** How do you prove a kernel is correct and then measure how close it gets to the ceiling cuda-010 drew?
- **Promise:** The reader can run the profile, do not guess loop, catch out-of-bounds writes and shared-memory races with `compute-sanitizer`, choose Nsight Systems for the timeline and Nsight Compute for per-kernel detail and run each, and read achieved occupancy and memory throughput against the roofline model.
- **Coverage:** the four-stage loop, memcheck, racecheck, and synccheck on the series' own kernels, `nsys` against `ncu`, measured against theoretical occupancy, reading the roofline chart in Nsight Compute, the build-and-measure recipe.
- **Dependencies:** requires cuda-010; recommends cuda-002, cuda-004; supports cuda-012.
- **Visuals:** hero `01-profiling-loop.svg` (write, measure, read the bottleneck, fix); the sanitizer report for the missing guard; the Nsight Compute occupancy and throughput panels.
- **Lab/gate:** Build `snippets/occupancy_demo.cu`, run it under `compute-sanitizer` and `ncu`, and compare the measured occupancy and DRAM throughput with the cuda-010 model's prediction for the same launch.
- **Anchors:** NVIDIA Nsight Compute documentation; NVIDIA Nsight Systems documentation; NVIDIA Compute Sanitizer documentation; NVIDIA Best Practices Guide.

## Part IV - Systems & Execution

### cuda-012 - CUDA streams — overlapping copy and compute for free throughput

- **Meta:** Part IV; intermediate; core; proof idea; code included.
- **Central question:** Why does the default stream leave two of the GPU's three engines idle, and how do streams and pinned memory overlap them?
- **Promise:** The reader can explain why serial execution wastes two-thirds of the engines, pipeline a workload across streams so copy and compute overlap, say why `cudaMemcpyAsync` needs pinned memory and what happens without it, and predict where streams help most and least.
- **Coverage:** the three engines and full-duplex PCIe, a stream as an ordered queue, pinned against pageable memory, chunking into four streams, the RTX 3080 table (45.2, 32.1, 12.5 ms over 256 MB), bandwidth-bound against compute-bound workloads.
- **Dependencies:** requires cuda-002; recommends cuda-011; supports cuda-013.
- **Visuals:** hero `01-serial-vs-pipeline.svg` (twelve serial blocks against three staggered engine rows with a steady-state band); the pinned-memory path; `04-streams-perf.svg`.
- **Lab/gate:** Run `snippets/streams_model.py` to reproduce the engine-utilization and pipeline-timing arithmetic, then build `snippets/streams.cu` and read its streamed time against the serial one.
- **Anchors:** Harris, "How to Overlap Data Transfers in CUDA C/C++" (2012); NVIDIA Programming Guide, streams; NVIDIA Best Practices Guide, asynchronous transfers; NVIDIA Nsight Systems User Guide.

### cuda-013 - Cutting launch overhead — CUDA Graphs and stream-ordered allocation

- **Meta:** Part IV; intermediate; core; proof idea; code included.
- **Central question:** When a workload is many tiny kernels, why is the CPU the bottleneck, and how do graphs and memory pools remove the two host-side stalls?
- **Promise:** The reader can explain why many small kernels are bound by launch overhead rather than compute, capture a stream into a graph and replay it with one `cudaGraphLaunch`, say when graphs help and when they do not, and replace device-synchronizing `cudaMalloc` and `cudaFree` with `cudaMallocAsync` on a stream-ordered pool.
- **Coverage:** the microseconds per launch, `cudaStreamBeginCapture` to `cudaStreamEndCapture`, instantiate once and replay many, the break-even against a few large kernels, `cudaMalloc` as a device-wide synchronisation, memory pools and trimming.
- **Dependencies:** requires cuda-012; recommends cuda-010; supports none.
- **Visuals:** hero `01-launch-overhead.svg` (host gaps before every kernel against one packed replay); the captured graph of nodes; the allocation stall across streams.
- **Lab/gate:** Run `snippets/graphs_model.py` to reproduce the launch-overhead arithmetic and the break-even point; build `snippets/graphs.cu` and compare direct launches with graph replay.
- **Anchors:** NVIDIA Programming Guide, CUDA Graphs; NVIDIA, "Getting Started with CUDA Graphs" (2019); NVIDIA, "Using the NVIDIA CUDA Stream-Ordered Memory Allocator" (2021); NVIDIA Programming Guide, stream ordered memory allocator.

### cuda-014 - Kernel fusion — the fastest kernel is the one you don't launch

- **Meta:** Part IV; intermediate; core; proof idea; code included.
- **Central question:** Why does `relu(conv(x))` as two kernels waste bandwidth on an intermediate nobody needs, and when is fusing them free?
- **Promise:** The reader can explain why separate elementwise kernels waste bandwidth, fuse a chain to read once, compute in registers, and write once, use arithmetic intensity to say which operations are free to fuse, and recognise the register-pressure limit and when not to fuse.
- **Coverage:** the intermediate round trip, traffic saved as $(N-1)/N$ for a chain of $N$, the fused kernel, memory-bound elementwise ops hiding inside a compute-bound producer, register pressure, representative published speedups from 1.07 times to 3.33 times, `torch.compile`, cuDNN fused ops, and FlashAttention as fusion at its limit.
- **Dependencies:** requires cuda-010; recommends cuda-003, cuda-007; supports cuda-015, cuda-018.
- **Visuals:** hero `01-fusion-traffic.svg` (the intermediate written and re-read against staying in registers); the register-pressure trade-off; `04-fusion-perf.svg`.
- **Lab/gate:** Run `snippets/fusion_model.py` to reproduce the byte counts of the fused and unfused chains and confirm the fused output equals the two-kernel one; build `snippets/fusion.cu` and confirm correctness.
- **Anchors:** Dao et al., "FlashAttention" (2022); NVIDIA cuDNN Developer Guide, fused operations; PyTorch, "Introduction to torch.compile"; Williams, Waterman and Patterson, "Roofline" (2009).

### cuda-015 - Asynchronous copy and software pipelining — cp.async, double buffering, and hiding memory latency

- **Meta:** Part IV; advanced; core; proof guided; code included.
- **Central question:** How does the tiled matmul stop waiting for its next tile, and how much latency does a double-buffered `cp.async` loop hide?
- **Promise:** The reader can point at the stall in a synchronous tiled kernel, use the `cp.async` model of issue, commit, and wait, write a ping-pong tile loop that prefetches tile $k+1$ while computing tile $k$, predict the latency-hiding speedup from the load-to-compute ratio, and state the Ampere (`sm_80`) requirement.
- **Coverage:** the load, barrier, compute, repeat stall, `cp.async` bypassing the register file, two shared tiles and the pipeline stages, software pipelining generalised to prologue, steady state, epilogue, the multi-stage pipelines of cuBLAS and CUTLASS.
- **Dependencies:** requires cuda-003; recommends cuda-010, cuda-014; supports cuda-017, cuda-019.
- **Visuals:** hero `01-stall-vs-pipeline.svg` (four load and compute pairs with red stall bubbles against staggered loads with a green steady state); the two-buffer ping-pong; the `cp.async` group timeline.
- **Lab/gate:** Run `snippets/pipeline_model.py` to reproduce the hidden-latency fraction for the series' load-to-compute ratios; on an `sm_80` or newer card build `snippets/async_pipeline.cu` and confirm it matches the synchronous tiled kernel.
- **Anchors:** NVIDIA Programming Guide, asynchronous data copies; NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020); NVIDIA CUTLASS, efficient GEMM; Allan et al., "Software Pipelining" (1995).

## Part V - Tensor & Attention

### cuda-016 - Numbers on a GPU — FP32, FP16, BF16, and numerical stability

- **Meta:** Part V; intermediate; core; proof guided; code included.
- **Central question:** How does a float spend its bits between range and precision, and what does that do to a parallel sum, an FP16 product, and a softmax?
- **Promise:** The reader can read a float as sign, exponent, and mantissa and say which formats trade range for precision, explain why a parallel reduction is not bit-for-bit reproducible, apply mixed precision (FP16 multiply, FP32 accumulate) and the stable-softmax max subtraction, and choose a test tolerance against an FP64 reference.
- **Coverage:** FP32, FP16, BF16, and TF32 bit budgets, non-associativity and block-count-dependent sums, overflow at 65,504 and underflow, mixed precision, $\text{softmax}(x) = \exp(x - \max x) / \sum \exp(x - \max x)$, pairwise and Kahan summation, tolerances.
- **Dependencies:** requires cuda-004; recommends cuda-003, cmp-035; supports cuda-017.
- **Visuals:** hero `01-float-formats.svg` (four bit strips with BF16's wide exponent highlighted); the order-dependent sum; the overflow cliff of FP16; the stable softmax.
- **Lab/gate:** Run `snippets/precision_model.py` to reproduce the sum that differs by summation order and block count, the FP16 overflow, and the stable against naive softmax; build `snippets/precision.cu` and confirm the device results agree within tolerance.
- **Anchors:** Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991); IEEE 754; Micikevicius et al., "Mixed Precision Training" (2018); Higham, Accuracy and Stability of Numerical Algorithms (2002); NVIDIA, "Train With Mixed Precision".

### cuda-017 - Tensor cores — matrix-at-once hardware and mixed precision

- **Meta:** Part V; advanced; core; proof idea; code included.
- **Central question:** How does one warp instruction compute a 16 by 16 by 16 matrix multiply-accumulate, and what does FP16 input cost in accuracy?
- **Promise:** The reader can explain how a Tensor Core differs from a CUDA core and why it is denser, use FP16 multiply with FP32 accumulate and say why it is accurate, write a WMMA matmul with fragments, `load_matrix_sync`, `mma_sync`, and `store_matrix_sync`, and avoid the alignment, dimension, and warp-synchrony traps.
- **Coverage:** the missing TFLOPS on the RTX 3080 (10.7 FP32 against about 119 FP16 peak), $D = A \cdot B + C$ as 4,096 fused multiply-adds per instruction, mixed precision, the WMMA fragment model, BF16 and TF32 for training, the table from 3.0 to 124.9 TFLOPS.
- **Dependencies:** requires cuda-003, cuda-016; recommends cuda-015; supports cuda-018.
- **Visuals:** hero `01-cuda-vs-tensor.svg` (one FMA per thread against a 16 by 16 by 16 tile per warp instruction); the fragment load, multiply, store cycle; `04-tensor-perf.svg`.
- **Lab/gate:** Run `snippets/tensor_core_model.py` to reproduce the per-instruction FMA count and the mixed-precision error against pure FP16; on a compute capability 7.0 or newer card build `snippets/tensor_core.cu` and confirm the WMMA result against the FP32 reference.
- **Anchors:** NVIDIA Programming Guide, warp matrix functions (WMMA); Markidis et al., "NVIDIA Tensor Core Programmability, Performance and Precision" (2018); Micikevicius et al., "Mixed Precision Training" (2017); NVIDIA, "Programming Tensor Cores in CUDA 9" (2017).

### cuda-018 - FlashAttention — making attention memory-linear with tiling and online softmax

- **Meta:** Part V; advanced; core; proof guided; code included.
- **Central question:** How can $\text{softmax}(QK^\top / \sqrt{d})\,V$ be computed exactly without ever materialising the $N \times N$ score matrix?
- **Promise:** The reader can explain why standard attention is memory bound and quadratic, derive the online-softmax update (running max, running sum, rescale), describe the tiling with the outer loop over $Q$ blocks and the inner loop over $K$ and $V$ blocks, and say why recomputation in the backward pass is a net win.
- **Coverage:** the 2 GB score matrix at 32K tokens and the two writes of it, the online-softmax recurrence, the SRAM tile loop, the A100 table (1.7 times at 512 tokens to 15.6 times at 16K and out of memory at 32K), 32 times to 512 times memory reduction, recomputation for training.
- **Dependencies:** requires cuda-017; recommends cuda-014, tf-003; supports cuda-019.
- **Visuals:** hero `01-quadratic-wall.svg` (the score matrix written twice against only $O$ written); the online-softmax rescaling step; the $Q$, $K$, $V$ block loops; `04-flash-perf.svg`.
- **Lab/gate:** Run `snippets/flash_attention.py` and confirm the tiled online-softmax output matches standard attention to tolerance while its peak memory grows linearly in $N$; on an `sm_80` card build `snippets/flash_attention.cu` and confirm the same against the reference.
- **Anchors:** Dao et al., "FlashAttention" (2022); Dao, "FlashAttention-2" (2023); Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018); Rabe and Staats, "Self-attention Does Not Need O(n^2) Memory" (2021).

### cuda-019 - CUTLASS and Triton — why nobody writes raw CUDA for matmul anymore

- **Meta:** Part V; advanced; survey; proof none; code included.
- **Central question:** Having written a tiled Tensor Core matmul by hand, which tool should actually produce a production kernel, and why was the hand-written one worth learning?
- **Promise:** The reader can place raw CUDA, CUTLASS, Triton, cuBLAS, and PyTorch on the abstraction ladder, read a Triton kernel and explain block-level against thread-level programming, describe the CUTLASS device, threadblock, warp, and MMA tiling hierarchy, and choose the right tool for a given kernel.
- **Coverage:** what a production GEMM adds (multi-level tiling, double buffering, async copies, epilogue fusion, split-K, per-generation tuning), the CUTLASS template hierarchy and CuTe, a Triton matmul, the A100 table from 165 to 275 TFLOPS against effort, the rule for standard, custom, and exotic kernels.
- **Dependencies:** requires cuda-018; recommends cuda-003, cuda-015; supports none.
- **Visuals:** hero `01-abstraction-ladder.svg` (five rungs trading control for ease, Triton highlighted); the CUTLASS hierarchy mapped onto the series' tiles; `04-tool-perf.svg`.
- **Lab/gate:** Run `snippets/tools_model.py` to reproduce the effort and throughput comparison; on a Linux machine with Triton run `snippets/triton_matmul.py` and confirm it matches `torch.matmul`, then map each line of the Triton kernel to the cuda-003 and cuda-015 kernels it replaces.
- **Anchors:** Tillet, Kung and Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2019); NVIDIA CUTLASS documentation; OpenAI Triton tutorials; Thakkar et al., "CUTLASS 3.x and CuTe" (GTC).

# 6. Series-specific contracts

## Notation

- `notation_guide.md` is the single source of truth. Concrete kernel operands are code spans `A`, `B`, `C` (and `d_A`, `d_B` for device copies), never typeset bold; bold $\mathbf{X}$, $\mathbf{W}$ is reserved for the abstract linear-algebra framing, and typeset $\mathbf{B}$ is banned as a matrix name because it is the bias alias of the family table.
- The global index is always `blockIdx.x * blockDim.x + threadIdx.x`, named `i` in 1D and `(row, col)` in 2D; `TILE` is the tile width, `s_A` and `s_B` the shared tiles; $I$ is arithmetic intensity in FLOP per byte and $\beta$ peak DRAM bandwidth in GB/s.
- Every shape is stated on first use (`A: [M, K]`, `tile: [TILE, TILE]`); every acronym is defined on first use; American spelling.

## Claim standards and measured numbers

- Every number is one of three kinds and the post says which: derived by the committed CPU model (whose true output is pasted), measured on a named card (GTX 1650 for cuda-002 to cuda-005 and the cuda-010 limits; RTX 3080 for cuda-006, cuda-007, cuda-008, cuda-012, cuda-017; A100 for cuda-018), or a representative published figure (cuda-014, cuda-019), labelled as such in the sentence after the table.
- Where the shipped `.cu` checks correctness only and the table was timed separately, the post states it (cuda-005, cuda-006, cuda-007, cuda-008, cuda-012, cuda-014, cuda-018).
- GPU wall-clock timings are reported with the card and compute capability; the reader's card is expected to differ, and the shape of the table, not the digits, is the lesson.

## Compute and hardware tiers

- `vocab.compute` is `[cpu, gpu-small]`: every `.py` model runs on a CPU with Python 3.9 and NumPy; every `.cu` kernel runs on one consumer or one datacenter GPU, never a cluster.
- `vocab.hardware` is `[gtx-1650, rtx-3080, a100]`, the three cards the series measured on. Feature floors: Tensor Cores and `__syncwarp` semantics need compute capability 7.0 or above (cuda-017); `cp.async`, BF16, and `flash_attention.cu` need `sm_80` (cuda-015, cuda-018); Triton needs Linux (cuda-019). Snippets default to `-arch=sm_75`.
- The sidecars do not yet carry `compute` or `hardware`; the migration pass fills them from this section.

## Code companions (no library)

- `library: null`. There is no shared code library; each post's `snippets/` holds one `.py` model and, for cuda-001 to cuda-018, one `.cu` kernel built with a single `nvcc` line whose `-arch` the header comment states. cuda-011 ships only `occupancy_demo.cu`; cuda-019 ships only Python (`tools_model.py`, `triton_matmul.py`).
- `examples/` holds standalone example kernels, the CMake build that compiles every post's `.cu` into one executable per kernel, the benchmark harness that produced the cuda-002 table, and practice challenges (vector add, ReLU, matrix add, transpose, reverse).
- RUNNING.md is the per-post run recipe, SETUP.md the toolchain guide, TROUBLESHOOTING.md the build-failure catalogue, CHEATSHEET.md the one-page reference, GLOSSARY.md and REFERENCES.md the shared term and citation lists.

## Interactives

- `kernel-visualizer/` is a separate web application beside the posts and is not referenced by any post body or sidecar; it is outside this lock and the posts make no claim that depends on it.

## Series-specific checks

- No `tools/lint_repo.py` exists yet; the family validator runs directly. Candidates for series checks: every `.cu` header states its `-arch`; every quoted table names its card or its model; every post's "What to read next" cites only same-series posts by number.

# 7. Sibling handoffs and deliberate omissions

## Assumed from siblings

- compute: C and Python at the level of pointers, loops, and running a script, and how a CPU executes a program (cmp-002, recommended by cuda-001); floating-point arithmetic as a first pass (cmp-035, recommended by cuda-016); profiling as a discipline (cmp-064) and the GPU intuition post (cmp-072) are the compute series' own treatments and are not required here.
- transformer: the attention equation and why it is quadratic (tf-003, recommended by cuda-018). The Transformer series' Phase III (tf-030 to tf-042) teaches GPU architecture, vector addition, matmul, coalescing, reduction and softmax, FlashAttention, and kernel fusion from the inference side; this series is the general GPU-programming treatment that those posts can cite, and it does not teach tokenization, the decoder stack, KV caches, or quantized kernels.

## Left to siblings

- transformer: FlashAttention inside an inference engine (tf-039, tf-040), the GEMV decode kernel (tf-034), quantized kernels (tf-042), the C++ inference engine and its CUDA backend (tf-060 to tf-066), distributed and mixed-precision training (tf-043 to tf-047).
- cnn-from-scratch: the convolution as a learned layer, im2col, and backward passes; cuda-007 treats convolution only as a stencil with a fixed filter.
- nnfs and machine-learning: what the matmul, softmax, and reduction are for; this series treats them as kernels.
- numerical-linear-algebra: blocked algorithms, conditioning, and error analysis beyond the cuda-016 tolerance rule.

## Deliberate omissions

- Multi-GPU and distributed execution (NCCL, NVLink topologies), unified memory, and dynamic parallelism: named where relevant, not taught.
- Cooperative groups and grid-wide synchronisation: cuda-009 explains why there is no in-kernel global barrier and stops there.
- Sorting, sparse matrices, and graph algorithms: scan (cuda-008) is shown as their engine; the algorithms themselves are left to discrete-math-and-algorithms.
- The FlashAttention backward kernel, Winograd and FFT convolution, and implicit GEMM: named in cuda-018 and cuda-007 as what the libraries add, not implemented.
- Writing CUTLASS templates or Triton autotuners in anger: cuda-019 reads them and places them; a production kernel is left to the reader's own project.

# 8. Projects

No projects are locked. `series.yaml` lists `projects: []` and `next_project_ordinal: 1`. The natural first project, after cuda-018, is a from-scratch attention kernel benchmarked against `torch.nn.functional.scaled_dot_product_attention` across sequence lengths with the roofline of cuda-010 as the judge; it is recorded here as a candidate, not a lock, and takes `cuda-p01` when it is written.

# 9. Risks and decisions

## Risk register

- Sidecars are minimal. Every sidecar carries only slug, title, date, tags, hero, reading_time, and part (plus the new id); the family keys `summary`, `updated`, `status`, `level`, `prerequisites`, `code`, and `outcomes` are missing. This plan is the record of level and dependencies until the migration pass copies them in. Mitigation: Section 10 step 2.
- Measured tables span three cards. Parts I and II mix GTX 1650 and RTX 3080 numbers, and Part V quotes an A100; a reader comparing tables across posts can be misled. Mitigation: every table names its card, and Section 6 lists which post used which.
- Several tables are not reproduced by the shipped `.cu`. cuda-005, cuda-006, cuda-007, cuda-008, cuda-012, and cuda-018 timed their tables separately; cuda-014 and cuda-019 quote published figures. Mitigation: each post says so in the sentence after the table; a benchmark harness per post is future work.
- Hardware floors exclude some readers from the `.cu` of cuda-015, cuda-017, and cuda-018. Mitigation: the CPU models reproduce the logic and the counted numbers without a GPU.
- The `kernel-visualizer/` application and the `examples/` challenges are untracked or partly tracked and are outside this lock; a later decision may register the challenges as projects.
- The posts do not carry prerequisites lines or a "Safe to skip?" test; readers entering mid-series rely on "What to read next" and this plan.

## Decisions locked

- Five parts with the README's titles, hyphenated (`Part IV - Systems & Execution`, `Part V - Tensor & Attention`; the ampersands are kept because they are the README's and the sidecars' spelling).
- Levels: cuda-001 to cuda-003 beginner; cuda-004 to cuda-014 and cuda-016 intermediate; cuda-015, cuda-017, cuda-018, cuda-019 advanced. These are this plan's assignment, since no sidecar carries a level.
- Scope: core for every post except cuda-019, which is `survey` because it reads and places tools rather than building one.
- Dependencies are the minimal hard chain the posts' own cross-references support (the cuda-004 tree feeds cuda-008; cuda-005 and cuda-008 feed cuda-009; cuda-003 and cuda-006 feed cuda-010; cuda-016 and cuda-003 feed cuda-017), with the "What to read next" back-references as recommends. The old 13-post PLAN.md's dependency list is superseded.
- Cross-series references are limited to locked series: cmp-002 and cmp-035 as recommends, tf-003 as a recommend of cuda-018. The Transformer Phase III overlap is described in Section 7, not encoded as dependencies.
- No projects, no gates, `status: complete` in `series.yaml` because all nineteen posts the README promises are shipped and no planned post exists.
- `vocab.compute` and `vocab.hardware` are declared now so the migration pass has an allowed list to fill the sidecars from.

## Recorded divergences

- Titles: every sidecar title keeps its `NN · ` prefix and an em-dash subtitle (`Why GPUs exist — the CUDA execution model from the ground up`), and the spec headings reproduce them verbatim; the body `# ` heading repeats the prefix. House style (sentence case, no em-dashes, no number prefix) is work for each post's migration commit.
- Body template: posts open with a blockquote TL;DR and "After reading this you will be able to" rather than the family's prerequisites line and sidecar `outcomes`; none has an Exercises section (compliant with 2.1.0) and none has a questions bank. The outcomes lists will seed `outcomes` in the migration.
- Part strings: the sidecars spelled parts with an em-dash (`Part I — First Kernels`); this lock rewrote them to the hyphen form in all nineteen sidecars.
- The old PLAN.md (13 posts in five phases, two published) is superseded by the 19 shipped posts; it was already deleted in the working tree and is preserved from git history as `audits/plan-pre-lock.md`. Its Blog 9 (streams) became cuda-012, Blog 10 (fusion) cuda-014, Blog 11 (Tensor Cores) cuda-017, Blog 12 (FlashAttention) cuda-018, Blog 13 (CUTLASS and Triton) cuda-019; cuda-009, cuda-010, cuda-011, cuda-013, cuda-015, and cuda-016 have no counterpart there.
- SETUP.md and RUNNING.md paths: SETUP.md steps 4 and 5 build from `blog/posts/...` while the directory is `posts/...`; RUNNING.md and README.md use `posts/`. Recorded for the maintainer; not a plan matter.
- cuda-002 cites its benchmark harness under `examples/src/Vector Addition/`, a directory with a space beside the hyphenated challenge folders; recorded, not changed.
- cuda-006 reports RTX 3080 numbers while cuda-001's reference card says every number is on a GTX 1650 unless stated; the post states it, so the claim holds, but the reference card should say "the early posts".

# 10. Roadmap

All 19 posts are drafted and shipped; nothing is planned. Next, in order:

1. Adopt this lock: `series.yaml` written by `assign_ids.py` and hand-edited (title, `status: complete`, `levels_used`, `vocab`), ids in every sidecar, this plan as the curriculum of record.
2. Sidecar migration (`series-standard/tools/migrate_2_1_0.py` and a hand pass): add `summary` from each TL;DR, `updated`, `status`, `level` and `prerequisites` and `recommended` from Section 5, `code: included`, `outcomes` from each "After reading" list, `compute` and `hardware` from Section 6; retitle to house style and drop the number prefix; fix the SETUP.md `blog/posts` paths.
3. Banks: write `questions.json` for every post to the family floor, seeded from the pitfalls and the lab of each spec.
4. Reproducibility: add a timing harness to the posts whose tables were measured separately (cuda-005 to cuda-008, cuda-012, cuda-018) so the shipped `.cu` prints its own table on the reader's card.
5. Project: decide whether the attention-kernel benchmark of Section 8 becomes `cuda-p01`.

Lock commit: `Lock the curriculum`, tag `lock-1.0.0`.
