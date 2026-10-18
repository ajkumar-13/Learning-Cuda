# Glossary

The terms used across Learning CUDA, one entry each, grouped by topic and alphabetical within a group. Each entry names the post that introduces the term. The fixed names and symbols (the operand names, the global-index formula, `TILE`, `s_A` and `s_B`, $I$ and $\beta$, the three kinds of number, the three cards) are set out in the [notation guide](notation_guide.md). Terms the series shares with the `compute` and `transformer` series are defined here as this series uses them, from the side of the kernel.

---

## Execution model

**Block (thread block).** A group of up to 1,024 threads that runs on one streaming multiprocessor, shares that SM's on-chip shared memory, and can synchronize with `__syncthreads()`. Its size is `blockDim.x` (and `.y`, `.z`), chosen as a multiple of 32. See [post 01](posts/01-introduction-to-cuda/index.md).

**Compute capability.** The version number of a GPU's architecture and feature set, written `major.minor` (7.5 for the GTX 1650, 8.6 for the RTX 3080, 8.0 for the A100). It decides which instructions a kernel may use and the `-arch=sm_XX` flag it is built with. See [post 01](posts/01-introduction-to-cuda/index.md).

**Global index.** A thread's position in the whole launch, always written `blockIdx.x * blockDim.x + threadIdx.x` and named `i` in 1D; the 2D form computes `row` and `col` the same way from the `.y` and `.x` components. See [post 01](posts/01-introduction-to-cuda/index.md) and [post 02](posts/02-vector-addition/index.md).

**Grid.** All the blocks launched by one kernel call; its size is `gridDim.x` (and `.y`, `.z`). Blocks run independently and in any order, which is why there is no barrier across the grid inside a running kernel. See [post 01](posts/01-introduction-to-cuda/index.md).

**Kernel.** A `__global__` function that runs on the GPU, written from the point of view of one thread and launched over a grid with `kernel<<<blocksPerGrid, blockSize>>>(args)`. On the GPU the grid is the loop. See [post 01](posts/01-introduction-to-cuda/index.md).

**Latency hiding.** Keeping enough independent work in flight, from many resident warps or from independent loads within each thread, that while some instructions wait hundreds of cycles for global memory the scheduler issues others; it is the GPU's answer to memory latency, where a CPU answers with large caches. See [post 01](posts/01-introduction-to-cuda/index.md) and [post 10](posts/10-roofline/index.md).

**nvcc, PTX, cubin.** `nvcc` is the CUDA compiler driver: it splits a `.cu` file into host code (passed to the host C++ compiler) and device code, which it compiles to PTX, a virtual instruction set, and then to a cubin, the machine code for one architecture. See [post 01](posts/01-introduction-to-cuda/index.md).

**Occupancy.** The ratio of warps resident on a streaming multiprocessor to the most it can hold. Theoretical occupancy is capped by whichever runs out first of the block size, the registers per thread and the shared memory per block; higher occupancy gives the scheduler more warps to hide latency with, up to a point. Derived by counting in [post 10](posts/10-roofline/index.md) and checked with the CUDA runtime's occupancy API in [post 11](posts/11-profiling-debugging/index.md); achieved occupancy, the warps actually resident during a run, is a performance-counter reading that Nsight Compute reports.

**SIMT (single instruction, multiple threads).** The GPU's execution style: the threads of a warp run one instruction stream together, each on its own data and registers. See [post 01](posts/01-introduction-to-cuda/index.md).

**Streaming multiprocessor (SM).** The hardware unit that runs blocks: it holds the warp schedulers, the arithmetic units, the register file and the shared memory. A GPU has many (14 on the GTX 1650). See [post 01](posts/01-introduction-to-cuda/index.md).

**Thread.** One instance of a kernel, the smallest unit of execution; it finds its data through `threadIdx` and `blockIdx`. See [post 01](posts/01-introduction-to-cuda/index.md).

**Warp.** A group of 32 threads of one block that the hardware schedules together and that issue one instruction at a time; from compute capability 7.0 each thread keeps its own program counter, so the lanes are not guaranteed to stay in lockstep. See [post 01](posts/01-introduction-to-cuda/index.md).

**Warp divergence.** The cost of a branch that splits a warp: the hardware runs each taken path in turn with the other lanes idle, so a two-way split roughly halves throughput for that code. See [post 01](posts/01-introduction-to-cuda/index.md) and [post 04](posts/04-reduction/index.md).

## Memory

**Bank conflict.** Shared memory is divided into 32 banks, and when two or more lanes of a warp touch different addresses in the same bank, the accesses serialize. The usual fix is padding a tile by one column (`[TILE][TILE + 1]`). See [post 06](posts/06-matrix-transpose/index.md).

**Coalescing.** When the 32 threads of a warp read or write consecutive addresses, the hardware serves the warp with as few 32-byte transactions (sectors) as the data needs on compute capability 6.0 and later: four, one 128-byte line, for 32 consecutive floats. A strided pattern puts each lane in its own sector: 32 sectors, 8 times the bytes, for 4-byte words. See [post 02](posts/02-vector-addition/index.md) and [post 06](posts/06-matrix-transpose/index.md).

**Constant memory.** A 64 KB read-only space cached on each SM (an 8 KB working set on compute capability 7.5); a warp's read costs one request per distinct address, so a value every lane reads at once is broadcast in one request; used for filter weights. See [post 07](posts/07-convolution/index.md).

**Global memory.** The card's main memory, off-chip DRAM (GDDR on consumer cards, HBM on datacenter ones): large, slow (hundreds of cycles), and the only tier every thread and the host can reach. See [post 01](posts/01-introduction-to-cuda/index.md).

**Host and device.** The CPU with its system memory, and the GPU with its global memory: separate address spaces, joined by copies across PCIe (`cudaMemcpy`). Device copies of an array are named `d_A`, `d_B`. See [post 01](posts/01-introduction-to-cuda/index.md) and [post 02](posts/02-vector-addition/index.md).

**L2 cache.** The hardware-managed cache over global memory, shared by the whole GPU. See [post 01](posts/01-introduction-to-cuda/index.md).

**Memory wall.** The gap between how fast a processor can compute and how fast memory can feed it; a kernel that does little arithmetic per byte runs at the speed of memory, not of the cores. See [post 02](posts/02-vector-addition/index.md).

**Pinned (page-locked) memory.** Host memory the operating system will not page out, so the GPU can copy it directly; required for a `cudaMemcpyAsync` that overlaps with compute. See [post 12](posts/12-cuda-streams/index.md).

**Registers.** The fastest storage, private to one thread. The register file of an SM is shared among its resident threads, so registers per thread is one of the limits on occupancy. See [post 01](posts/01-introduction-to-cuda/index.md).

**Shared memory.** A small, fast, on-chip scratchpad shared by the threads of one block and managed by hand (`__shared__`); the tier that tiling, reductions and transposes are built on. See [post 01](posts/01-introduction-to-cuda/index.md) and [post 03](posts/03-matrix-multiplication/index.md).

**Tile, `TILE`, `s_A` and `s_B`.** A tile is a `TILE x TILE` block of a matrix that a thread block loads into shared memory once and reuses many times; `TILE` is its width, and `s_A` and `s_B` are the shared-memory copies of the tiles of `A` and `B`. See [post 03](posts/03-matrix-multiplication/index.md).

## Patterns and operations

**Atomic operation.** An indivisible read-modify-write such as `atomicAdd`: no update is lost under contention, but contended atomics on one address serialize, and atomicity says nothing about the order in which other threads see other writes. See [post 04](posts/04-reduction/index.md), [post 05](posts/05-histogram/index.md) and [post 09](posts/09-synchronization/index.md).

**Blelloch scan.** A work-efficient scan in two passes over a tree, an up-sweep that builds partial sums and a down-sweep that distributes them, doing $O(N)$ additions in $O(\log N)$ steps. See [post 08](posts/08-parallel-scan/index.md).

**Decoupled look-back.** The single-pass scan behind CUB's `DeviceScan` (Merrill and Garland, 2016): each block publishes its total and reads the running totals that earlier blocks publish, so the array is read and written once instead of twice as in a three-phase scan. See [post 08](posts/08-parallel-scan/index.md).

**Grid-stride loop.** A kernel loop in which thread `i` handles elements `i`, `i + stride`, `i + 2 * stride` and so on, where `stride = blockDim.x * gridDim.x` is the number of threads in the grid: a launch of fixed size covers any `N`, adjacent threads still read adjacent addresses on every pass, and each thread combines many elements before any cooperation with other threads. See [post 04](posts/04-reduction/index.md) and [post 05](posts/05-histogram/index.md).

**Halo (apron).** The border of input a tile needs beyond its own output elements, as wide as the filter radius; loaded into shared memory beside the tile. See [post 07](posts/07-convolution/index.md).

**Hillis-Steele scan.** The simple parallel scan: at each step, with a doubling stride, every element adds the element that many positions back, which takes $\log N$ steps but $O(N \log N)$ additions; the right choice inside a warp. See [post 08](posts/08-parallel-scan/index.md).

**Map, reduce, scan.** The three basic shapes of data-parallel work: a map is elementwise (each output depends on one input), a reduction combines $N$ inputs into one value, and a scan produces every running total. See [post 02](posts/02-vector-addition/index.md), [post 04](posts/04-reduction/index.md) and [post 08](posts/08-parallel-scan/index.md).

**Online softmax.** Computing a softmax block by block with a running maximum and a running sum, rescaling the partial result whenever the maximum grows, so the whole row never has to be held at once; the update that makes FlashAttention memory-linear. See [post 18](posts/18-flash-attention/index.md).

**Privatization.** Giving each block (or warp) its own private copy of a shared result, such as a histogram in shared memory, and merging the copies once at the end, to cut atomic contention on global memory. See [post 05](posts/05-histogram/index.md).

**Race condition.** Two or more threads accessing one location without ordering, at least one of them writing, so the result depends on timing; a plain `counter += 1` from many threads loses most of its updates. See [post 09](posts/09-synchronization/index.md).

**Separable filter.** A `K x K` filter that is the outer product of a column vector and a row vector, applied as a row pass and a column pass: $2K$ multiply-adds per pixel instead of $K^2$, at the cost of an intermediate image. See [post 07](posts/07-convolution/index.md).

**Stream compaction.** Removing elements in parallel: an exclusive scan of a keep-or-drop flag gives each kept element its output position. See [post 08](posts/08-parallel-scan/index.md).

**Tiling.** Loading a block of data into shared memory (or registers) once and reusing it many times, to cut global-memory traffic; the step that moves matrix multiplication from memory bound toward compute bound. See [post 03](posts/03-matrix-multiplication/index.md).

**Warp aggregation.** Combining the atomic updates of a warp's lanes that target one address, so that one leader lane issues one atomic for the group. See [post 05](posts/05-histogram/index.md).

**Warp shuffle (`__shfl_*_sync`).** Instructions that let the lanes of a warp read each other's registers directly, with no shared memory and no block barrier. See [post 04](posts/04-reduction/index.md).

## Synchronization

**Barrier (`__syncthreads`).** A point every thread of a block must reach before any continues; it orders the block's shared-memory writes before the reads that follow. Placed inside a branch that not every thread of the block takes, it is undefined: it can hang the block or release it with its phases paired wrongly. See [post 03](posts/03-matrix-multiplication/index.md) and [post 09](posts/09-synchronization/index.md).

**Memory fence (`__threadfence`).** An instruction that orders a thread's memory reads and writes as seen by other threads within a scope; unlike a barrier, it does not wait for any other thread, and it does not by itself make the writes visible to a reader that does not order its own reads. See [post 09](posts/09-synchronization/index.md).

**Thread scope.** The set of threads whose view of memory a fence or an atomic orders: block, device or system, nested like rings, the wider costing more. See [post 09](posts/09-synchronization/index.md).

**Warp synchronization (`__syncwarp`).** A barrier over the lanes of one warp named by a 32-bit mask that also orders their memory accesses; needed between the shared-memory steps of warp-level code because, since compute capability 7.0, the lanes of a warp are not guaranteed to run in lockstep. The `_sync` shuffles and votes synchronize their own lanes and need no `__syncwarp()` between them. See [post 09](posts/09-synchronization/index.md).

## Performance

**Arithmetic intensity ($I$).** The floating-point operations a kernel performs per byte it moves to and from DRAM, in FLOP per byte. Vector addition does about $1/12 \approx 0.083$; a kernel whose $I$ is below the card's ratio of peak FLOP/s to $\beta$ is memory bound. Met in [post 02](posts/02-vector-addition/index.md), defined in [post 10](posts/10-roofline/index.md).

**Memory bound and compute bound.** A kernel is memory bound when it waits on bytes (its arithmetic intensity is low, and only better access patterns speed it up) and compute bound when it waits on arithmetic. See [post 02](posts/02-vector-addition/index.md) and [post 10](posts/10-roofline/index.md).

**Peak bandwidth ($\beta$).** The card's theoretical DRAM bandwidth in GB/s (160 GB/s for the author's GTX 1650 with Max-Q Design, as its device query reports); achieved bandwidth is reported as a fraction of it. See [post 02](posts/02-vector-addition/index.md) and [post 10](posts/10-roofline/index.md).

**Register pressure.** Too many registers per thread: fewer threads fit on an SM, so occupancy falls, and past the hardware limit the compiler spills registers to slow local memory; the limit on how much fusion pays. See [post 14](posts/14-kernel-fusion/index.md).

**Roofline.** The ceiling on a kernel's performance as a function of its arithmetic intensity: attainable GFLOP/s is the smaller of the compute peak and $I \cdot \beta$. Plotted log-log, the bandwidth roof slopes and the compute roof is flat, and they meet at the ridge point. See [post 10](posts/10-roofline/index.md).

## Systems and tools

**Asynchronous copy (`cp.async`).** An instruction of compute capability 8.0 and above (`sm_80`) that copies from global to shared memory in the background without passing through registers, so a block can compute on one tile while the next loads. See [post 15](posts/15-async-copy-pipelining/index.md).

**Compute Sanitizer.** The CUDA toolkit's correctness tool, `compute-sanitizer`, with the tools memcheck (out-of-bounds and misaligned accesses), racecheck (shared-memory races), synccheck (illegal barrier use) and initcheck (reads of uninitialized global memory). See [post 11](posts/11-profiling-debugging/index.md).

**CUDA Graph.** A recorded sequence of kernels and copies with their dependencies, captured once from a stream and replayed with one launch, which removes most of the per-launch CPU overhead. See [post 13](posts/13-cuda-graphs/index.md).

**Double buffering.** Keeping two shared-memory buffers per operand (`s_A[2][TILE][TILE]`) so the block computes on one while the `cp.async` load of the next fills the other. See [post 15](posts/15-async-copy-pipelining/index.md).

**Kernel fusion.** Combining several operations into one kernel so the intermediate results stay in registers instead of making a round trip through global memory. See [post 14](posts/14-kernel-fusion/index.md).

**Nsight Systems and Nsight Compute.** NVIDIA's two profilers: Nsight Systems (`nsys`) shows the whole timeline of copies, kernels and API calls; Nsight Compute (`ncu`) shows one kernel's counters, such as achieved bandwidth, achieved occupancy and stall reasons. See [post 11](posts/11-profiling-debugging/index.md).

**Software pipelining.** A multi-stage schedule (a prologue, a steady state and an epilogue) that overlaps the load of tile `k + 1` with the compute on tile `k`. See [post 15](posts/15-async-copy-pipelining/index.md).

**Stream.** An ordered queue of GPU work; work in different streams may overlap, a copy on a copy engine beside a kernel on the SMs. See [post 12](posts/12-cuda-streams/index.md).

**Stream-ordered allocation.** Allocating and freeing device memory as stream operations (`cudaMallocAsync`, `cudaFreeAsync`) from a memory pool, so allocation no longer synchronizes the device. See [post 13](posts/13-cuda-graphs/index.md).

## Numerics and Tensor Cores

**BF16, FP16, FP32, TF32.** Floating-point formats by their bits of sign, exponent and mantissa: FP32 is 1, 8 and 23; FP16 is 1, 5 and 10, with a maximum of 65,504; BF16 is 1, 8 and 7, FP32's range with less precision; TF32 is a Tensor Core compute mode with FP32's 8-bit exponent and a 10-bit mantissa. See [post 16](posts/16-numerical-precision/index.md).

**Kahan (compensated) summation.** Summation that carries a second variable holding the low-order bits each addition discards and feeds them back, recovering a nearly exact sum for a few extra operations per element. See [post 16](posts/16-numerical-precision/index.md).

**Mixed precision.** Multiplying in a narrow format (FP16 or BF16) for speed and accumulating in FP32 for accuracy. See [post 16](posts/16-numerical-precision/index.md) and [post 17](posts/17-tensor-cores/index.md).

**Non-associativity.** Floating-point addition is not associative, so a parallel reduction that adds in a different order from a serial loop gives a slightly different answer; the tolerance, not bit equality, is the test. See [post 16](posts/16-numerical-precision/index.md).

**Stable softmax.** Subtracting the row maximum before exponentiating, which leaves the softmax unchanged and keeps every exponent at or below zero, so nothing overflows. See [post 16](posts/16-numerical-precision/index.md).

**Tensor Core.** A hardware unit, from compute capability 7.0, that computes a small matrix multiply-accumulate `D = A * B + C` as one warp-wide operation; through WMMA the tile is `16 x 16 x 16`, 4096 fused multiply-adds. See [post 17](posts/17-tensor-cores/index.md).

**WMMA (warp matrix multiply-accumulate).** The CUDA C++ API to Tensor Cores: a warp loads tiles into opaque fragments (`load_matrix_sync`), multiplies them (`mma_sync`) and stores the result (`store_matrix_sync`). See [post 17](posts/17-tensor-cores/index.md).

## Libraries and tools above CUDA

**cuBLAS, cuDNN, CUB.** NVIDIA's libraries for dense linear algebra, deep-learning primitives, and block- and device-wide parallel primitives (reduce, scan, sort); the baselines the kernel posts compare against. See [post 03](posts/03-matrix-multiplication/index.md), [post 08](posts/08-parallel-scan/index.md) and [post 19](posts/19-cutlass-triton/index.md).

**CUTLASS.** NVIDIA's C++ template library for matrix multiplication, which exposes the tiling hierarchy (device, thread block, warp, Tensor Core instruction) as composable templates. See [post 19](posts/19-cutlass-triton/index.md).

**Triton.** A Python language and compiler for GPU kernels written at the level of blocks rather than threads, with autotuning and just-in-time compilation; it runs on Linux. See [post 19](posts/19-cutlass-triton/index.md).

## How numbers are labeled

**The three kinds of number.** Every number in a post is derived by the post's CPU model, measured on a named card, or a representative published figure, and the post says which in the sentence that uses it or after its table. See the [notation guide](notation_guide.md).

**The three cards.** The GPUs the series measured on: the GeForce GTX 1650 (Turing, compute capability 7.5), the GeForce RTX 3080 (Ampere, 8.6) and the A100 (Ampere, 8.0). A measured number always names its card. See the [notation guide](notation_guide.md).
