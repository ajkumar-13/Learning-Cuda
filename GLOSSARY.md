# Glossary

The terms used across the series, defined once. Each links to the post that introduces it.

## Execution model

- **Thread** — one instance of a kernel, running over one element. The smallest unit of execution. ([01](posts/01-introduction-to-cuda/index.md))
- **Block (thread block)** — a group of up to 1024 threads that run on one SM, share fast on-chip memory, and can synchronize with `__syncthreads()`. ([01](posts/01-introduction-to-cuda/index.md))
- **Grid** — all the blocks launched by one kernel call; blocks run independently and in any order. ([01](posts/01-introduction-to-cuda/index.md))
- **Kernel** — a `__global__` function that runs on the GPU, written from the point of view of one thread, launched with `kernel<<<grid, block>>>(...)`. ([01](posts/01-introduction-to-cuda/index.md))
- **Global index** — a thread's flat position, `blockIdx.x * blockDim.x + threadIdx.x` (and its 2D form). ([01](posts/01-introduction-to-cuda/index.md), [02](posts/02-vector-addition/index.md))
- **Streaming multiprocessor (SM)** — the hardware unit that runs blocks; a GPU has many. ([01](posts/01-introduction-to-cuda/index.md))
- **Warp** — a group of 32 threads that execute the same instruction in lockstep (SIMT). ([01](posts/01-introduction-to-cuda/index.md))
- **Warp divergence** — when threads in a warp take different branches, so the hardware runs both paths in series. ([01](posts/01-introduction-to-cuda/index.md))
- **Occupancy** — the ratio of active warps to the maximum an SM supports; higher occupancy hides memory latency better. ([11](posts/11-kernel-fusion/index.md))

## Memory

- **Host / device** — the CPU (and system RAM) versus the GPU (and its global memory); separate address spaces. ([01](posts/01-introduction-to-cuda/index.md), [02](posts/02-vector-addition/index.md))
- **Global memory (HBM/DRAM)** — large, off-chip, slow GPU memory; the only tier visible to every thread. ([01](posts/01-introduction-to-cuda/index.md))
- **Shared memory (SRAM)** — small, fast, per-block on-chip scratchpad you manage by hand. ([01](posts/01-introduction-to-cuda/index.md), [03](posts/03-matrix-multiplication/index.md))
- **Registers** — the fastest, per-thread storage. ([01](posts/01-introduction-to-cuda/index.md))
- **Constant memory** — a 64 KB read-only space whose cache broadcasts one value to a whole warp. ([07](posts/07-convolution/index.md))
- **Coalescing** — when the 32 threads of a warp read consecutive addresses, served as one 128-byte transaction; strided access costs up to 32. ([06](posts/06-matrix-transpose/index.md))
- **Bank conflict** — when multiple lanes hit the same shared-memory bank and serialize; fixed by padding. ([06](posts/06-matrix-transpose/index.md))
- **Pinned (page-locked) memory** — host memory the GPU can DMA directly, required for truly async transfers. ([10](posts/10-cuda-streams/index.md))
- **Arithmetic intensity** — FLOPs per byte moved; decides whether a kernel is memory-bound or compute-bound. ([02](posts/02-vector-addition/index.md), [11](posts/11-kernel-fusion/index.md))

## Patterns and operations

- **Map / reduce / scan** — the three pillars: element-wise (N→N independent), aggregate (N→1), running total (N→N dependent). ([02](posts/02-vector-addition/index.md), [04](posts/04-reduction/index.md), [08](posts/08-parallel-scan/index.md))
- **Tiling** — loading a block of data into shared memory once and reusing it many times. ([03](posts/03-matrix-multiplication/index.md))
- **Warp shuffle (`__shfl_*_sync`)** — instructions that let lanes read each other's registers directly, no shared memory. ([04](posts/04-reduction/index.md))
- **Atomic operation** — an indivisible read-modify-write (`atomicAdd`); correct under contention but serializing. ([04](posts/04-reduction/index.md), [05](posts/05-histogram/index.md))
- **Privatization** — giving each block a private copy (e.g. a shared histogram) and merging once, to cut contention. ([05](posts/05-histogram/index.md))
- **Halo (apron)** — the border of input a tile needs beyond its output, equal to the filter radius. ([07](posts/07-convolution/index.md))
- **Online softmax** — computing a softmax block by block with a running max and sum, rescaled as the max grows. ([14](posts/14-flash-attention/index.md))
- **Stream compaction** — removing elements in parallel using an exclusive scan of a keep/drop predicate as output positions. ([08](posts/08-parallel-scan/index.md))

## Systems and hardware

- **Profiler (Nsight Systems / Nsight Compute)** — `nsys` shows the system timeline (copy/compute overlap); `ncu` shows per-kernel counters: achieved bandwidth, occupancy, and stall reasons. ([09](posts/09-profiling-debugging/index.md))
- **Compute Sanitizer** — the CUDA correctness suite: `memcheck` (out-of-bounds), `racecheck` (shared-memory races), and `synccheck`. ([09](posts/09-profiling-debugging/index.md))
- **Stream** — an ordered queue of GPU operations; operations in different streams can overlap on separate engines. ([10](posts/10-cuda-streams/index.md))
- **Kernel fusion** — combining several operations into one kernel so intermediates stay in registers, not memory. ([11](posts/11-kernel-fusion/index.md))
- **Asynchronous copy (`cp.async`)** — an Ampere (`sm_80`+) instruction that copies global to shared memory in the background, bypassing the register file. ([12](posts/12-async-copy-pipelining/index.md))
- **Double buffering** — keeping two shared-memory tiles so compute on one overlaps the `cp.async` load of the next. ([12](posts/12-async-copy-pipelining/index.md))
- **Software pipelining** — a multi-stage schedule (prologue, steady state, epilogue) that overlaps the load of tile `k+1` with the compute on tile `k`. ([12](posts/12-async-copy-pipelining/index.md))
- **Tensor Core** — hardware that computes a whole 16×16×16 matrix multiply-accumulate per warp instruction. ([13](posts/13-tensor-cores/index.md))
- **Mixed precision** — multiplying in FP16 (fast) but accumulating in FP32 (accurate). ([13](posts/13-tensor-cores/index.md))
- **WMMA** — the Warp Matrix Multiply-Accumulate API: fragments, `load_matrix_sync`, `mma_sync`, `store_matrix_sync`. ([13](posts/13-tensor-cores/index.md))
- **CUTLASS** — NVIDIA's C++ template library exposing the device→threadblock→warp→MMA tiling hierarchy. ([15](posts/15-cutlass-triton/index.md))
- **Triton** — a Python DSL for writing GPU kernels at the block level, with autotuning and JIT compilation. ([15](posts/15-cutlass-triton/index.md))
