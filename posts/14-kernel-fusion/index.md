# 14 - Kernel fusion: the fastest kernel is the one never launched

> **TL;DR.** A chain of elementwise operations written as one kernel per operation writes every intermediate array to global memory only for the next kernel to read it back, so a chain of $n$ operations moves $n$ times the bytes of one fused kernel that reads its input once, keeps the value in a register and writes once; fusing saves $(n-1)/n$ of the traffic. On the author's GTX 1650 the fused chain of 2, 3 and 4 operations over 16,777,216 floats ran 2.00, 3.00 and 3.95 times faster than the separate kernels, close to the traffic ratio while each operation stays memory bound. An elementwise operation fused into the epilogue of a compute-bound kernel costs almost nothing: a rectified linear unit (ReLU) applied inside post 03's tiled matrix multiplication added a median of 0.007 ms to its 6.8 ms, where a separate ReLU kernel added 0.113 ms. Fusion's limit is register pressure: on compute capability 7.5 occupancy falls above 64 registers per thread, and a spill sends values back through device memory.
>
> **Prerequisites:** [post 10](../10-roofline/index.md).
> **Safe to skip?** Skip it if the reader can already count the global-memory traffic of a chain of kernels, write a fused elementwise kernel, say from arithmetic intensity when a fused operation is free, and compute the occupancy a kernel's register count allows.
>
> **After reading, you will be able to:**
>
> - Explain why separate elementwise kernels waste bandwidth on intermediates, and count the traffic that fusing a chain saves.
> - Fuse a chain of elementwise operations into one kernel that reads its input once, computes in registers, and writes once.
> - Use arithmetic intensity to say when a fused operation is nearly free and when the fused kernel stops being memory bound.
> - Compute the occupancy a fused kernel's register count allows on compute capability 7.5, and say when not to fuse.

![Two rows over a green band of global memory. Top: four separate kernels, bias, scale, ReLU and GELU, each reading a whole array and writing one back; the three intermediates between them are written and read back, in red, 536.9 MB in all. Bottom: one fused kernel reads x once and writes z once, 134.2 MB.](diagrams/01-fusion-traffic.svg)

*Every intermediate a separate kernel writes is read straight back by the next one; the fused kernel keeps it in a register and moves a quarter of the bytes.*

---

## 1. The question: what does an intermediate cost?

A neural-network layer is often written as a short chain, `relu(conv(x))` or `gelu(x * W + b)`, and run as one graphics processing unit (GPU) kernel per operation. Each kernel reads its input array from global memory, which is device DRAM (dynamic random-access memory), and writes its result there. The next kernel reads that result back, applies one cheap operation and writes another array. The intermediate exists only to be written and read once.

For an elementwise chain the cost is easy to count. Each separate kernel reads the whole array and writes the whole array, 8 bytes per element for 32-bit floats, so a chain of $n$ operations moves $8n$ bytes per element. One kernel that does all $n$ operations reads 4 bytes and writes 4 bytes per element and nothing in between. The separate chain moves $n$ times as much, and fusing saves

$$\frac{8n - 8}{8n} = \frac{n-1}{n}$$

of the traffic: half for two operations, three quarters for four. The hero figure draws the four-operation chain of this post.

[Post 10](../10-roofline/index.md) gave the reason this traffic sets the time. Arithmetic intensity $I$ is the floating-point operations (FLOP) a kernel performs per byte it moves to and from DRAM; a bias add does 1 FLOP per 8 bytes, $I = 0.125$, about 156 times below the GTX 1650's ridge point of 19.5 FLOP per byte. Such a kernel is memory bound, so its time is its bytes divided by the bandwidth, and a kernel that moves fewer bytes finishes sooner. There is a smaller cost too: $n$ launches instead of one ([post 13](../13-cuda-graphs/index.md) measures a launch), which matters only when the arrays are small.

## 2. The fused kernel: read once, compute in registers, write once

The chain of this post is the one the companion model and kernel share: add a bias $b = 0.5$, multiply by a scale $s = 2$, apply a ReLU ($\max(v, 0)$), then a Gaussian error linear unit (GELU) in its common tanh approximation. Separately, thread `i` of each kernel computes one element, its global index `i = blockIdx.x * blockDim.x + threadIdx.x`, and the kernels differ only in the line that transforms it:

```cuda
__global__ void addBias(const float* in, float* out, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + b;               // 4 bytes read, 4 bytes written, 1 FLOP
}
```

Fused, thread `i` reads `x[i]` into a register, applies every operation to that register, and writes `z[i]` once. A template parameter fixes the chain's length at compile time, so the program builds the 2-, 3- and 4-operation kernels from one source:

```cuda
template <int OPS>                               // the chain's length, 2 to 4, fixed at compile time
__global__ void fusedChain(const float* x, float* z, float b, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + b;                      // op 1, bias: v is a register, never written to memory
        v = v * s;                               // op 2, scale
        if (OPS >= 3) v = fmaxf(v, 0.0f);        // op 3, ReLU
        if (OPS >= 4) v = gelu(v);               // op 4, GELU
        z[i] = v;                                // one read of x, one write of z
    }
}
```

Both are excerpts of [`snippets/fusion.cu`](snippets/fusion.cu). The order bias, then scale is deliberate: written as `x * s + b`, the fused kernel would compile to one fused multiply-add with a single rounding while the separate kernels round twice, and for most scales the two results would differ in the last bit (this post's scale of 2 is the exception: its product is exact). An add followed by a multiply is not contracted, so as written the fused output equals the separate one bit for bit for any scale, which the program checks.

The central processing unit (CPU) model, [`snippets/fusion_model.py`](snippets/fusion_model.py), runs the same chain in NumPy both ways at the kernel's size, checks the outputs, and counts the bytes:

```text
1. traffic of the elementwise chain (derived)
  tensor  = 16,777,216 floats = 67.1 MB
  chain   = + bias -> * scale -> relu -> gelu
  ops  separate (MB)  fused (MB)  separate / fused  saved by fusing  fused == separate
    2          268.4       134.2              2.0x              50%  True
    3          402.7       134.2              3.0x              67%  True
    4          536.9       134.2              4.0x              75%  True
  one separate bias, scale or ReLU: 1 FLOP per 8 bytes, I = 0.125 FLOP per byte
```

These numbers are derived by the model: the traffic ratio is the chain's length and the saving is $(n-1)/n$.

## 3. Measured: separate kernels against one fused kernel

`fusion.cu` times both versions over $N = 16{,}777{,}216$ floats (67.1 MB per array) in blocks of 256 threads. It first runs about half a second of untimed launches so that the card reaches its clock, then times each row with CUDA events as the fastest of 5 passes, a pass being the whole chain (all of its launches) for the separate version and one launch for the fused one; the rows of a chain length are timed in turn within each pass, so that a change of clock falls on both. GB/s is the bytes the version moves divided by its time. It measures the streaming multiprocessor (SM) clock before and after the timed rows, as `clock64()` cycles over CUDA-event time, compares every fused element with the separate one, and compares both with a double-precision host reference. Measured on the author's GTX 1650 (compute capability 7.5), CUDA 12.3, October 2026:

```text
device: NVIDIA GeForce GTX 1650 with Max-Q Design (compute capability 7.5), 14 SMs, theoretical peak 160 GB/s
registers per thread and theoretical occupancy for 256-thread blocks, from the runtime
  kernel                 regs  blocks/SM  occupancy
  addBias                  10          4       100%
  scale                    10          4       100%
  relu                     10          4       100%
  geluKernel               12          4       100%
  fusedChain<2>            10          4       100%
  fusedChain<3>            10          4       100%
  fusedChain<4>            10          4       100%
  matmulTiled<false>       39          4       100%
  matmulTiled<true>        39          4       100%
elementwise chain over N = 16777216 floats (67.1 MB per array); the fastest of 5 passes
  ops  version   launches  traffic (MB)  time (ms)    GB/s  speed-up  check
    2  separate         2         268.4      1.886   142.3
    2  fused            1         134.2      0.943   142.4     2.00x  ok
    3  separate         3         402.7      2.849   141.3
    3  fused            1         134.2      0.950   141.3     3.00x  ok
    4  separate         4         536.9      3.811   140.9
    4  fused            1         134.2      0.965   139.1     3.95x  ok
  traffic separate / fused = ops; fused results equal separate in 16777216, 16777216 and 16777216 of 16777216 elements;
  largest difference from a double-precision host reference: 0.0e+00, 0.0e+00, 4.3e-07
tiled matrix multiplication, TILE 16, 1024 x 1024 x 1024, and ReLU over C (4.2 MB); the fastest of 20 passes
  version                            fastest (ms)  median added to matmul alone (ms)
  matmul alone                              6.770
  ReLU kernel alone                         0.061
  matmul, then ReLU kernel                  6.829      +0.113
  matmul with ReLU in its epilogue          6.999      +0.007  ok
  fused C equals separate C in 1048576 of 1048576 elements; sampled error 3.1e-05
SM clock: 1200 MHz before the timed rows, 1440 MHz after
```

The table is measured at the SM clock the program printed, 1,200 to 1,440 MHz; under a sustained kernel the laptop card sits at its software power cap of about 35 W, and the reader's card will give other digits. Every fused result equals the separate one in all 16,777,216 elements.

Each fused kernel took 0.943, 0.950 and 0.965 ms, about the time of one read and one write of the array whatever the chain's length, while the separate chains took 1.886, 2.849 and 3.811 ms, one such pass per operation. The speed-ups, 2.00, 3.00 and 3.95 times, sit at the traffic ratios of 2, 3 and 4 the model counted, because every row moved its bytes at 139.1 to 142.4 GB/s, 87% to 89% of the card's 160 GB/s theoretical peak: the time is the bytes, so fusing saved half, two thirds and three quarters of the time with the traffic. The extra launches of the separate chains do not show at this size, where each kernel moves its arrays for about a millisecond. The one row that falls short of its ratio is the 4-operation fused kernel, at 139.1 GB/s against 141.3 for three: its GELU, with a hyperbolic tangent, is the first operation of the chain with real arithmetic, and it cost 2% of the fused kernel's time in this run; on reruns whose clock fell toward 1,000 MHz it cost more and the 4-operation speed-up came out lower, while the 2- and 3-operation ratios did not move.

![Horizontal bars of measured time for the elementwise chain over 16,777,216 floats on the GTX 1650: 2 operations, separate 1.886 ms and fused 0.943; 3, 2.849 and 0.950; 4, 3.811 and 0.965. Beside them, the median time a ReLU adds to a 6.8 ms tiled matrix multiplication: 0.113 ms as a separate kernel, 0.007 ms in its epilogue.](diagrams/02-fusion-perf.svg)

*While every kernel is memory bound, fusing an elementwise chain divides its time by its length; in an epilogue the ReLU costs nothing measurable.*

## 4. Why a memory-bound operation is nearly free inside a compute-bound kernel

A separate elementwise kernel is memory bound because it has almost no arithmetic per byte. Fused into a kernel that is already busy, the same operation adds a few instructions to each output element and no bytes at all: the value is in a register when the producer would have written it, and the operation is applied before the one write. In a library this is the epilogue of a matrix multiplication or convolution, the code that runs on each output after its last multiply-add.

The program measures it on post 03's tiled matrix multiplication (`TILE` 16, `A: [1024, 1024]`, `B: [1024, 1024]`, `C: [1024, 1024]`), once followed by a ReLU kernel over `C` and once with the ReLU applied to the register `sum` before the write, the only change to the kernel:

```cuda
    if (row < n && col < n)
        C[row * n + col] = RELU ? fmaxf(sum, 0.0f) : sum;   // the epilogue: ReLU on the register, one write
```

The multiplication alone took 6.770 ms at best and the ReLU kernel alone 0.061 ms, its 8.4 MB at 138 GB/s. Under a sustained multiplication the card's clock moves at its power cap by more than these rows differ, so the program also takes each ReLU's cost pass by pass, as the difference from the multiplication alone in the same pass, and reports the median of the 20 passes: the separate ReLU kernel added 0.113 ms, 1.7% of the multiplication, more than its own fastest time alone, by an amount that changed from run to run (the run does not separate the kernel from the gap between the two launches); the ReLU in the epilogue added 0.007 ms, 0.1%, within the noise of the measurement. The fused `C` equals the separate one in every element.

The two experiments show one rule. An operation with almost no arithmetic per byte is priced by its bytes: run alone it costs a full pass over memory, and fused it costs nearly nothing, whether into a chain or into a compute-bound producer. An operation with more arithmetic, like the GELU with its hyperbolic tangent, brings its instructions into the fused kernel; here they hid almost entirely behind the memory traffic, but the more arithmetic a chain fuses per element, the closer the fused kernel comes to being limited by it. The roofline says which side of that line a fused kernel sits on: its intensity is the FLOP of the whole chain over the 8 bytes it moves per element, and it stays memory bound, its operations nearly free, until that intensity reaches the ridge of 19.5 FLOP per byte, 156 FLOP per element.

## 5. The limit: register pressure

Fusion keeps intermediates in registers, and registers are the scarcest storage on the chip. A streaming multiprocessor of the GTX 1650 has 65,536 32-bit registers for at most 1,024 resident threads, 32 warps. [Post 10](../10-roofline/index.md) derived the rule the hardware follows on compute capability 7.5: registers are allocated per warp in units of 256, from four quarters of 16,384 that each hold whole warps, and a block is resident only whole. The compiler reports each kernel's count with `-Xptxas=-v`; for this post's kernels it printed, among the other kernels' lines:

```text
ptxas info    : Compiling entry function '_Z10fusedChainILi4EEvPKfPfffi' for 'sm_75'
ptxas info    : Function properties for _Z10fusedChainILi4EEvPKfPfffi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z10geluKernelPKfPfi' for 'sm_75'
ptxas info    : Function properties for _Z10geluKernelPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 372 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11matmulTiledILb1EEvPKfS1_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z11matmulTiledILb1EEvPKfS1_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 39 registers, 2048 bytes smem, 380 bytes cmem[0]
```

The fused chain needs 10 registers per thread, as few as one separate bias kernel, and the GELU kernel alone 12: fusing four elementwise operations did not raise the count, because each operation overwrites the same register `v`. The tiled matrix multiplication needs 39 and its ReLU epilogue none more. The program's runtime query agrees: every kernel in it reaches 4 blocks of 256 threads per SM, 100% theoretical occupancy. The model applies post 10's rule to heavier register counts:

```text
2. register pressure on sm_75, 256-thread blocks (derived, post 10's rule)
  regs/thread  regs/warp  blocks/SM  warps/SM  occupancy
           10        512          4        32       100%
           12        512          4        32       100%
           39      1,280          4        32       100%
           64      2,048          4        32       100%
           65      2,304          3        24        75%
           96      3,072          2        16        50%
          128      4,096          2        16        50%
          255      8,192          1         8        25%
  100% up to 64 registers per thread (65,536 / 1,024 threads)
```

These are derived. On compute capability 7.5 a kernel keeps every warp slot up to 64 registers per thread, $65{,}536 / 1{,}024$; at 65 a warp needs 2,304 registers, a quarter holds 7 warps, and only 3 blocks of 8 warps fit, 75%. From 81 to 128 registers two blocks fit, 50%, and from 129 one, 25%. An earlier draft of this post put the cliff at 32 registers, counting 2,048 resident threads per SM, the limit of compute capability 8.0 (the A100); Turing's limit is 1,024 threads, which moves it to 64.

![A step curve of theoretical occupancy against registers per thread for 256-thread blocks on compute capability 7.5: 100% up to 64, 75% from 65 to 80, 50% from 81 to 128, 25% from 129. The fused chain at 10 registers and the tiled matrix multiplication at 39 sit at 100%.](diagrams/03-register-pressure.svg)

*A fused kernel pays for its registers only past 64 per thread; then whole blocks leave the SM.*

Lower occupancy is not by itself a loss: post 10 showed that a memory-bound kernel needs only enough bytes in flight to cover the latency, and independent loads per thread can supply them at lower occupancy. The real hazard past the hardware limit of 255 registers per thread, or a cap set by `-maxrregcount` or `__launch_bounds__`, is a spill. The compiler then stores registers to local memory, which, in the CUDA C++ Programming Guide's words, "resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses": a spilled fused kernel moves through DRAM the very values fusion kept out of it. `-Xptxas=-v` prints `spill stores` and `spill loads`, 0 bytes for every kernel here, and a fused kernel whose count passes 64, or which spills at all, is measured against the unfused version before it is kept.

## 6. Fusion in practice

Few programs fuse by hand. Three places where it happens without a hand-written kernel:

- **Compilers.** `torch.compile`, in PyTorch's words, "makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels"; its default compiler, TorchInductor, "for NVIDIA, AMD and Intel GPUs ... leverages OpenAI Triton as the key building block", and PyTorch's getting-started guide calls fusion "the single most important optimization that inductor performs": a chain of pointwise operations becomes one generated kernel with one read and one write.
- **Libraries.** cuDNN fuses an epilogue into a convolution: its `cudnnConvolutionBiasActivationForward` computes `y = act(alpha1 * conv(x) + alpha2 * z + bias)` in one call (deprecated in cuDNN 9.0; the graph application programming interface (API) of cuDNN builds such fused chains at run time with its fusion engines).
- **Algorithms.** FlashAttention (Dao and others, 2022) is fusion carried through a whole attention layer: it computes $\mathbf{Q}\mathbf{K}^\top$, the softmax and the product with $\mathbf{V}$ block by block in on-chip memory, so the $S \times S$ score matrix for a sequence of length $S$ is never written to global memory. Its extra memory is linear in $S$, and its Theorem 2 counts $\Theta(S^2 d^2 / M)$ global-memory accesses against $\Theta(S d + S^2)$ for standard attention, with head dimension $d$ and on-chip memory size $M$. The paper's Figure 2 reports, for the forward and backward pass of GPT-2 medium's attention at sequence length 1,024 (head dimension 64, 16 heads, batch 64) on an A100, 40.3 GB of memory reads and writes and 41.7 ms for standard attention against 4.4 GB and 7.3 ms for FlashAttention, which does more FLOP (75.2 against 66.6 billion) and still runs 5.7 times faster. These are published figures, not measured here; [post 18](../18-flash-attention/index.md) builds the algorithm.

## 7. Make it run

The model needs NumPy, which the series environment carries. From the series root:

```bash
uv run python posts/14-kernel-fusion/snippets/fusion_model.py
```

It prints both parts quoted above in about 2 seconds. The kernel needs an NVIDIA GPU and the CUDA Toolkit, and builds with `-arch` set to the card's compute capability:

```bash
nvcc -O3 -arch=sm_75 -Xptxas=-v posts/14-kernel-fusion/snippets/fusion.cu -o fusion
./fusion
```

On Windows the program is `fusion.exe` and `nvcc` may need `-ccbin`; [RUNNING.md](../../RUNNING.md) covers both. It builds in about 10 seconds, runs in about 5, and exits with status 1 if any check fails. The `-Xptxas=-v` flag only adds the register report. If the printed SM clock is far below the card's boost clock, the card was not at full speed and the times are not comparable; rerun when it is idle.

## 8. What can go wrong?

- **Timing a card that is not at its clock.** A run of an earlier build of `fusion.cu` while the laptop's power source kept changing printed an SM clock of 211 MHz before the timed rows and 246 MHz after; its times were several times slower and its fused GELU chain looked compute bound. The printed clock is how such a run is caught and discarded. A dip in the middle of a run escapes the two probes: in another run both read about 1,500 MHz while the 4-operation rows fell to 70.7 and 57.0 GB/s, so the GB/s column is read row by row before a run is kept.
- **Expecting bit-identical results after reordering.** Fusing `x * s + b` lets the compiler contract the multiply and add into one fused multiply-add, so the fused result can differ from the separate one in the last bit; a check that demands equality fails, and one with a tolerance passes. Either keep the order that does not contract or compare with a tolerance that says why.
- **Counting the separate chain's bytes from its input and output only.** The separate chain's cost is in the intermediates: three of the four reads and writes of the four-operation chain are arrays no one asked for.
- **Fusing across a dependency on the whole array.** An operation that needs a result over every element, such as subtracting the mean of the array, cannot run in the same kernel as the reduction that produces it without a grid-wide barrier, which needs a cooperative launch with every block resident ([post 09](../09-synchronization/index.md)); the usual answer is two kernels, or a reduction fused with its producer and a second pass.

---

## Common pitfalls

1. **Fusing a heavy chain without reading the register report.** Past 64 registers per thread a 256-thread block loses warp slots on compute capability 7.5, and a spill sends values back through DRAM; check `-Xptxas=-v` and measure against the unfused version.
2. **Expecting every fused operation to be free.** Bias, scale and ReLU cost nothing fused; an operation with real arithmetic, like the GELU's tanh, adds its instructions, about 2% of the fused kernel's time here and more as a chain's FLOP per byte approach the ridge.
3. **Writing a fused kernel before checking the library.** `torch.compile`, cuDNN's fused operations and CUTLASS epilogues already fuse the common patterns; a hand-fused kernel pays when profiling shows an intermediate the library still writes.
4. **Judging a fused kernel by GB/s alone.** It moves fewer bytes, so its GB/s can fall while its time improves; compare times, and GB/s against the bytes each version really moves.
5. **Comparing the occupancy cliff across cards.** The 64-register threshold is $65{,}536 / 1{,}024$ on compute capability 7.5; a card with 2,048 threads per SM reaches it at 32.

---

## Further reading

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. and Ré, C., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS, 2022). Fusion through a whole attention layer, Theorem 2's count of memory accesses, and Figure 2's measurements.
- NVIDIA, *cuDNN Documentation*, `cudnnConvolutionBiasActivationForward` and the graph API's fusion engines, consulted October 2026.
- PyTorch, "Introduction to torch.compile", and the `torch.compiler` documentation with its "Getting Started" page (TorchInductor and fusion), consulted October 2026.
- NVIDIA, *CUDA C++ Programming Guide (Legacy)*, section "Local Memory", consulted October 2026 (CUDA 12.3 edition). Where spilled registers live.
- Williams, S., Waterman, A. and Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (Communications of the ACM 52(4), 2009). Why an elementwise kernel is priced by its bytes.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 15 - Asynchronous copy and software pipelining: cp.async, double buffering, and hiding memory latency](../15-async-copy-pipelining/index.md):** hides the latency of the loads that fusion cannot remove.
- **[Post 18 - FlashAttention: making attention memory-linear with tiling and online softmax](../18-flash-attention/index.md):** fusion carried through a whole attention layer, so the score matrix never reaches global memory.
