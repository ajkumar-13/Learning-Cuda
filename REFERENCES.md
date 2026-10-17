# References

Full citations for the series, grouped by post. Each post's "Further reading" section lists a short form; the complete reference is here. Each entry gives the author, the title (a book or document in italics, an article or chapter in quotation marks), the venue and the year, then the address. NVIDIA's documentation changes with every toolkit release, so each NVIDIA guide is cited as consulted in October 2026, when the online guides were those of CUDA Toolkit 13.4; the series' kernels are built and run with CUDA 12.3, and the sections cited here describe behavior that holds across CUDA 12.x.

---

## Post 01 - Why GPUs exist: the CUDA execution model from the ground up

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*. CUDA Toolkit documentation, page last updated 12 October 2026, consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA. *CUDA C++ Best Practices Guide*. CUDA Toolkit 13.4 documentation, page last updated 12 October 2026, consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- Harris, M. "An Even Easier Introduction to CUDA." NVIDIA Technical Blog, 25 January 2017; updated edition 2 May 2025. <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>
- Patterson, D. A. and Hennessy, J. L. *Computer Organization and Design: The Hardware/Software Interface*. 5th ed. Morgan Kaufmann, 2013. Latency against throughput, and the CPU the GPU is contrasted with.

---

## Post 02 - Your first CUDA kernel: vector addition and the memory wall

- Harris, M. "An Even Easier Introduction to CUDA." NVIDIA Technical Blog, 25 January 2017; updated edition 2 May 2025. <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>
- Harris, M. "How to Implement Performance Metrics in CUDA C/C++." NVIDIA Technical Blog, 2012. <https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/>
- Harris, M. "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops." NVIDIA Technical Blog, 2013. <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*. Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- Williams, S., Waterman, A. and Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>

---

## Post 03 - Matrix multiplication: from a memory-bound dot product to shared-memory tiling

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Shared Memory". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory>
- Boehm, S. "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog." 2022. <https://siboehm.com/articles/22/CUDA-MMM>
- NVIDIA. *CUTLASS: CUDA Templates for Linear Algebra Subroutines*. Source repository, consulted October 2026. <https://github.com/NVIDIA/cutlass>
- NVIDIA. *cuBLAS Library Documentation*. Consulted October 2026. <https://docs.nvidia.com/cuda/cublas/>

---

## Post 04 - Reduction: summing millions of numbers without a serial bottleneck

- Harris, M. "Optimizing Parallel Reduction in CUDA." NVIDIA, 2007. <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>
- Luitjens, J. "Faster Parallel Reductions on Kepler." NVIDIA Technical Blog, 2014. <https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Warp Shuffle Functions". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions>
- Lin, Y. and Grover, V. "Using CUDA Warp-Level Primitives." NVIDIA Technical Blog, 2018. <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>

---

## Post 05 - Histogram: beating atomic contention with privatization

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Atomic Functions". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions>
- NVIDIA. *CUDA C++ Best Practices Guide*, section "Shared Memory". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory>
- Gómez-Luna, J., González-Linares, J. M., Benavides, J. I. and Guil, N. "An Optimized Approach to Histogram Computation on GPU." *Machine Vision and Applications*, 2013.
- Adinetz, A. "CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics." NVIDIA Technical Blog, 2014. <https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/>

---

## Post 06 - Matrix transpose: coalescing, shared memory, and bank conflicts

- Harris, M. "An Efficient Matrix Transpose in CUDA C/C++." NVIDIA Technical Blog, 2013. <https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Device Memory Accesses". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses>
- NVIDIA. *CUDA C++ Best Practices Guide*, section "Shared Memory". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory>
- Volkov, V. "Better Performance at Lower Occupancy." GPU Technology Conference, 2010.

---

## Post 07 - Convolution: constant memory, shared-memory halos, and separable filters

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Constant Memory". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant-memory>
- Kirk, D. B. and Hwu, W. W. *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann. The chapter on convolution.
- Podlozhnyuk, V. "Image Convolution with CUDA." NVIDIA, 2007.
- NVIDIA. *cuDNN Documentation*. Consulted October 2026. <https://docs.nvidia.com/deeplearning/cudnn/>

---

## Post 08 - Parallel scan: breaking the dependency chain of a prefix sum

- Harris, M., Sengupta, S. and Owens, J. D. "Parallel Prefix Sum (Scan) with CUDA." In *GPU Gems 3*, chapter 39. NVIDIA, 2007. <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
- Blelloch, G. E. "Prefix Sums and Their Applications." Carnegie Mellon University technical report CMU-CS-90-190, 1990.
- Hillis, W. D. and Steele, G. L. "Data Parallel Algorithms." *Communications of the ACM* 29(12), 1986.
- NVIDIA. *CUB* documentation, `cub::DeviceScan`. Consulted October 2026. <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html>

---

## Post 09 - When threads share data: barriers, atomics, and the memory model

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, sections "Memory Fence Functions" and "Atomic Functions". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Independent Thread Scheduling". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#independent-thread-scheduling>
- Lin, Y. and Grover, V. "Using CUDA Warp-Level Primitives." NVIDIA Technical Blog, 2018. <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>
- Sorensen, T., Donaldson, A. F., Batty, M., Gopalakrishnan, G. and Rakamarić, Z. "Portable Inter-Workgroup Barrier Synchronisation for GPUs." OOPSLA, 2016.
- Boehm, H.-J. and Adve, S. V. "Foundations of the C++ Concurrency Memory Model." PLDI, 2008. <https://doi.org/10.1145/1375581.1375591>

---

## Post 10 - How fast can a kernel be: roofline, arithmetic intensity, and occupancy

- Williams, S., Waterman, A. and Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>
- NVIDIA. *CUDA C++ Best Practices Guide*. Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Occupancy Calculator". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#occupancy-calculator>
- Volkov, V. "Better Performance at Lower Occupancy." GPU Technology Conference, 2010.

---

## Post 11 - Profiling and debugging: Nsight, compute-sanitizer, and measured occupancy

- NVIDIA. *Nsight Compute Documentation*. Consulted October 2026. <https://docs.nvidia.com/nsight-compute/>
- NVIDIA. *Nsight Systems Documentation*. Consulted October 2026. <https://docs.nvidia.com/nsight-systems/>
- NVIDIA. *Compute Sanitizer Documentation*. Consulted October 2026. <https://docs.nvidia.com/compute-sanitizer/>
- Williams, S., Waterman, A. and Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>
- NVIDIA. *CUDA C++ Best Practices Guide*. Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

---

## Post 12 - CUDA streams: overlapping copy and compute for free throughput

- Harris, M. "How to Overlap Data Transfers in CUDA C/C++." NVIDIA Technical Blog, 2012. <https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Streams". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams>
- NVIDIA. *CUDA C++ Best Practices Guide*, section "Asynchronous and Overlapping Transfers with Computation". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation>
- NVIDIA. *Nsight Systems Documentation*. Consulted October 2026. <https://docs.nvidia.com/nsight-systems/>

---

## Post 13 - Cutting launch overhead: CUDA Graphs and stream-ordered allocation

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "CUDA Graphs". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs>
- NVIDIA. "Getting Started with CUDA Graphs." NVIDIA Technical Blog, 2019. <https://developer.nvidia.com/blog/cuda-graphs/>
- NVIDIA. "Using the NVIDIA CUDA Stream-Ordered Memory Allocator, Part 1." NVIDIA Technical Blog, 2021. <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/>
- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Stream Ordered Memory Allocator". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator>

---

## Post 14 - Kernel fusion: the fastest kernel is the one never launched

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. and Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022. <https://arxiv.org/abs/2205.14135>
- NVIDIA. *cuDNN Documentation*, fused operations. Consulted October 2026. <https://docs.nvidia.com/deeplearning/cudnn/>
- PyTorch. "Introduction to torch.compile." PyTorch tutorials, consulted October 2026. <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- Williams, S., Waterman, A. and Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>

---

## Post 15 - Asynchronous copy and software pipelining: cp.async, double buffering, and hiding memory latency

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Asynchronous Data Copies". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies>
- NVIDIA. *NVIDIA A100 Tensor Core GPU Architecture*. Whitepaper, 2020. <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>
- NVIDIA. *CUTLASS: CUDA Templates for Linear Algebra Subroutines*. Source repository, consulted October 2026. <https://github.com/NVIDIA/cutlass>
- Allan, V. H., Jones, R. B., Lee, R. M. and Allan, S. J. "Software Pipelining." *ACM Computing Surveys* 27(3), 1995.

---

## Post 16 - Numbers on a GPU: FP32, FP16, BF16, and numerical stability

- Goldberg, D. "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys* 23(1), 1991. <https://doi.org/10.1145/103162.103163>
- IEEE. *IEEE Standard for Floating-Point Arithmetic* (IEEE 754-2019). <https://standards.ieee.org/ieee/754/6210/>
- Micikevicius, P. et al. "Mixed Precision Training." ICLR, 2018. <https://arxiv.org/abs/1710.03740>
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*. 2nd ed. SIAM, 2002. Compensated and pairwise summation.
- NVIDIA. "Train With Mixed Precision." NVIDIA Deep Learning Performance documentation, consulted October 2026. <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/>

---

## Post 17 - Tensor cores: matrix-at-once hardware and mixed precision

- NVIDIA. *CUDA C++ Programming Guide (Legacy)*, section "Warp Matrix Functions". Consulted October 2026. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma>
- Markidis, S., Der Chien, S. W., Laure, E., Peng, I. B. and Vetter, J. S. "NVIDIA Tensor Core Programmability, Performance & Precision." IPDPSW, 2018. <https://arxiv.org/abs/1803.04014>
- Micikevicius, P. et al. "Mixed Precision Training." ICLR, 2018. <https://arxiv.org/abs/1710.03740>
- NVIDIA. "Programming Tensor Cores in CUDA 9." NVIDIA Technical Blog, 2017. <https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>

---

## Post 18 - FlashAttention: making attention memory-linear with tiling and online softmax

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. and Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022. <https://arxiv.org/abs/2205.14135>
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023. <https://arxiv.org/abs/2307.08691>
- Milakov, M. and Gimelshein, N. "Online normalizer calculation for softmax." 2018. <https://arxiv.org/abs/1805.02867>
- Rabe, M. N. and Staats, C. "Self-attention Does Not Need $O(n^2)$ Memory." 2021. <https://arxiv.org/abs/2112.05682>

---

## Post 19 - CUTLASS and Triton: why nobody writes raw CUDA for matmul anymore

- Tillet, P., Kung, H. T. and Cox, D. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL, 2019. <https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf>
- NVIDIA. *CUTLASS: CUDA Templates for Linear Algebra Subroutines*. Source repository, consulted October 2026. <https://github.com/NVIDIA/cutlass>
- OpenAI. *Triton Documentation and Tutorials*. Consulted October 2026. <https://triton-lang.org/>
- Thakkar, V. et al. "CUTLASS 3.x: CuTe and the Modern GEMM API." NVIDIA GTC. Documentation consulted October 2026. <https://nvidia.github.io/cutlass/>
