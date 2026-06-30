# References

Full citations for the series, grouped by post. Each post's "Further reading"
section lists a short form; the complete reference is here.

---

## Post 01 — Introduction to CUDA

- NVIDIA. *"CUDA C++ Programming Guide."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA. *"CUDA C++ Best Practices Guide."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- Harris, M. *"An Even Easier Introduction to CUDA."* NVIDIA Developer Blog, 2017.
  <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>
- Patterson, D. & Hennessy, J. *Computer Organization and Design: The
  Hardware/Software Interface.* Morgan Kaufmann (latency vs throughput).

---

## Post 02 — Vector addition

- Harris, M. *"An Even Easier Introduction to CUDA."* NVIDIA Developer Blog, 2017.
  <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>
- Harris, M. *"How to Implement Performance Metrics in CUDA C/C++."* NVIDIA
  Developer Blog, 2012.
  <https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/>
- Harris, M. *"CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops."*
  NVIDIA Developer Blog, 2013.
  <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>
- NVIDIA. *"CUDA C++ Programming Guide."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- Williams, S., Waterman, A., & Patterson, D. *"Roofline: An Insightful Visual
  Performance Model for Multicore Architectures."* Communications of the ACM,
  52(4), 2009.

---

## Post 03 — Matrix multiplication

- NVIDIA. *"CUDA C++ Programming Guide — Shared Memory."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory>
- Boehm, S. *"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance."*
  2022. <https://siboehm.com/articles/22/CUDA-MMM>
- NVIDIA. *"CUTLASS: CUDA Templates for Linear Algebra Subroutines."*
  <https://github.com/NVIDIA/cutlass>
- NVIDIA. *"cuBLAS Library Documentation."* (current release).
  <https://docs.nvidia.com/cuda/cublas/>

---

## Post 04 — Reduction

- Harris, M. *"Optimizing Parallel Reduction in CUDA."* NVIDIA, 2007.
  <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>
- Luitjens, J. *"Faster Parallel Reductions on Kepler."* NVIDIA Developer Blog, 2014.
  <https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/>
- NVIDIA. *"CUDA C++ Programming Guide — Warp Shuffle Functions."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions>
- Lin, Y. & Grover, V. *"Using CUDA Warp-Level Primitives."* NVIDIA Developer Blog, 2018.
  <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>

---

## Post 05 — Histogram

- NVIDIA. *"CUDA C++ Programming Guide — Atomic Functions."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions>
- NVIDIA. *"CUDA C++ Best Practices Guide — Shared Memory."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory>
- Gómez-Luna, J., González-Linares, J. M., Benavides, J. I., & Guil, N.
  *"An Optimized Approach to Histogram Computation on GPU."* Machine Vision and
  Applications, 2013.
- Adinetz, A. *"CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics."*
  NVIDIA Developer Blog, 2014.
  <https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/>

---

## Post 06 — Matrix transpose

- Harris, M. *"An Efficient Matrix Transpose in CUDA C/C++."* NVIDIA Developer
  Blog, 2013.
  <https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/>
- NVIDIA. *"CUDA C++ Programming Guide — Device Memory Accesses."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses>
- NVIDIA. *"CUDA C++ Best Practices Guide — Shared Memory."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory>
- Volkov, V. *"Better Performance at Lower Occupancy."* GPU Technology Conference, 2010.

---

## Post 07 — Convolution

- NVIDIA. *"CUDA C++ Programming Guide — Constant Memory."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant-memory>
- Kirk, D. B. & Hwu, W. W. *Programming Massively Parallel Processors: A Hands-on
  Approach.* Morgan Kaufmann (ch. 7, Convolution).
- Podlozhnyuk, V. *"Image Convolution with CUDA."* NVIDIA, 2007.
- NVIDIA. *"cuDNN Developer Guide."* (current release).
  <https://docs.nvidia.com/deeplearning/cudnn/>

---

## Post 08 — Parallel scan

- Harris, M., Sengupta, S., & Owens, J. D. *"Parallel Prefix Sum (Scan) with
  CUDA."* GPU Gems 3, ch. 39, NVIDIA, 2007.
  <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
- Blelloch, G. E. *"Prefix Sums and Their Applications."* Carnegie Mellon
  University technical report CMU-CS-90-190, 1990.
- Hillis, W. D. & Steele, G. L. *"Data Parallel Algorithms."* Communications of
  the ACM, 29(12), 1986.
- NVIDIA. *"CUB: DeviceScan."* (current release).
  <https://nvidia.github.io/cccl/cub/>

---

## Post 09 — Profiling and debugging

- NVIDIA. *"Nsight Compute Documentation."* (current release).
  <https://docs.nvidia.com/nsight-compute/>
- NVIDIA. *"Nsight Systems Documentation."* (current release).
  <https://docs.nvidia.com/nsight-systems/>
- NVIDIA. *"Compute Sanitizer User Manual."* (current release).
  <https://docs.nvidia.com/compute-sanitizer/>
- Williams, S., Waterman, A., & Patterson, D. *"Roofline: An Insightful Visual
  Performance Model for Multicore Architectures."* Communications of the ACM,
  52(4), 2009.
- NVIDIA. *"CUDA C++ Best Practices Guide."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

---

## Post 10 — CUDA streams

- Harris, M. *"How to Overlap Data Transfers in CUDA C/C++."* NVIDIA Developer
  Blog, 2012.
  <https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/>
- NVIDIA. *"CUDA C++ Programming Guide — Streams."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams>
- NVIDIA. *"CUDA C++ Best Practices Guide — Asynchronous Transfers."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation>
- NVIDIA. *"Nsight Systems User Guide."* (current release).
  <https://docs.nvidia.com/nsight-systems/>

---

## Post 11 — Kernel fusion

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. *"FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness."* NeurIPS, 2022.
  <https://arxiv.org/abs/2205.14135>
- NVIDIA. *"cuDNN Developer Guide — Fused Operations."* (current release).
  <https://docs.nvidia.com/deeplearning/cudnn/>
- PyTorch. *"Introduction to torch.compile."* (current release).
  <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- Williams, S., Waterman, A., & Patterson, D. *"Roofline: An Insightful Visual
  Performance Model for Multicore Architectures."* Communications of the ACM,
  52(4), 2009.

---

## Post 12 — Asynchronous copy and software pipelining

- NVIDIA. *"CUDA C++ Programming Guide — Asynchronous Data Copies."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies>
- NVIDIA. *"NVIDIA A100 Tensor Core GPU Architecture."* Whitepaper, 2020.
  <https://www.nvidia.com/en-us/data-center/a100/>
- NVIDIA. *"CUTLASS: Efficient GEMM in CUDA."* (current release).
  <https://github.com/NVIDIA/cutlass>
- Allan, V. H., Jones, R. B., Lee, R. M., & Allan, S. J. *"Software Pipelining."*
  ACM Computing Surveys, 27(3), 1995.

---

## Post 13 — Tensor cores

- NVIDIA. *"CUDA C++ Programming Guide — Warp Matrix Functions."* (current release).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma>
- Markidis, S., Der Chien, S. W., Laure, E., Peng, I. B., & Vetter, J. S.
  *"NVIDIA Tensor Core Programmability, Performance & Precision."* IPDPSW, 2018.
  <https://arxiv.org/abs/1803.04014>
- Micikevicius, P. et al. *"Mixed Precision Training."* ICLR, 2018.
  <https://arxiv.org/abs/1710.03740>
- NVIDIA. *"Programming Tensor Cores in CUDA 9."* NVIDIA Developer Blog, 2017.
  <https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>

---

## Post 14 — FlashAttention

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. *"FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness."* NeurIPS, 2022.
  <https://arxiv.org/abs/2205.14135>
- Dao, T. *"FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning."* 2023. <https://arxiv.org/abs/2307.08691>
- Milakov, M. & Gimelshein, N. *"Online normalizer calculation for softmax."* 2018.
  <https://arxiv.org/abs/1805.02867>
- Rabe, M. N. & Staats, C. *"Self-attention Does Not Need O(n^2) Memory."* 2021.
  <https://arxiv.org/abs/2112.05682>

---

## Post 15 — CUTLASS and Triton

- Tillet, P., Kung, H. T., & Cox, D. *"Triton: An Intermediate Language and
  Compiler for Tiled Neural Network Computations."* MAPL, 2019.
  <https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf>
- NVIDIA. *"CUTLASS: CUDA Templates for Linear Algebra Subroutines."* (current release).
  <https://github.com/NVIDIA/cutlass>
- OpenAI. *"Triton Documentation and Tutorials."* (current release).
  <https://triton-lang.org/>
- Thakkar, V. et al. *"CUTLASS 3.x: CuTe and the Modern GEMM API."* NVIDIA GTC.
  <https://nvidia.github.io/cutlass/>
