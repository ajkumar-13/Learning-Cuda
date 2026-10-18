// async_pipeline_sm75.cu: async_pipeline.cu built for compute capability 7.5 (Turing), where cp.async does not exist.
//
// Build: nvcc -O3 -arch=sm_75 posts/15-async-copy-pipelining/snippets/async_pipeline_sm75.cu -o async_pipeline_sm75   (from the series root)
// Run:   ./async_pipeline_sm75        (on Windows: .\async_pipeline_sm75.exe)
//
// Below compute capability 8.0, cuda_pipeline.h compiles __pipeline_memcpy_async to an ordinary load into a register and
// a store to shared memory, and __pipeline_commit and __pipeline_wait_prior to nothing. So this build runs the pipelined
// kernel's logic (prologue, two buffers, the wait counts, the barriers) and checks it against post 03's synchronous kernel,
// but its copy is not asynchronous: its times measure the double-buffer structure, not cp.async. The program is
// async_pipeline.cu unchanged; this file exists so that a Turing build has its own Build line.

#include "async_pipeline.cu"
