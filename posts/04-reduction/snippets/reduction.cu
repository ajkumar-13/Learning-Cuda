// reduction.cu: sum 16M floats five ways and time each one. The rows apply one fix at a time:
// one atomicAdd per element; a shared-memory tree with interleaved addressing; the same tree with
// sequential addressing; that tree fed by a grid-stride loop; and a warp-shuffle block reduction
// fed by the same loop. Every tree kernel ends with one atomicAdd per block.
//
// Build: nvcc -O3 -arch=sm_75 posts/04-reduction/snippets/reduction.cu -o reduction   (from the series root)
// Run:   ./reduction        (on Windows: .\reduction.exe)
//
// Prints the card, then one row per method: time, achieved bandwidth over the 64 MiB input in
// decimal GB/s, the share of the card's theoretical peak, and a check of the sum against a
// double-precision host sum. Each GPU time is the median of ITERS timed launches after one
// untimed warm-up; the output is zeroed before each launch, outside the timed region. The CPU row
// times the host reference loop (one thread, double accumulator), best of 5.

#include <cuda_runtime.h>
#include <algorithm>   // std::sort
#include <chrono>
#include <cmath>       // fabs
#include <cstdio>
#include <cstdlib>     // malloc, rand, RAND_MAX

#define BLOCK 256
#define WARP  32
#define ITERS 21

#define CUDA_CHECK(c) do { cudaError_t e = (c); if (e != cudaSuccess) {                  \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
    exit(1); } } while (0)

// one atomicAdd per element: every thread queues on one global address
__global__ void reduceNaiveAtomic(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(out, in[i]);
}

// shared-memory tree with INTERLEAVED addressing: the active threads are scattered
__global__ void reduceInterleaved(const float* in, float* out, int N) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];   // threads 0, 2, 4, ... first
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// shared-memory tree with SEQUENTIAL addressing: the active threads are contiguous
__global__ void reduceShared(const float* in, float* out, int N) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];   // the first half stays active
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// the same tree, but each thread first sums many elements in a register (grid-stride loop)
__global__ void reduceSharedStride(const float* in, float* out, int N) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        sum += in[i];
    s[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// warp level: five shuffle-and-add steps, no shared memory
__device__ __forceinline__ float warpReduce(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);   // lane reads lane + off
    return v;                                         // lane 0 has the sum
}

// block level: each warp reduces, lane 0 of each warp writes its total, warp 0 finishes
__device__ __forceinline__ float blockReduce(float v) {
    __shared__ float warpSums[BLOCK / WARP];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warpReduce(v);
    if (lane == 0) warpSums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < BLOCK / WARP) ? warpSums[lane] : 0.0f;
        v = warpReduce(v);
    }
    return v;
}

// grid level: a grid-stride loop, then one atomicAdd per block
__global__ void reduceWarpShuffle(const float* in, float* out, int N) {
    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) sum += in[i];   // grid-stride: any N, coalesced
    sum = blockReduce(sum);                    // warp shuffle + shared + warp 0
    if (threadIdx.x == 0) atomicAdd(out, sum); // one atomic per block
}

typedef void (*Kernel)(const float*, float*, int);

// median time of ITERS launches in ms; *sum receives the result of the last launch
float timeKernel(Kernel k, int blocks, const float* d_in, float* d_out, int N, float* sum) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    k<<<blocks, BLOCK>>>(d_in, d_out, N);                  // untimed warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms[ITERS];
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));   // zero outside the timed region
        CUDA_CHECK(cudaEventRecord(start));
        k<<<blocks, BLOCK>>>(d_in, d_out, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms[it], start, stop));
    }
    CUDA_CHECK(cudaMemcpy(sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(ms, ms + ITERS);
    return ms[ITERS / 2];
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    // theoretical peak DRAM bandwidth (GB/s) = 2 transfers per clock * bus width in bytes * memory clock
    double peak = 2.0 * (p.memoryBusWidth / 8.0) * (p.memoryClockRate * 1e3) / 1e9;
    printf("%s, compute capability %d.%d, theoretical peak %.0f GB/s\n", p.name, p.major, p.minor, peak);

    const int N = 16 * 1024 * 1024;        // 16,777,216 floats = 64 MiB
    size_t bytes = (size_t)N * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h[i] = (float)rand() / RAND_MAX;

    // the host reference, which is also the CPU row: one thread, double accumulator, best of 5
    double ref = 0.0, cpuMs = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += h[i];
        auto t1 = std::chrono::steady_clock::now();
        cpuMs = std::min(cpuMs, std::chrono::duration<double, std::milli>(t1 - t0).count());
        ref = s;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    const int blocksAll = (N + BLOCK - 1) / BLOCK;   // one thread per element: 65,536 blocks
    const int blocksGrid = 1024;                     // grid-stride: 64 elements per thread
    struct { const char* name; Kernel k; int blocks; } runs[] = {
        {"naive atomic", reduceNaiveAtomic, blocksAll},
        {"interleaved", reduceInterleaved, blocksAll},
        {"sequential", reduceShared, blocksAll},
        {"seq. + stride", reduceSharedStride, blocksGrid},
        {"warp shuffle", reduceWarpShuffle, blocksGrid},
    };

    printf("N = %d floats (%zu bytes), %d threads per block, median of %d launches\n", N, bytes, BLOCK, ITERS);
    printf("%-14s %6s %10s %7s %7s   %s\n", "method", "blocks", "time (ms)", "GB/s", "% peak", "sum check");
    printf("%-14s %6s %10.3f %7.1f %6.1f%%   reference %.1f\n", "CPU, 1 thread", "-", cpuMs,
           bytes / cpuMs / 1e6, 100.0 * bytes / cpuMs / 1e6 / peak, ref);
    bool ok = true;
    for (auto& r : runs) {
        float sum = 0.0f;
        float ms = timeKernel(r.k, r.blocks, d_in, d_out, N, &sum);
        double relErr = fabs(sum - ref) / ref;
        bool pass = relErr < 1e-4;
        ok = ok && pass;
        printf("%-14s %6d %10.3f %7.1f %6.1f%%   %s, rel err %.0e\n", r.name, r.blocks, ms,
               bytes / ms / 1e6, 100.0 * bytes / ms / 1e6 / peak, pass ? "PASS" : "FAIL", relErr);
    }

    cudaFree(d_in); cudaFree(d_out); free(h);
    return ok ? 0 : 1;
}
