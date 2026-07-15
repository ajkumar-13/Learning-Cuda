// reduction.cu — sum 16M floats three ways: naive atomic, shared-memory tree,
// and warp-shuffle. Shows why the first is catastrophic and the last is optimal.
//
// Build: nvcc -O3 -arch=sm_75 snippets/reduction.cu -o reduction   (from the post dir)
// Run:   ./reduction

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>     // malloc, rand, RAND_MAX
#include <cmath>       // fabs
#include <algorithm>   // std::min

#define BLOCK 256
#define WARP  32

#define CUDA_CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){            \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,                     \
    cudaGetErrorString(e)); return 1; } } while(0)

// --- warp-level reduction: 5 register-to-register shuffles, no shared memory ---
__device__ __forceinline__ float warpReduce(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);   // lane reads lane+off
    return v;                                         // lane 0 holds the sum
}

// --- block-level: each warp reduces, lane 0s go to shared, warp 0 finishes ---
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

// catastrophic: every thread serializes on one global address
__global__ void reduceNaiveAtomic(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(out, in[i]);
}

// shared-memory tree with SEQUENTIAL addressing (contiguous threads stay active)
__global__ void reduceShared(const float* in, float* out, int N) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];   // first half active
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// warp shuffle + grid-stride loop: registers only inside the warp
__global__ void reduceWarpShuffle(const float* in, float* out, int N) {
    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) sum += in[i];
    sum = blockReduce(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

float benchmark(void(*k)(const float*,float*,int), const float* d_in,
                float* d_out, int N, int blocks, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaMemset(d_out, 0, 4); k<<<blocks,BLOCK>>>(d_in,d_out,N); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { cudaMemset(d_out,0,4); k<<<blocks,BLOCK>>>(d_in,d_out,N); }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / iters;
}

int main() {
    const int N = 16 * 1024 * 1024;        // 16M elements = 64 MB
    size_t bytes = (size_t)N * 4;
    float* h = (float*)malloc(bytes);
    double ref = 0.0;
    for (int i = 0; i < N; ++i) { h[i] = (float)rand() / RAND_MAX; ref += h[i]; }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, 4));
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));
    int blocksAll = (N + BLOCK - 1) / BLOCK, blocksCap = std::min(blocksAll, 1024);

    float t1 = benchmark(reduceNaiveAtomic, d_in, d_out, N, blocksAll, 100);
    float t2 = benchmark(reduceShared,      d_in, d_out, N, blocksAll, 100);
    float t3 = benchmark(reduceWarpShuffle, d_in, d_out, N, blocksCap, 100);
    printf("naive atomic %8.3f ms   %6.2f GB/s\n", t1, bytes / t1 / 1e6);
    printf("shared       %8.3f ms   %6.2f GB/s\n", t2, bytes / t2 / 1e6);
    printf("warp shuffle %8.3f ms   %6.2f GB/s\n", t3, bytes / t3 / 1e6);

    // d_out still holds the last warp-shuffle result: verify it against the host sum.
    float gpuSum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpuSum, d_out, 4, cudaMemcpyDeviceToHost));
    double relErr = fabs(gpuSum - ref) / ref;
    printf("sum %.3f vs ref %.3f   (rel err %.2e)   %s\n",
           gpuSum, ref, relErr, relErr < 1e-3 ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out); free(h);
    return relErr < 1e-3 ? 0 : 1;
}
