// histogram.cu — count value frequencies two ways: global atomics (collapses
// under contention) and per-block shared-memory privatization (robust).
//
// Build: nvcc -O3 -arch=sm_75 snippets/histogram.cu -o histogram   (from the repo root; drop snippets/ if you cd into it)
// Run:   ./histogram

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>      // malloc, free
#include <algorithm>    // std::min

#define BINS 256

// naive: every thread atomically increments a GLOBAL bin. On "boring" data
// (e.g. a white wall, all one value) every thread hits one address and serializes.
__global__ void histGlobal(const int* in, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&bins[in[i]], 1);          // global atomic, ~400 cycles
}

// privatized: each block keeps a private histogram in shared memory, counts
// there (fast atomics, contention stays inside the block), then merges once.
__global__ void histShared(const int* in, int* bins, int n) {
    __shared__ int s[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) s[b] = 0;  // zero it
    __syncthreads();                          // all bins clean before counting

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&s[in[i]], 1);              // shared atomic, ~20 cycles
    __syncthreads();                          // all counting done before merge

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&bins[b], s[b]);            // one global atomic per bin per block
}

int main() {
    const int N = 16 * 1024 * 1024;
    size_t bytes = (size_t)N * sizeof(int);
    int* h = (int*)malloc(bytes);
    for (int i = 0; i < N; ++i) h[i] = 128;   // solid color: worst-case contention

    int *d_in, *d_bins;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_bins, BINS * sizeof(int));
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

    int block = 256, grid = std::min(256, (N + block - 1) / block);
    cudaMemset(d_bins, 0, BINS * sizeof(int));
    histShared<<<grid, block>>>(d_in, d_bins, N);   // swap for histGlobal to compare
    cudaDeviceSynchronize();

    int hb[BINS];
    cudaMemcpy(hb, d_bins, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("bins[128] = %d (expect %d)  %s\n", hb[128], N, hb[128] == N ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_bins); free(h);
    return 0;
}
