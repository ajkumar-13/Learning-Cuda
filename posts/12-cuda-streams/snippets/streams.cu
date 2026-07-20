// streams.cu — overlap copy and compute by splitting the work across CUDA
// streams. Each stream runs H2D -> kernel -> D2H in order, but different streams
// run concurrently on the GPU's independent copy and compute engines.
//
// Build: nvcc -O3 -arch=sm_75 snippets/streams.cu -o streams   (from the post directory)
// Run:   ./streams

#include <cuda_runtime.h>
#include <cstdio>

#define N (1 << 24)        // 16M elements per array
#define NS 4               // number of streams

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    int chunk = N / NS;
    size_t chunkBytes = bytes / NS;

    // pinned host memory is REQUIRED for cudaMemcpyAsync to be truly async
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes); cudaMallocHost(&h_b, bytes); cudaMallocHost(&h_c, bytes);
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);

    cudaStream_t s[NS];
    for (int i = 0; i < NS; ++i) cudaStreamCreate(&s[i]);

    dim3 block(256), grid((chunk + 255) / 256);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int i = 0; i < NS; ++i) {                 // issue all streams; they overlap
        int off = i * chunk;
        cudaMemcpyAsync(&d_a[off], &h_a[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
        cudaMemcpyAsync(&d_b[off], &h_b[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
        vectorAdd<<<grid, block, 0, s[i]>>>(&d_a[off], &d_b[off], &d_c[off], chunk);
        cudaMemcpyAsync(&h_c[off], &d_c[off], chunkBytes, cudaMemcpyDeviceToHost, s[i]);
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("streamed: %.2f ms,  %.1f GB/s effective\n", ms, (3.0 * bytes / 1e9) / (ms / 1e3));

    for (int i = 0; i < NS; ++i) cudaStreamDestroy(s[i]);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
