// histogram.cu: count N = 16,777,216 ints into 256 bins two ways, with global atomics (histGlobal) and
// with a private histogram per block in shared memory (histShared), on random and on solid-color data,
// and time each kernel with CUDA events. A third kernel, histRacy, increments without an atomic, to show
// the lost updates of the race.
//
// Build: nvcc -O3 -arch=sm_75 posts/05-histogram/snippets/histogram.cu -o histogram   (from the series root)
// Run:   ./histogram        (on Windows: .\histogram.exe)
//
// A laptop GPU idles at a low clock and needs a few hundred milliseconds of work to reach its boost clock,
// so the program first runs histGlobal untimed 100 times. Then each kernel gets one untimed warm-up launch
// and ITERS timed launches; the global bins are zeroed before every launch, outside the timed interval. Bandwidth is the 4 * N bytes of input read, divided
// by the average kernel time. Every timed run's bins are compared with a histogram counted on the host.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BINS 256

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// The race: a plain read-modify-write, so two threads can read the same old count and one increment is lost.
__global__ void histRacy(const int* in, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        bins[in[i]]++;                        // read, add 1, write back: not indivisible
}

__global__ void histGlobal(const int* in, int* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&bins[in[i]], 1);           // one global atomic per element
}

__global__ void histShared(const int* in, int* bins, int n) {
    __shared__ int s[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) s[b] = 0;   // 1. zero
    __syncthreads();                          // every bin is zero before any thread counts

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        atomicAdd(&s[in[i]], 1);              // 2. count: shared-memory atomic
    __syncthreads();                          // every thread has counted before any bin is merged

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&bins[b], s[b]);            // 3. merge: one global atomic per bin per block
}

typedef void (*Kernel)(const int*, int*, int);

int main() {
    const int N = 16 * 1024 * 1024, ITERS = 20;
    const int block = 256, grid = 256;        // 65,536 threads, each striding over 256 elements
    const size_t bytes = (size_t)N * sizeof(int);

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);

    // Two inputs: uniform random values 0..255 (a fixed linear congruential generator) and a solid color.
    std::vector<int> random(N), solid(N, 128);
    unsigned int x = 12345u;
    for (int i = 0; i < N; ++i) { x = 1664525u * x + 1013904223u; random[i] = (int)(x >> 24); }

    int *d_in, *d_bins;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_bins, BINS * sizeof(int)));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Clock warm-up: without it the first cell is timed while the GPU is still leaving its idle clock.
    CUDA_CHECK(cudaMemcpy(d_in, random.data(), bytes, cudaMemcpyHostToDevice));
    for (int w = 0; w < 100; ++w) histGlobal<<<grid, block>>>(d_in, d_bins, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("N = %d ints, %d bins, <<<%d, %d>>>, average of %d timed runs after warm-up\n",
           N, BINS, grid, block, ITERS);
    printf("data    kernel       time (ms)    GB/s   bins match host\n");

    const char* dataName[2] = {"random", "solid"};
    std::vector<int>* data[2] = {&random, &solid};
    const char* kernelName[2] = {"histGlobal", "histShared"};
    Kernel kernels[2] = {histGlobal, histShared};
    int racyTotal[2] = {0, 0};
    bool allOk = true;

    for (int d = 0; d < 2; ++d) {
        CUDA_CHECK(cudaMemcpy(d_in, data[d]->data(), bytes, cudaMemcpyHostToDevice));
        int ref[BINS] = {0};
        for (int i = 0; i < N; ++i) ref[(*data[d])[i]]++;

        for (int k = 0; k < 2; ++k) {
            float total = 0.0f;
            bool ok = true;
            for (int it = -1; it < ITERS; ++it) {             // it = -1 is the untimed warm-up
                CUDA_CHECK(cudaMemset(d_bins, 0, BINS * sizeof(int)));
                CUDA_CHECK(cudaEventRecord(start));
                kernels[k]<<<grid, block>>>(d_in, d_bins, N);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                if (it >= 0) total += ms;
                int hb[BINS];
                CUDA_CHECK(cudaMemcpy(hb, d_bins, BINS * sizeof(int), cudaMemcpyDeviceToHost));
                for (int b = 0; b < BINS; ++b) ok = ok && (hb[b] == ref[b]);
            }
            double ms = total / ITERS;
            printf("%-7s %-11s %10.3f %7.1f   %s\n", dataName[d], kernelName[k], ms,
                   bytes / (ms * 1e6), ok ? "yes" : "NO");
            allOk = allOk && ok;
        }

        CUDA_CHECK(cudaMemset(d_bins, 0, BINS * sizeof(int)));
        histRacy<<<grid, block>>>(d_in, d_bins, N);
        CUDA_CHECK(cudaGetLastError());
        int hb[BINS];
        CUDA_CHECK(cudaMemcpy(hb, d_bins, BINS * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < BINS; ++b) racyTotal[d] += hb[b];
    }

    for (int d = 0; d < 2; ++d)
        printf("histRacy (no atomic), %s data: counted %d of %d increments\n", dataName[d], racyTotal[d], N);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bins));
    return allOk ? 0 : 1;
}
