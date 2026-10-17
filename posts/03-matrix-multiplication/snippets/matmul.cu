// matmul.cu: the naive and the tiled (shared-memory) matrix multiply C = A * B, timed against cuBLAS.
//
// Build: nvcc -O3 -arch=sm_75 posts/03-matrix-multiplication/snippets/matmul.cu -o matmul -lcublas   (from the series root)
// Run:   ./matmul        (on Windows: .\matmul.exe)
//
// A: [M, K], B: [K, N], C: [M, N], all row-major floats. One thread of the naive and the tiled kernel
// computes one output element C[row * N + col], the dot product of row `row` of A and column `col` of B;
// the tiled kernel stages TILE x TILE tiles of A and B in shared memory (s_A, s_B) and reuses them.
// For square sizes 256, 512, 1024 and 2048 the program times each kernel and cublasSgemm with CUDA
// events (after 2 untimed warm-up launches, the median of 5 runs of 10 timed launches each), prints
// ms and GFLOP/s (2 * M * N * K floating-point operations per product), and checks every result
// against a double-precision CPU reference on 4,096 sampled elements. It exits 1 if a check fails.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define TILE 16

#define CUDA_CHECK(call)                                                      \
    do { cudaError_t e = (call);                                             \
         if (e != cudaSuccess) {                                             \
             printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];     // K loads from each of A and B
        C[row * N + col] = sum;
    }
}

__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[TILE][TILE];                   // launched with blockDim = (TILE, TILE)
    __shared__ float s_B[TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;              // coalesced load of A
        s_A[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;              // coalesced load of B
        s_B[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                // tiles are loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();                                // done before the next load
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

static const int WARMUP = 2, RUNS = 5, ITERS = 10, SAMPLES = 4096;

// cuBLAS is column-major: the row-major C = A * B is the column-major C^T = B^T * A^T.
static void sgemm(cublasHandle_t h, const float* d_A, const float* d_B, float* d_C, int M, int N, int K)
{
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
}

static void launch(int which, cublasHandle_t h, const float* d_A, const float* d_B, float* d_C, int n)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    if (which == 0)      matmulNaive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
    else if (which == 1) matmulTiled<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
    else                 sgemm(h, d_A, d_B, d_C, n, n, n);
}

// largest relative error of C over SAMPLES elements, against a double-precision dot product
static double check(const float* h_A, const float* h_B, const float* h_C, int n)
{
    double worst = 0.0;
    unsigned s = 12345u;
    for (int q = 0; q < SAMPLES; ++q) {
        s = s * 1664525u + 1013904223u; int r = (int)(s % (unsigned)n);
        s = s * 1664525u + 1013904223u; int c = (int)(s % (unsigned)n);
        double ref = 0.0;
        for (int k = 0; k < n; ++k) ref += (double)h_A[(size_t)r * n + k] * h_B[(size_t)k * n + c];
        double err = fabs(h_C[(size_t)r * n + c] - ref) / fmax(fabs(ref), 1e-30);
        if (err > worst) worst = err;
    }
    return worst;
}

int main()
{
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    printf("C = A * B with M = N = K = size; ms per launch, median of %d runs of %d launches\n", RUNS, ITERS);
    printf("  size    naive ms  GFLOP/s    tiled ms  GFLOP/s   cuBLAS ms  GFLOP/s   max rel err\n");

    const int sizes[] = {256, 512, 1024, 2048};
    const int maxN = 2048;
    size_t maxBytes = (size_t)maxN * maxN * sizeof(float);
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMallocHost(&h_A, maxBytes));
    CUDA_CHECK(cudaMallocHost(&h_B, maxBytes));
    CUDA_CHECK(cudaMallocHost(&h_C, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_A, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_B, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_C, maxBytes));
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 1; }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // bring the clocks up before the first timed size: about half a second of 1024^3 products
    CUDA_CHECK(cudaMemset(d_A, 0, maxBytes));
    CUDA_CHECK(cudaMemset(d_B, 0, maxBytes));
    for (int r = 0; r < 60; ++r) launch(1, h, d_A, d_B, d_C, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    for (int n : sizes) {
        size_t elems = (size_t)n * n, bytes = elems * sizeof(float);
        srand(1);
        for (size_t i = 0; i < elems; ++i) h_A[i] = (rand() % 100) / 100.0f;
        for (size_t i = 0; i < elems; ++i) h_B[i] = (rand() % 100) / 100.0f;
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        double flop = 2.0 * n * n * n, ms[3], worst = 0.0;
        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            for (int r = 0; r < WARMUP; ++r) launch(w, h, d_A, d_B, d_C, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            float run[RUNS];
            for (int q = 0; q < RUNS; ++q) {
                CUDA_CHECK(cudaEventRecord(start));
                for (int r = 0; r < ITERS; ++r) launch(w, h, d_A, d_B, d_C, n);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventElapsedTime(&run[q], start, stop));
                run[q] /= ITERS;
            }
            std::sort(run, run + RUNS);
            ms[w] = run[RUNS / 2];
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
            double e = check(h_A, h_B, h_C, n);
            if (e > worst) worst = e;
        }
        if (!(worst < 1e-4)) ok = false;
        printf("  %4d  %10.3f  %7.0f  %10.3f  %7.0f  %10.3f  %7.0f   %.1e\n", n,
               ms[0], flop / (ms[0] * 1e6), ms[1], flop / (ms[1] * 1e6), ms[2], flop / (ms[2] * 1e6), worst);
    }
    printf("every result matches the CPU reference (relative error below 1e-4): %s\n", ok ? "PASS" : "FAIL");

    cublasDestroy(h);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    return ok ? 0 : 1;
}
