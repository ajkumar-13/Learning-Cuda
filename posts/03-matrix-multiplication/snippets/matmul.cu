// matmul.cu — naive and tiled (shared-memory) matrix multiply, C = A * B.
//
// Build: nvcc -O3 snippets/matmul.cu -o matmul   (from the post directory)
// Run:   ./matmul
//
// A is [M,K], B is [K,N], C is [M,N], all row-major. One thread computes one
// output element C[row,col] = dot(A row, B col). The tiled kernel loads
// TILE x TILE blocks of A and B into shared memory once and reuses them.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 16

#define CUDA_CHECK(call)                                                      \
    do { cudaError_t e = (call);                                             \
         if (e != cudaSuccess) {                                             \
             printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(e)); return 1; } } while (0)

// --- naive: every thread reads a full row of A and a full column of B ---
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];     // K reads from each of A, B
        C[row * N + col] = sum;
    }
}

// --- tiled: the block stages TILE x TILE tiles in shared memory and reuses them ---
__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;              // coalesced load of A
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;              // coalesced load of B
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                // tiles are loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();                                // done before reuse
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

void matmulCPU(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main()
{
    const int M = 1024, K = 1024, N = 1024;
    size_t bA = (size_t)M * K * 4, bB = (size_t)K * N * 4, bC = (size_t)M * N * 4;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, bA); cudaMallocHost(&h_B, bB); cudaMallocHost(&h_C, bC);
    float* h_ref = new float[(size_t)M * N];
    for (int i = 0; i < M * K; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bA)); CUDA_CHECK(cudaMalloc(&d_B, bB)); CUDA_CHECK(cudaMalloc(&d_C, bC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bC, cudaMemcpyDeviceToHost));

    matmulCPU(h_A, h_B, h_ref, M, N, K);
    float maxErr = 0.0f;
    for (int i = 0; i < M * N; ++i) maxErr = fmaxf(maxErr, fabsf(h_C[i] - h_ref[i]));
    printf("%dx%dx%d  max error %e  %s\n", M, K, N, maxErr, maxErr < 1e-3f ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); delete[] h_ref;
    return 0;
}
