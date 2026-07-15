// vector_add.cu — your first CUDA kernel: C[i] = A[i] + B[i].
//
// Build: nvcc -O2 snippets/vector_add.cu -o vector_add   (from the post directory)
// Run:   ./vector_add
//
// One thread per element. Because we launch whole blocks, the grid is almost
// always slightly larger than N, so every thread guards its write with i < N.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Wrap every CUDA call: a launch reports errors asynchronously, so unchecked
// failures surface later in some unrelated call. CUDA_CHECK makes them local.
#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                                 \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        C[i] = A[i] + B[i];                          // the whole kernel: one add
}

int main()
{
    const int N = 1000003;                  // deliberately NOT a multiple of 256
    const size_t bytes = (size_t)N * sizeof(float);

    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for (int i = 0; i < N; ++i) { h_A[i] = (float)i; h_B[i] = 0.5f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int blocksPerGrid = (N + blockSize - 1) / blockSize;  // ceiling division
    vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());          // catch a bad launch configuration
    CUDA_CHECK(cudaDeviceSynchronize());     // wait, then catch a runtime fault

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify every element, not just the endpoints: a stray index or an
    // unguarded surplus thread would corrupt some interior cell we'd miss.
    int mismatches = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f)
            ++mismatches;

    printf("C[0]      = %.1f (expect 0.5)\n", h_C[0]);
    printf("C[N-1]    = %.1f (expect %.1f)\n", h_C[N - 1], (float)(N - 1) + 0.5f);
    printf("verified  = %s (%d / %d elements correct)\n",
           mismatches == 0 ? "PASS" : "FAIL", N - mismatches, N);

    delete[] h_A; delete[] h_B; delete[] h_C;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return mismatches == 0 ? 0 : 1;   // non-zero exit on any mismatch
}
