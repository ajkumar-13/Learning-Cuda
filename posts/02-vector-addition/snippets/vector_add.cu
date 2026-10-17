// vector_add.cu: the first complete CUDA program of the series, C[i] = A[i] + B[i].
//
// Build: nvcc -O3 -arch=sm_75 posts/02-vector-addition/snippets/vector_add.cu -o vector_add   (from the series root)
// Run:   ./vector_add        (on Windows: .\vector_add.exe)
//
// One thread per element. The launch comes in whole blocks, so the grid is almost always slightly
// larger than N, and every thread guards its write with i < N. The program checks every element.

#include <cuda_runtime.h>
#include <cstdio>

// Wrap every CUDA call: a launch reports errors asynchronously, so an unchecked failure surfaces
// later in some unrelated call. CUDA_CHECK makes each failure name its own line.
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
        C[i] = A[i] + B[i];                          // the work: one add
}

int main()
{
    const int N = 1000003;                  // deliberately not a multiple of 256
    const size_t bytes = (size_t)N * sizeof(float);
    const int blockSize = 256;
    const int blocksPerGrid = (N + blockSize - 1) / blockSize;   // round up, never down

    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for (int i = 0; i < N; ++i) { h_A[i] = (float)i; h_B[i] = 0.5f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));                                   // 1. allocate
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));       // 2. copy inputs in
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    vectorAdd<<<blocksPerGrid, blockSize>>>(d_A, d_B, d_C, N);             // 3. launch
    CUDA_CHECK(cudaGetLastError());         // a bad launch configuration, reported now
    CUDA_CHECK(cudaDeviceSynchronize());    // 4. wait, then report any fault inside the kernel

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));       // 5. copy the result out
    CUDA_CHECK(cudaFree(d_A));                                             // 6. free
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Check every element, not just the ends: a stray index or an unguarded surplus thread
    // would corrupt an interior cell that a spot check misses.
    int mismatches = 0;
    for (int i = 0; i < N; ++i)
        if (h_C[i] != h_A[i] + h_B[i])
            ++mismatches;

    printf("vectorAdd<<<%d, %d>>> over N = %d: %d threads, %d surplus\n",
           blocksPerGrid, blockSize, N, blocksPerGrid * blockSize, blocksPerGrid * blockSize - N);
    printf("C[0] = %.1f, C[N-1] = %.1f\n", h_C[0], h_C[N - 1]);
    printf("elements equal to A + B: %d of %d\n", N - mismatches, N);

    delete[] h_A; delete[] h_B; delete[] h_C;
    return mismatches == 0 ? 0 : 1;         // non-zero exit on any mismatch
}
