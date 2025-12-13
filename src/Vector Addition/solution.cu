#include <cuda_runtime.h>

// CUDA kernel: C[i] = A[i] + B[i]
__global__ void vector_add(const float* A, const float* B, float* C, int N)
{
    int workindex = blockDim.x * blockIdx.x + threadIdx.x;
    if (workindex < N)
    {
        C[workindex] = A[workindex] + B[workindex];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
