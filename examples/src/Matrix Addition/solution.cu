#include <cuda_runtime.h>


__global__ void matrix_add(const float* A, const float* B, float* C, int N) 
{
    int workindex = blockDim.x * blockIdx.x + threadIdx.x;

    int row = workindex / N;
    int col = workindex % N;

    if (row < N && col < N)
    {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
