#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {

    int workindex = blockDim.x*blockIdx.x + threadIdx.x;

    if(workindex < N/2)
    {
        float temp = input[N - workindex - 1];
        input[N - workindex - 1] = input[workindex];
        input[workindex] = temp;
    }

}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}