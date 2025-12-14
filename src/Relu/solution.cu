#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    

    if(index < N)
    {
        if(input[index]<=0)
        {
            output[index] = 0.0f;
        }

        else
        {
            output[index] = input[index];
        }
    }


}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
