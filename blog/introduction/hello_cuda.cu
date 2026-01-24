/**
 * hello_cuda.cu - Your first CUDA program
 * 
 * This minimal example demonstrates:
 * - Defining a __global__ kernel function
 * - Launching a kernel with <<<blocks, threads>>>
 * - Synchronizing with cudaDeviceSynchronize()
 */

#include <cstdio>

// A simple kernel that prints from the GPU
__global__ void helloFromGPU()
{
    // Each thread prints its ID
    printf("Hello from GPU! Thread %d in Block %d\n", 
           threadIdx.x, blockIdx.x);
}

int main()
{
    printf("Hello from CPU!\n");
    
    // Launch kernel with 2 blocks, 4 threads per block
    // Total: 8 threads will execute
    helloFromGPU<<<2, 4>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("Back to CPU!\n");
    
    return 0;
}
