/**
 * vector_add.cu - Basic vector addition kernel
 * 
 * Demonstrates:
 * - Memory allocation (cudaMalloc, cudaFree)
 * - Data transfer (cudaMemcpy)
 * - Kernel launch configuration
 * - Thread indexing with grid-stride loops
 * 
 * Compile: nvcc -O3 -o vector_add vector_add.cu
 * Run:     ./vector_add
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for handling large arrays
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

// CPU reference implementation
void vectorAddCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1000000;
    const size_t bytes = N * sizeof(float);
    
    printf("Vector Addition Example\n");
    printf("Vector size: %d elements (%.2f MB)\n", N, bytes / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launch config: %d blocks, %d threads/block\n", blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // CPU reference
    vectorAddCPU(h_A, h_B, h_C_ref, N);
    
    // Verify result
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float error = fabs(h_C[i] - h_C_ref[i]);
        if (error > maxError) maxError = error;
    }
    
    printf("Max error: %e\n", maxError);
    printf("Result: %s\n", maxError < 1e-5 ? "PASS" : "FAIL");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}
