/**
 * vector_add_benchmark.cu - Benchmarking vector addition
 * 
 * Demonstrates:
 * - CUDA event timing
 * - Pinned memory for faster transfers
 * - Proper benchmarking methodology
 * - CPU vs GPU performance comparison
 * 
 * Compile: nvcc -O3 -o vector_add_benchmark vector_add_benchmark.cu
 * Run:     ./vector_add_benchmark
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

void vectorAddCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    printf("============================================\n");
    printf("   Vector Addition Benchmark\n");
    printf("============================================\n\n");
    
    // Test multiple sizes
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    
    const int NUM_ITERATIONS = 10;
    const int WARMUP = 3;
    
    printf("%-15s %-12s %-12s %-12s %-10s\n", 
           "Size", "CPU (ms)", "GPU (ms)", "Kernel (ms)", "Speedup");
    printf("─────────────────────────────────────────────────────────────\n");
    
    for (int s = 0; s < numSizes; s++)
    {
        int N = sizes[s];
        size_t bytes = N * sizeof(float);
        
        // Allocate pinned host memory (faster transfers)
        float *h_A, *h_B, *h_C, *h_C_ref;
        CUDA_CHECK(cudaMallocHost(&h_A, bytes));
        CUDA_CHECK(cudaMallocHost(&h_B, bytes));
        CUDA_CHECK(cudaMallocHost(&h_C, bytes));
        h_C_ref = (float*)malloc(bytes);
        
        // Initialize
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
        
        // Create CUDA events for timing
        cudaEvent_t start, stop, kernelStart, kernelStop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelStop);
        
        int threadsPerBlock = 256;
        int blocksPerGrid = min((N + threadsPerBlock - 1) / threadsPerBlock, 65535);
        
        // Warmup
        for (int i = 0; i < WARMUP; i++)
        {
            CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
            vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark CPU
        auto cpuStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            vectorAddCPU(h_A, h_B, h_C_ref, N);
        }
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        float cpuTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count() / NUM_ITERATIONS;
        
        // Benchmark GPU (total including transfers)
        float gpuTotalTime = 0.0f;
        float gpuKernelTime = 0.0f;
        
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            cudaEventRecord(start);
            
            CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
            
            cudaEventRecord(kernelStart);
            vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            cudaEventRecord(kernelStop);
            
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float totalMs, kernelMs;
            cudaEventElapsedTime(&totalMs, start, stop);
            cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop);
            
            gpuTotalTime += totalMs;
            gpuKernelTime += kernelMs;
        }
        gpuTotalTime /= NUM_ITERATIONS;
        gpuKernelTime /= NUM_ITERATIONS;
        
        // Verify
        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
        {
            float error = fabs(h_C[i] - h_C_ref[i]);
            if (error > maxError) maxError = error;
        }
        
        float speedup = cpuTime / gpuTotalTime;
        
        printf("%-15d %-12.3f %-12.3f %-12.3f %-10.1fx %s\n",
               N, cpuTime, gpuTotalTime, gpuKernelTime, speedup,
               maxError < 1e-5 ? "✓" : "✗");
        
        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);
        free(h_C_ref);
    }
    
    printf("\n");
    printf("Notes:\n");
    printf("- GPU time includes memory transfers (H2D + D2H)\n");
    printf("- Kernel time is compute only\n");
    printf("- Speedup improves with larger arrays due to transfer amortization\n");
    
    return 0;
}
