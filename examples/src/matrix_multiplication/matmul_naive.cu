// matmul_naive.cu
// Naive matrix multiplication kernel - one thread per output element
// Build: nvcc -O2 matmul_naive.cu -o matmul_naive

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(e));        \
        std::exit(1);                                              \
    }                                                              \
} while (0)

// Naive matrix multiplication kernel
// Each thread computes one element of C
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < M && col < N)
    {
        float sum = 0.0f;
        
        // Dot product: row of A × column of B
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// CPU reference implementation
void matmulCPU(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv)
{
    // Matrix dimensions (can be passed as args)
    int M = (argc > 1) ? atoi(argv[1]) : 512;  // A is M×K
    int K = (argc > 2) ? atoi(argv[2]) : 512;  // B is K×N
    int N = (argc > 3) ? atoi(argv[3]) : 512;  // C is M×N
    
    printf("Matrix Multiplication (Naive): A(%d×%d) × B(%d×%d) = C(%d×%d)\n", 
           M, K, K, N, M, N);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];
    
    // Initialize matrices with random values
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Warmup
    matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int numRuns = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numRuns; i++)
    {
        matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avgTime = milliseconds / numRuns;
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;  // multiply-add = 2 FLOPs
    double gflops = (flops / (avgTime / 1000.0)) / 1e9;
    
    printf("GPU Naive: %.3f ms, %.2f GFLOPS\n", avgTime, gflops);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // CPU reference (for small matrices only)
    if (M <= 512 && N <= 512 && K <= 512)
    {
        matmulCPU(h_A, h_B, h_C_ref, M, N, K);
        
        // Verify
        float maxError = 0.0f;
        for (int i = 0; i < M * N; i++)
        {
            float error = fabs(h_C[i] - h_C_ref[i]);
            if (error > maxError) maxError = error;
        }
        printf("Max error vs CPU: %e\n", maxError);
        printf("Result: %s\n", maxError < 1e-3 ? "PASS" : "FAIL");
    }
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    
    return 0;
}
