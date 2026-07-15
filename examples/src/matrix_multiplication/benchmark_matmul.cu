// benchmark_matmul.cu
// Comprehensive benchmark: Naive vs Tiled vs cuBLAS
// Build: nvcc -O2 -lcublas benchmark_matmul.cu -o benchmark_matmul

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(e));        \
        std::exit(1);                                              \
    }                                                              \
} while (0)

#define CUBLAS_CHECK(call) do {                                    \
    cublasStatus_t s = (call);                                     \
    if (s != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuBLAS error %s:%d\n", __FILE__, __LINE__);\
        std::exit(1);                                              \
    }                                                              \
} while (0)

#define TILE_SIZE 16

// ============================================================================
// KERNELS
// ============================================================================

// Naive kernel
__global__ void matmulNaive(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Tiled kernel with shared memory
__global__ void matmulTiled(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C,
                            int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++)
    {
        // Coalesced load: threadIdx.x varies → adjacent threads load adjacent floats
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Coalesced load: col depends on threadIdx.x → adjacent threads load adjacent floats
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // TILE_SIZE is compile-time constant → compiler unrolls this loop
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================================
// CPU REFERENCE
// ============================================================================

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

// ============================================================================
// BENCHMARK
// ============================================================================

void benchmark(int M, int N, int K)
{
    printf("\n");
    printf("================================================================================\n");
    printf("Matrix Size: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, K, K, N, M, N);
    printf("Total FLOPs: %.2f billion\n", 2.0 * M * N * K / 1e9);
    printf("================================================================================\n");
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory (pinned for consistent timing)
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, sizeA));
    CUDA_CHECK(cudaMallocHost(&h_B, sizeB));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeC));
    
    // Initialize
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
    
    // Setup
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int numRuns = 10;
    float milliseconds;
    double flops = 2.0 * M * N * K;
    
    // ========== CPU (single-threaded) ==========
    if (M <= 512)
    {
        auto t0 = std::chrono::steady_clock::now();
        matmulCPU(h_A, h_B, h_C, M, N, K);
        auto t1 = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double cpu_gflops = (flops / (cpu_ms / 1000.0)) / 1e9;
        printf("CPU (1 thread):     %8.2f ms  |  %6.2f GFLOPS\n", cpu_ms, cpu_gflops);
    }
    else
    {
        printf("CPU (1 thread):     [skipped - too slow for large matrices]\n");
    }
    
    // ========== GPU Naive ==========
    // Warmup
    matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numRuns; i++)
        matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float naive_ms = milliseconds / numRuns;
    double naive_gflops = (flops / (naive_ms / 1000.0)) / 1e9;
    printf("GPU Naive:          %8.2f ms  |  %6.2f GFLOPS\n", naive_ms, naive_gflops);
    
    // ========== GPU Tiled ==========
    // Warmup
    matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numRuns; i++)
        matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float tiled_ms = milliseconds / numRuns;
    double tiled_gflops = (flops / (tiled_ms / 1000.0)) / 1e9;
    printf("GPU Tiled (TILE=%d): %8.2f ms  |  %6.2f GFLOPS\n", TILE_SIZE, tiled_ms, tiled_gflops);
    
    // ========== cuBLAS ==========
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // which gives us row-major C = A * B
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             d_B, N, d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numRuns; i++)
    {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha,
                                 d_B, N, d_A, K, &beta, d_C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float cublas_ms = milliseconds / numRuns;
    double cublas_gflops = (flops / (cublas_ms / 1000.0)) / 1e9;
    printf("cuBLAS:             %8.2f ms  |  %6.2f GFLOPS\n", cublas_ms, cublas_gflops);
    
    // ========== Summary ==========
    printf("--------------------------------------------------------------------------------\n");
    printf("Speedup: Tiled vs Naive = %.1fx, cuBLAS vs Tiled = %.1fx\n",
           naive_ms / tiled_ms, tiled_ms / cublas_ms);
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
}

int main()
{
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, 
           prop.multiProcessorCount);
    printf("Memory: %.2f GB, Bandwidth: %.2f GB/s\n",
           prop.totalGlobalMem / 1e9,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    
    // Run benchmarks for different sizes
    benchmark(256, 256, 256);
    benchmark(512, 512, 512);
    benchmark(1024, 1024, 1024);
    benchmark(2048, 2048, 2048);
    
    return 0;
}
