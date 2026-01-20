// matmul_tiled.cu
// Tiled matrix multiplication with shared memory
// Build: nvcc -O2 matmul_tiled.cu -o matmul_tiled

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

#define TILE_SIZE 16

// Tiled matrix multiplication kernel
// Uses shared memory to reduce global memory traffic
// __restrict__ tells the compiler pointers don't overlap, enabling more optimizations
__global__ void matmulTiled(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C,
                            int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Thread's position in the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Number of tiles needed to cover K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over tiles along the K dimension
    for (int t = 0; t < numTiles; t++)
    {
        // Load tile from A: Adjacent threads (varying threadIdx.x) load adjacent floats → COALESCED
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile from B: col varies with threadIdx.x → COALESCED
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Wait for all threads to finish loading
        __syncthreads();
        
        // Compute partial dot product for this tile
        // TILE_SIZE is compile-time constant, so compiler can unroll this loop
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Wait before loading next tile (prevent data race)
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N)
    {
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
    int M = (argc > 1) ? atoi(argv[1]) : 1024;  // A is M×K
    int K = (argc > 2) ? atoi(argv[2]) : 1024;  // B is K×N
    int N = (argc > 3) ? atoi(argv[3]) : 1024;  // C is M×N
    
    printf("Matrix Multiplication (Tiled, TILE_SIZE=%d): A(%d×%d) × B(%d×%d) = C(%d×%d)\n", 
           TILE_SIZE, M, K, K, N, M, N);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Print memory usage
    printf("Memory: A=%.2f MB, B=%.2f MB, C=%.2f MB\n",
           sizeA / 1e6, sizeB / 1e6, sizeC / 1e6);
    
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
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Warmup
    matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int numRuns = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numRuns; i++)
    {
        matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avgTime = milliseconds / numRuns;
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;  // multiply-add = 2 FLOPs
    double gflops = (flops / (avgTime / 1000.0)) / 1e9;
    
    printf("GPU Tiled: %.3f ms, %.2f GFLOPS\n", avgTime, gflops);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // CPU reference (for small matrices only)
    if (M <= 512 && N <= 512 && K <= 512)
    {
        printf("Running CPU reference...\n");
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
    else
    {
        // Spot check for large matrices
        printf("Spot check C[0][0] = %.6f\n", h_C[0]);
        printf("Spot check C[M-1][N-1] = %.6f\n", h_C[(M-1)*N + (N-1)]);
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
