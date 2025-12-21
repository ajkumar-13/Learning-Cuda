#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <omp.h>

// GPU kernel
__global__ void vectorAddGPU(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// CPU single-threaded
void vectorAddCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// CPU multi-threaded (OpenMP)
void vectorAddCPU_MT(const float *A, const float *B, float *C, int N)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void benchmark(int N)
{
    size_t size = (size_t)N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize
    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(i % 1000);
        h_B[i] = static_cast<float>((i * 2) % 1000);
    }

    // ========== CPU Single-threaded ==========
    auto start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C, N);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();

    // ========== CPU Multi-threaded (OpenMP) ==========
    start = std::chrono::high_resolution_clock::now();
    vectorAddCPU_MT(h_A, h_B, h_C, N);
    end = std::chrono::high_resolution_clock::now();
    double cpu_mt_time = std::chrono::duration<double, std::milli>(end - start).count();

    // ========== GPU (including memory transfer) ==========
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Warm up
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Timed run (including memory transfers)
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end - start).count();

    // ========== GPU (kernel only, no transfer) ==========
    start = std::chrono::high_resolution_clock::now();
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double gpu_kernel_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    printf("| %12d | %10.3f ms | %10.3f ms | %10.3f ms | %10.3f ms |\n",
           N, cpu_time, cpu_mt_time, gpu_time, gpu_kernel_time);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main()
{
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128); // Approximate for Turing/Ampere
    printf("OpenMP Threads: %d\n\n", omp_get_max_threads());

    printf("| %12s | %13s | %13s | %13s | %13s |\n",
           "N (elements)", "CPU (1 thread)", "CPU (OpenMP)", "GPU (w/ xfer)", "GPU (kernel)");
    printf("|--------------|---------------|---------------|---------------|---------------|\n");

    benchmark(1000);
    benchmark(10000);
    benchmark(100000);
    benchmark(1000000);
    benchmark(10000000);
    benchmark(100000000);  // 100 million elements, comment out if RAM is limited

    return 0;
}
