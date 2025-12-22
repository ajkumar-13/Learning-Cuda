// benchmark_vector_add.cu
// Proper CUDA benchmark: uses CUDA events, pinned memory, repetitions, and checksum to prevent dead-code elimination
// Build (Windows): nvcc -O2 -Xcompiler "/openmp" benchmark_vector_add.cu -o benchmark_vector_add.exe
// Build (Linux):   nvcc -O3 -std=c++17 -Xcompiler -fopenmp benchmark_vector_add.cu -o bench

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <omp.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(e));        \
        std::exit(1);                                              \
    }                                                              \
} while (0)

__global__ void vectorAddGPU(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

static inline void vectorAddCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

static inline void vectorAddCPU_MT(const float *A, const float *B, float *C, int N)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

// Prevent dead-code elimination by consuming the result
static volatile double g_sink = 0.0;

static inline double checksum_sampled(const float* C, int N)
{
    const int stride = 1024;
    double s = 0.0;
    for (int i = 0; i < N; i += stride) s += C[i];
    return s;
}

static inline int choose_cpu_reps(int N)
{
    const long long target_elems = 200LL * 1000LL * 1000LL;
    long long reps = target_elems / (std::max)(1, N);
    if (reps < 1) reps = 1;
    if (reps > 2000) reps = 2000;
    return (int)reps;
}

static inline int choose_gpu_reps(int N)
{
    const long long target_elems = 50LL * 1000LL * 1000LL;
    long long reps = target_elems / (std::max)(1, N);
    if (reps < 1) reps = 1;
    if (reps > 500) reps = 500;
    return (int)reps;
}

static inline bool validate_sampled(const float* ref, const float* out, int N)
{
    if (N <= 0) return true;
    const int idxs[] = {0, N/3, N/2, (2*N)/3, N-1};
    for (int k = 0; k < 5; k++) {
        int i = idxs[k];
        float diff = std::fabs(ref[i] - out[i]);
        float tol  = 1e-5f * (std::max)(1.0f, std::fabs(ref[i]));
        if (diff > tol) return false;
    }
    return true;
}

void benchmark(int N)
{
    size_t size = (size_t)N * sizeof(float);

    // Pinned host memory for consistent transfer timing
    float *h_A=nullptr, *h_B=nullptr, *h_C_cpu=nullptr, *h_C_gpu=nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size));
    CUDA_CHECK(cudaMallocHost(&h_B, size));
    CUDA_CHECK(cudaMallocHost(&h_C_cpu, size));
    CUDA_CHECK(cudaMallocHost(&h_C_gpu, size));

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 1000);
        h_B[i] = (float)((i * 2) % 1000);
    }

    float *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Warmup
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpyAsync(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    g_sink += checksum_sampled(h_C_cpu, N);

    vectorAddCPU_MT(h_A, h_B, h_C_cpu, N);
    g_sink += checksum_sampled(h_C_cpu, N);

    // CPU timings (average over multiple reps)
    int cpu_reps = choose_cpu_reps(N);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < cpu_reps; r++) {
        vectorAddCPU(h_A, h_B, h_C_cpu, N);
        g_sink += checksum_sampled(h_C_cpu, N);
    }
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_reps;

    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < cpu_reps; r++) {
        vectorAddCPU_MT(h_A, h_B, h_C_cpu, N);
        g_sink += checksum_sampled(h_C_cpu, N);
    }
    t1 = std::chrono::steady_clock::now();
    double omp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_reps;

    // GPU timings (average) using CUDA events for accurate measurement
    int gpu_reps = choose_gpu_reps(N);

    // GPU w/ transfers
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int r = 0; r < gpu_reps; r++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
        vectorAddGPU<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaMemcpyAsync(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));
    double gpu_xfer_ms = (double)total_ms / gpu_reps;

    // Ensure inputs on device for kernel-only measurement
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // GPU kernel-only
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int r = 0; r < gpu_reps; r++) {
        vectorAddGPU<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e0, e1));
    double gpu_kernel_ms = (double)kernel_ms / gpu_reps;

    // Correctness check
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpyAsync(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bool ok = validate_sampled(h_C_cpu, h_C_gpu, N);

    printf("| %12d | %10.3f ms | %10.3f ms | %10.3f ms | %10.3f ms | %s |\n",
           N, cpu_ms, omp_ms, gpu_xfer_ms, gpu_kernel_ms, ok ? "OK" : "BAD");

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C_cpu));
    CUDA_CHECK(cudaFreeHost(h_C_gpu));
}

int main()
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d) | SMs: %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("OpenMP max threads: %d\n\n", omp_get_max_threads());

    printf("| %12s | %13s | %13s | %13s | %13s | %s |\n",
           "N (elements)", "CPU (1 thr)", "CPU (OpenMP)", "GPU (w/ xfer)", "GPU (kernel)", "check");
    printf("|--------------|---------------|---------------|---------------|---------------|-------|\n");

    benchmark(1000);
    benchmark(10000);
    benchmark(100000);
    benchmark(1000000);
    benchmark(10000000);
    benchmark(100000000);  // 100 million

    fprintf(stderr, "sink=%f\n", (double)g_sink);
    return 0;
}
