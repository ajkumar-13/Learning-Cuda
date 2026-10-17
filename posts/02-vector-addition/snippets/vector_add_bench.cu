// vector_add_bench.cu: the benchmark of post 02. It times C[i] = A[i] + B[i] four ways on the
// reader's machine and prints the post's table: one CPU thread, the CPU with OpenMP, the GPU with
// the copies across the bus, and the GPU kernel alone on data already in device memory.
//
// Build: nvcc -O3 -arch=sm_75 -Xcompiler /openmp posts/02-vector-addition/snippets/vector_add_bench.cu -o vector_add_bench   (from the series root)
//        (on Linux the OpenMP flag is -Xcompiler -fopenmp)
// Run:   ./vector_add_bench        (on Windows: .\vector_add_bench.exe)
//
// Method: pinned host memory (cudaMallocHost) for the copies, one untimed warm-up of every
// variant, then an average over many repetitions; GPU times come from CUDA events, CPU times
// from std::chrono::steady_clock. Every result is checked element by element against A + B, so
// no variant can skip its work. Sizes stop at 10 million elements so that the three pinned host
// arrays (120 MB) fit beside other programs on a laptop with 8 GB of memory.

#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                                 \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

static void addCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

static void addOpenMP(const float *A, const float *B, float *C, int N)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

// Count the elements of C that are not exactly A + B (the inputs are small integers, so the
// float sums are exact). Also clears C, so the next variant cannot pass on stale results.
static int wrong(const float *A, const float *B, float *C, int N)
{
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        bad += (C[i] != A[i] + B[i]);
        C[i] = -1.0f;
    }
    return bad;
}

// Average milliseconds of one call of f over reps calls, on the host clock.
template <typename F> static double hostMs(F f, int reps)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

// Average milliseconds of one call of f over reps calls, between two CUDA events on stream s.
template <typename F> static double deviceMs(F f, int reps, cudaStream_t s)
{
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, s));
    for (int r = 0; r < reps; ++r) f();
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return ms / reps;
}

// A time in microseconds below 1 ms, in milliseconds above.
static const char *fmt(double ms, char *buf)
{
    if (ms < 1.0) snprintf(buf, 16, "%7.1f us", ms * 1000.0);
    else          snprintf(buf, 16, "%7.2f ms", ms);
    return buf;
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak = 2.0 * (prop.memoryBusWidth / 8) * (prop.memoryClockRate * 1e3) / 1e9;
    printf("GPU: %s (compute capability %d.%d), %d SMs, %.0f GB/s peak (theoretical)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, peak);
    printf("CPU: %d OpenMP threads\n", omp_get_max_threads());

    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    const int maxN = 10000000;
    const size_t maxBytes = (size_t)maxN * sizeof(float);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMallocHost(&h_A, maxBytes));        // pinned (page-locked) host memory
    CUDA_CHECK(cudaMallocHost(&h_B, maxBytes));
    CUDA_CHECK(cudaMallocHost(&h_C, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_A, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_B, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_C, maxBytes));
    for (int i = 0; i < maxN; ++i) {
        h_A[i] = (float)(i % 1000);
        h_B[i] = (float)((2 * i) % 1000);
    }
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    printf("each time: the average of R repetitions after one untimed warm-up\n\n");
    printf("%10s %6s %12s %12s %12s %12s %11s %6s\n", "N", "R", "CPU 1 thr", "CPU OpenMP",
           "GPU+copies", "GPU kernel", "kernel GB/s", "check");

    int failures = 0;
    for (int N : sizes) {
        const size_t bytes = (size_t)N * sizeof(float);
        const int threads = 256, blocks = (N + threads - 1) / threads;
        const int reps = (std::max)(20, (std::min)(2000, 200000000 / N));

        auto cpu = [&] { addCPU(h_A, h_B, h_C, N); };
        auto omp = [&] { addOpenMP(h_A, h_B, h_C, N); };
        auto kernel = [&] { vectorAdd<<<blocks, threads, 0, s>>>(d_A, d_B, d_C, N); };
        auto copies = [&] {
            CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, s));
            kernel();
            CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, s));
        };

        cpu();                                             // warm-up, then check
        int bad = wrong(h_A, h_B, h_C, N);
        double cpuMs = hostMs(cpu, reps);
        bad += wrong(h_A, h_B, h_C, N);

        omp();
        bad += wrong(h_A, h_B, h_C, N);
        double ompMs = hostMs(omp, reps);
        bad += wrong(h_A, h_B, h_C, N);

        copies();
        CUDA_CHECK(cudaStreamSynchronize(s));
        bad += wrong(h_A, h_B, h_C, N);
        double copyMs = deviceMs(copies, reps, s);
        CUDA_CHECK(cudaGetLastError());
        bad += wrong(h_A, h_B, h_C, N);

        CUDA_CHECK(cudaMemsetAsync(d_C, 0, bytes, s));     // the inputs are already on the device
        kernel();
        double kernelMs = deviceMs(kernel, reps, s);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        bad += wrong(h_A, h_B, h_C, N);

        char b1[16], b2[16], b3[16], b4[16];
        printf("%10d %6d %12s %12s %12s %12s %11.1f %6s\n", N, reps, fmt(cpuMs, b1), fmt(ompMs, b2),
               fmt(copyMs, b3), fmt(kernelMs, b4), 12.0 * N / (kernelMs * 1e6), bad ? "FAIL" : "ok");
        failures += bad;
    }

    // One copy of 40 MB to the device, from pinned memory and from ordinary (pageable) memory.
    std::vector<float> pageable(h_A, h_A + maxN);
    auto pinnedCopy = [&] { CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, maxBytes, cudaMemcpyHostToDevice, s)); };
    auto pageableCopy = [&] { CUDA_CHECK(cudaMemcpyAsync(d_A, pageable.data(), maxBytes, cudaMemcpyHostToDevice, s)); };
    pinnedCopy();
    pageableCopy();
    double pinnedMs = deviceMs(pinnedCopy, 20, s), pageableMs = deviceMs(pageableCopy, 20, s);
    printf("\nhost to device, %d MB: pinned %.1f GB/s, pageable %.1f GB/s\n", (int)(maxBytes / 1000000),
           maxBytes / (pinnedMs * 1e6), maxBytes / (pageableMs * 1e6));

    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B)); CUDA_CHECK(cudaFreeHost(h_C));
    return failures == 0 ? 0 : 1;
}
