// occupancy_demo.cu — a tiny, real kernel you can put under the profilers.
//
// This program exists to be *measured*, not to be fast. It runs a SAXPY-style
// vector operation y = a*x + y over a large array, verifies the result on the
// host, and prints a checksum so the compiler cannot delete the work. The
// interesting part is what the tools say about it.
//
// Build (and print registers + shared memory per thread/block):
//   nvcc -O3 -arch=sm_75 -Xptxas=-v occupancy_demo.cu -o occupancy_demo
//
// Check correctness (CUDA 12 tool, replaces cuda-memcheck):
//   compute-sanitizer ./occupancy_demo
//   compute-sanitizer --tool memcheck   ./occupancy_demo   # out-of-bounds / leaks
//   compute-sanitizer --tool racecheck  ./occupancy_demo   # shared-memory races
//
// Profile the timeline, then the kernel in detail:
//   nsys profile -o occ_timeline ./occupancy_demo          # Nsight Systems
//   ncu --set full -o occ_kernel ./occupancy_demo          # Nsight Compute
//
// Run: ./occupancy_demo

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Wrap every CUDA call. A launch reports errors asynchronously, so an unchecked
// failure surfaces later in some unrelated call; CUDA_CHECK makes it local.
#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(err));                                  \
            return 1;                                                         \
        }                                                                     \
    } while (0)

// __launch_bounds__(maxThreadsPerBlock) tells the compiler the largest block we
// will launch, so it can cap registers per thread to keep occupancy up. Try
// commenting it out and re-running -Xptxas=-v to watch the register count move.
__global__ void __launch_bounds__(256)
saxpy(const float *x, float *y, float a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        y[i] = a * x[i] + y[i];                      // one fused multiply-add
}

int main()
{
    const int N = 1 << 24;                  // 16,777,216 elements (~64 MiB per array)
    const size_t bytes = (size_t)N * sizeof(float);
    const float a = 2.0f;

    float *h_x = new float[N], *h_y = new float[N];
    for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }   // expect 4.0

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;                                  // 8 warps per block
    const int blocksPerGrid = (N + blockSize - 1) / blockSize;  // ceiling division

    // A warm-up launch: the first kernel pays one-time context and JIT costs, so
    // profilers should measure the second. Run untimed, then the real one.
    saxpy<<<blocksPerGrid, blockSize>>>(d_x, d_y, a, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset y and run the measured launch (this is the one the profiler reads).
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    saxpy<<<blocksPerGrid, blockSize>>>(d_x, d_y, a, N);
    CUDA_CHECK(cudaGetLastError());          // catch a bad launch configuration
    CUDA_CHECK(cudaDeviceSynchronize());     // wait, then catch a runtime fault

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // Verify on the host and checksum so the compiler keeps the work.
    double checksum = 0.0;
    int wrong = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_y[i] - 4.0f) > 1e-5f) ++wrong;
        checksum += h_y[i];
    }
    printf("N           = %d\n", N);
    printf("y[0]        = %.1f (expect 4.0)\n", h_y[0]);
    printf("mismatches  = %d\n", wrong);
    printf("checksum    = %.1f (expect %.1f)\n", checksum, 4.0 * N);

    delete[] h_x; delete[] h_y;
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    return wrong == 0 ? 0 : 1;
}
