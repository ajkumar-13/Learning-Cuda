// occupancy_demo.cu: the kernel post 11 puts under the tools, SAXPY (y = a * x + y) over 16,777,216 floats.
//
// It prints, for block sizes 32 to 1,024:
//   - the theoretical occupancy the CUDA runtime computes for the compiled kernel
//     (cudaOccupancyMaxActiveBlocksPerMultiprocessor: blocks per SM, warps per SM, percent of the 32 warp slots);
//   - the kernel's time, the fastest of 5 launches after about half a second of untimed launches, as GB/s
//     (12 bytes per element: x and y read, y written) and as a share of the 160 GB/s theoretical peak;
// then cudaOccupancyMaxPotentialBlockSize, the SM clock measured before and after the timed rows (clock64()
// cycles over CUDA-event time), and a host check of every element.
//
// Build: nvcc -O3 -arch=sm_75 -Xptxas=-v posts/11-profiling-debugging/snippets/occupancy_demo.cu -o occupancy_demo   (from the series root)
// Run:   ./occupancy_demo            (on Windows: .\occupancy_demo.exe)
//        ./occupancy_demo --check    one launch and the host check only, for compute-sanitizer
//
// -Xptxas=-v makes the compiler print each kernel's registers per thread ("Used 10 registers" for saxpy).
// The program exits with status 1 if the check fails or a CUDA call fails.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

__global__ void saxpy(const float* x, float* y, float a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's element
    if (i < N)                                       // guard the surplus threads
        y[i] = a * x[i] + y[i];                      // one fused multiply-add
}

// The SM clock: clock64() counts SM cycles; CUDA events time the same spin in milliseconds.
__global__ void clockSpin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

float smClockMHz(cudaEvent_t start, cudaEvent_t stop) {
    const long long cycles = 20000000;
    cudaEventRecord(start);
    clockSpin<<<1, 1>>>(cycles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return cycles / (ms * 1e3f);                    // cycles per microsecond = MHz
}

int main(int argc, char** argv) {
    const bool checkOnly = (argc > 1 && strcmp(argv[1], "--check") == 0);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int warpSlots = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    printf("device: %s (compute capability %d.%d), %d SMs, %d warp slots per SM\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, warpSlots);
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, saxpy));
    printf("saxpy: %d registers per thread, %zu bytes of static shared memory, %zu bytes of local memory\n",
           fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);

    const int N = 1 << 24;                           // 16,777,216 elements, 64 MiB per array
    const size_t bytes = (size_t)N * sizeof(float);
    const float a = 2.0f;
    std::vector<float> h_x(N, 1.0f), h_y(N, 2.0f);
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    if (!checkOnly) {
        const int sizes[] = {32, 64, 128, 256, 512, 1024};
        const int NS = sizeof(sizes) / sizeof(sizes[0]);
        int blocksPerSM[NS];
        float bestMs[NS];
        for (int s = 0; s < NS; ++s)
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM[s], saxpy, sizes[s], 0));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (int w = 0; w < 400; ++w) saxpy<<<N / 256, 256>>>(d_x, d_y, a, N);   // untimed: reach the clock
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float mhzBefore = smClockMHz(start, stop);
        for (int s = 0; s < NS; ++s) {
            int grid = (N + sizes[s] - 1) / sizes[s];
            bestMs[s] = 1e30f;
            for (int r = 0; r < 5; ++r) {
                cudaEventRecord(start);
                saxpy<<<grid, sizes[s]>>>(d_x, d_y, a, N);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                bestMs[s] = std::min(bestMs[s], ms);
            }
        }
        float mhzAfter = smClockMHz(start, stop);
        CUDA_CHECK(cudaGetLastError());

        printf("saxpy over N = %d floats; theoretical occupancy from the runtime;\n", N);
        printf("time the fastest of 5 launches; GB/s = 12 bytes per element over the time; peak 160 GB/s\n");
        printf("block  blocks/SM  warps/SM  theoretical  time (ms)    GB/s  %% of peak\n");
        for (int s = 0; s < NS; ++s) {
            int warps = blocksPerSM[s] * sizes[s] / prop.warpSize;
            double gbs = 12.0 * N / (bestMs[s] * 1e6);
            printf("%5d  %9d  %8d  %10.0f%%  %9.3f  %6.1f  %8.0f%%\n", sizes[s], blocksPerSM[s], warps,
                   100.0 * warps / warpSlots, bestMs[s], gbs, 100.0 * gbs / 160.0);
        }
        int minGrid = 0, bestBlock = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &bestBlock, saxpy, 0, 0));
        printf("cudaOccupancyMaxPotentialBlockSize: block %d, at least %d blocks to fill the GPU\n", bestBlock, minGrid);
        printf("SM clock: %.0f MHz before the timed launches, %.0f MHz after\n", mhzBefore, mhzAfter);
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));   // reset y for the check
    }

    // One launch of 256-thread blocks, checked on the host: every y must be 2 * 1 + 2 = 4.
    saxpy<<<(N + 255) / 256, 256>>>(d_x, d_y, a, N);
    CUDA_CHECK(cudaGetLastError());                  // a bad launch configuration
    CUDA_CHECK(cudaDeviceSynchronize());             // a fault inside the kernel
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < N; ++i) wrong += (h_y[i] != 4.0f);
    printf("check: %d of %d elements equal 4.0  %s\n", N - wrong, N, wrong ? "MISMATCH" : "ok");

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    return wrong ? EXIT_FAILURE : EXIT_SUCCESS;
}
