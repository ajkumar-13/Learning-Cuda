// device_query.cu: print the properties of the GPU that the rest of the series reasons about.
//
// Build: nvcc -O3 -arch=sm_75 posts/01-introduction-to-cuda/snippets/device_query.cu -o device_query   (from the series root)
// Run:   ./device_query        (on Windows: .\device_query.exe)
//
// Every value comes from cudaGetDeviceProperties, except the last line: the theoretical peak
// bandwidth, computed from the reported memory clock and bus width (two transfers per clock).

#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("No CUDA device found.\n");
        return 1;
    }

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        // theoretical peak DRAM bandwidth (GB/s) = 2 * (busWidth / 8 bytes) * memoryClockRate
        double bw = 2.0 * (p.memoryBusWidth / 8.0) * (p.memoryClockRate * 1e3) / 1e9;

        printf("Device %d: %s\n", d, p.name);
        printf("  compute capability   : %d.%d\n", p.major, p.minor);
        printf("  streaming multiproc. : %d SMs\n", p.multiProcessorCount);
        printf("  warp size            : %d threads\n", p.warpSize);
        printf("  max threads / block  : %d\n", p.maxThreadsPerBlock);
        printf("  max threads / SM     : %d\n", p.maxThreadsPerMultiProcessor);
        printf("  registers / SM       : %d (32-bit)\n", p.regsPerMultiprocessor);
        printf("  shared mem / block   : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  L2 cache             : %d KB\n", p.l2CacheSize / 1024);
        printf("  global memory        : %.2f GiB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  peak bandwidth       : %.0f GB/s (theoretical)\n", bw);
    }
    return 0;
}
