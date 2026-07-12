// device_query.cu — print the key properties of the GPU you are about to program.
//
// Build: nvcc -O3 -arch=sm_75 snippets/device_query.cu -o device_query   (from the post directory)
// Run:   ./device_query
//
// Every number the rest of this series reasons about (SM count, warp size, shared
// memory per block, peak bandwidth) comes from cudaGetDeviceProperties.

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

        // peak DRAM bandwidth (GB/s) = 2 (DDR) * busWidth/8 * memClock
        double bw = 2.0 * (p.memoryBusWidth / 8.0) * (p.memoryClockRate * 1e3) / 1e9;

        printf("Device %d: %s\n", d, p.name);
        printf("  compute capability   : %d.%d\n", p.major, p.minor);
        printf("  streaming multiproc. : %d SMs\n", p.multiProcessorCount);
        printf("  warp size            : %d threads\n", p.warpSize);
        printf("  max threads / block  : %d\n", p.maxThreadsPerBlock);
        printf("  shared mem / block   : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  global memory        : %.1f GB\n", p.totalGlobalMem / 1e9);
        printf("  peak bandwidth       : %.0f GB/s\n", bw);
    }
    return 0;
}
