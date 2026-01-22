/**
 * device_query.cu - Query GPU device properties
 * 
 * This program demonstrates how to:
 * - Detect available CUDA devices
 * - Query device properties (SMs, memory, compute capability, etc.)
 * - Understand your GPU's capabilities
 * 
 * Compile: nvcc -o device_query device_query.cu
 * Run:     ./device_query
 */

#include <cstdio>

int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("========== Device %d: %s ==========\n", dev, prop.name);
        printf("\n--- Compute Capability ---\n");
        printf("  Compute Capability:     %d.%d\n", prop.major, prop.minor);
        
        printf("\n--- Multiprocessors ---\n");
        printf("  SM Count:               %d\n", prop.multiProcessorCount);
        printf("  Max Threads per SM:     %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block:  %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size:              %d\n", prop.warpSize);
        
        printf("\n--- Block Dimensions ---\n");
        printf("  Max Block Size:         (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size:          (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        printf("\n--- Memory ---\n");
        printf("  Global Memory:          %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per SM:   %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per Block:    %d\n", prop.regsPerBlock);
        printf("  Registers per SM:       %d\n", prop.regsPerMultiprocessor);
        printf("  L2 Cache Size:          %d KB\n", prop.l2CacheSize / 1024);
        printf("  Memory Bus Width:       %d bits\n", prop.memoryBusWidth);
        printf("  Memory Clock Rate:      %.2f GHz\n", prop.memoryClockRate / 1e6);
        
        // Calculate theoretical memory bandwidth
        double bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
        printf("  Peak Memory Bandwidth:  %.2f GB/s\n", bandwidth);
        
        printf("\n--- Features ---\n");
        printf("  Concurrent Kernels:     %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Async Engine Count:     %d\n", prop.asyncEngineCount);
        printf("  Unified Addressing:     %s\n", prop.unifiedAddressing ? "Yes" : "No");
        
        printf("\n");
    }
    
    return 0;
}
