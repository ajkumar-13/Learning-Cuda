// sync.cu — the lost-update race and its fix. Many threads each increment one
// shared global counter. The plain `*c += 1` read-modify-write races and loses
// updates; `atomicAdd(c, 1)` makes each increment indivisible and is correct.
//
// Build: nvcc -O3 -arch=sm_75 snippets/sync.cu -o sync   (from the post dir; drop snippets/ if you cd into it)
// Run:   ./sync

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

#define BLOCKS  256
#define THREADS 256
#define ITERS   100                       // increments per thread

// RACY: a non-atomic read-modify-write. Each thread loads *c into a register,
// adds 1, and stores it back; concurrent threads read the same value and
// overwrite each other, so most increments are lost.
__global__ void raceKernel(int* c) {
    for (int k = 0; k < ITERS; ++k)
        *c = *c + 1;                      // three steps (load, add, store): not indivisible
}

// FIXED: atomicAdd performs the whole read-modify-write as one indivisible
// operation, so no update is ever lost (the hardware serializes contenders).
__global__ void atomicKernel(int* c) {
    for (int k = 0; k < ITERS; ++k)
        atomicAdd(c, 1);                  // one indivisible increment
}

int main(void) {
    const int expected = BLOCKS * THREADS * ITERS;   // host reference

    int *d_c;
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(int)));

    // Racy version.
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(int)));
    raceKernel<<<BLOCKS, THREADS>>>(d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int racy = 0;
    CUDA_CHECK(cudaMemcpy(&racy, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    // Fixed (atomic) version.
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(int)));
    atomicKernel<<<BLOCKS, THREADS>>>(d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int atom = 0;
    CUDA_CHECK(cudaMemcpy(&atom, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("expected           = %d\n", expected);
    printf("racy    (plain ++) = %-9d %s\n", racy, racy == expected ? "PASS" : "FAIL (updates lost)");
    printf("atomic  (atomicAdd)= %-9d %s\n", atom, atom == expected ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_c));
    return (atom == expected && racy != expected) ? EXIT_SUCCESS : EXIT_FAILURE;
}
