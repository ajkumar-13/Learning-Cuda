// shared_race.cu: two shared-memory bugs from post 09, for compute-sanitizer's racecheck and synccheck.
//
// A 256-thread block sums its 256 values in shared memory: __syncthreads() between the steps down to 64
// values, then the last six steps in one warp. Three versions, chosen on the command line:
//   (default)            the tail with __syncwarp() between every read and every write, as post 09 fixes it
//   --race               the plain tail of post 09, no barrier: lanes read values other lanes are writing
//   --divergent-barrier  the whole tree with its __syncthreads() inside `if (tid < stride)`: from stride 128
//                        down, the threads with tid < stride wait there while the others wait at the final barrier
// 64 blocks; every block's sum is checked against the host.
//
// Build: nvcc -O3 -arch=sm_75 -lineinfo posts/11-profiling-debugging/snippets/shared_race.cu -o shared_race   (from the series root)
// Run:   ./shared_race                                                        (correct; exits 0)
//        compute-sanitizer --tool racecheck ./shared_race --race
//        compute-sanitizer --tool synccheck ./shared_race --divergent-barrier
//        (on Windows: .\shared_race.exe)
//
// The default version exits with status 1 if a block sum is wrong or a CUDA call fails; the two buggy versions
// print their count of wrong sums and exit 0, since a rule violation need not change the number on every card.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

#define THREADS 256
#define BLOCKS  64
enum { SYNCWARP = 0, RACE = 1, DIVERGENT = 2 };

template <int MODE>
__global__ void blockSum(const int* in, int* out) {
    __shared__ int s[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = in[i];
    __syncthreads();
    for (int stride = THREADS / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        if (MODE == RACE) {                       // no barrier: assumes the warp runs in lockstep
            for (int off = 32; off > 0; off >>= 1)
                s[tid] += s[tid + off];
        } else {                                  // every read before any write, every write before the next read
            int v = s[tid];
            for (int off = 32; off > 0; off >>= 1) {
                v += s[tid + off];
                __syncwarp();
                s[tid] = v;
                __syncwarp();
            }
        }
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// The whole tree with its barrier inside the branch: WRONG.
__global__ void blockSumDivergent(const int* in, int* out) {
    __shared__ int s[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = in[i];
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
            __syncthreads();                      // reached only by the threads with tid < stride
        }
    }
    __syncthreads();                              // every thread waits here before thread 0 writes the sum
    if (tid == 0) out[blockIdx.x] = s[0];
}

int main(int argc, char** argv) {
    int mode = SYNCWARP;
    if (argc > 1 && strcmp(argv[1], "--race") == 0) mode = RACE;
    if (argc > 1 && strcmp(argv[1], "--divergent-barrier") == 0) mode = DIVERGENT;
    const int N = BLOCKS * THREADS;
    int h_in[N], h_ref[BLOCKS], h_out[BLOCKS];
    for (int b = 0; b < BLOCKS; ++b) {
        h_ref[b] = 0;
        for (int t = 0; t < THREADS; ++t) {
            int v = (int)((b * 7919u + t * 104729u) % 7u);
            h_in[b * THREADS + t] = v;
            h_ref[b] += v;
        }
    }
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(h_in)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));
    if (mode == RACE)           blockSum<RACE><<<BLOCKS, THREADS>>>(d_in, d_out);
    else if (mode == DIVERGENT) blockSumDivergent<<<BLOCKS, THREADS>>>(d_in, d_out);
    else                        blockSum<SYNCWARP><<<BLOCKS, THREADS>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    int wrong = 0;
    for (int b = 0; b < BLOCKS; ++b) wrong += (h_out[b] != h_ref[b]);
    const char* names[3] = {"__syncwarp between read and write", "plain tail, no barrier",
                            "__syncthreads inside if (tid < stride)"};
    printf("%s: wrong block sums %d of %d  %s\n", names[mode], wrong, BLOCKS,
           mode == SYNCWARP ? (wrong ? "MISMATCH" : "ok") : "(breaks the rule)");
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    return (mode == SYNCWARP && wrong) ? EXIT_FAILURE : EXIT_SUCCESS;
}
