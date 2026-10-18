// sync.cu: the synchronization rules of post 09 on a real GPU, in four parts.
//   1. The lost update: 65,536 threads increment one counter 100 times each, with a plain
//      read-modify-write and with atomicAdd, against the host's 6,553,600.
//   2. The warp-synchronous tail of a shared-memory tree reduction, once one warp remains: plain
//      shared memory, volatile shared memory (no barrier), __syncwarp between every read and write,
//      and a warp shuffle. Each block's sum is checked against the host.
//   3. Publishing data across blocks: the Programming Guide's "last block" sum, in which each block
//      stores a partial sum, fences, and counts itself done with an atomic; the last block adds the
//      partial sums. Checked against the host on every launch.
//   4. What a fence costs at each scope: a loop in which every thread stores and then fences, with no
//      fence, __threadfence_block(), __threadfence() and __threadfence_system(); the fastest of 5
//      timed launches each, after 400 untimed ones, reported as time per iteration, with the SM clock
//      measured before and after (clock64() cycles over CUDA-event time).
//
// Build: nvcc -O3 -arch=sm_75 posts/09-synchronization/snippets/sync.cu -o sync   (from the series root)
// Run:   ./sync        (on Windows: .\sync.exe)
//
// Part 1's plain count and part 2's first two rows break rules, so their numbers change from run to
// run, card to card and compiler to compiler; the program exits with status 1 only if a row that keeps
// the rules (atomicAdd, __syncwarp, shuffle, the fenced sum) disagrees with the host or a CUDA call fails.

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
#define ITERS   100                       // increments per thread in part 1

// ---------------------------------------------------------------- part 1: the lost update
// RACY: a plain read-modify-write. `volatile` makes every increment a real load and store, as the
// source says; without it the compiler merges the 100 increments into one load, one add of 100 and
// one store, and the race happens once per thread instead of 100 times.
__global__ void raceKernel(volatile int* c) {
    for (int k = 0; k < ITERS; ++k)
        *c = *c + 1;                      // load, add, store: three steps, not one
}

// FIXED: atomicAdd performs the whole read-modify-write as one indivisible operation.
__global__ void atomicKernel(int* c) {
    for (int k = 0; k < ITERS; ++k)
        atomicAdd(c, 1);                  // one indivisible increment
}

// ---------------------------------------------------------------- part 2: the warp-synchronous tail
// A 256-thread tree reduction in shared memory. The block barrier orders the steps down to 64
// values; the last six steps (32, 16, 8, 4, 2, 1) involve one warp only, and MODE says how the
// lanes of that warp are ordered.
enum { PLAIN = 0, VOLATILE = 1, SYNCWARP = 2, SHUFFLE = 3 };

template <int MODE>
__global__ void blockSum(const int* in, int* out) {
    __shared__ int s[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = in[i];
    __syncthreads();
    for (int stride = THREADS / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();                  // every step's writes before the next step's reads
    }
    if (tid < 32) {
        if (MODE == PLAIN) {              // no barrier: assumes the warp runs in lockstep
            for (int off = 32; off > 0; off >>= 1)
                s[tid] += s[tid + off];
        } else if (MODE == VOLATILE) {    // the pre-2018 idiom: volatile, still no barrier
            volatile int* v = s;
            for (int off = 32; off > 0; off >>= 1)
                v[tid] += v[tid + off];
        } else if (MODE == SYNCWARP) {    // every read before any write, every write before the next read
            int v = s[tid];
            for (int off = 32; off > 0; off >>= 1) {
                v += s[tid + off];
                __syncwarp();
                s[tid] = v;
                __syncwarp();
            }
        } else {                          // SHUFFLE: registers only; the _sync shuffle orders its lanes
            int v = s[tid] + s[tid + 32];
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) s[0] = v;
        }
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ---------------------------------------------------------------- part 3: publish, fence, count
// The Programming Guide's example (section "Memory Fence Functions"): thread 0 of each block stores
// its block's partial sum, fences, and increments a counter; the block that sees gridDim.x - 1 is the
// last one done and adds every partial sum. The fence keeps the counter from being incremented before
// the partial sum is written, so the last block never reads a stale partial sum.
__device__ unsigned int count = 0;

__global__ void sumLastBlock(const int* in, volatile int* partial, int* total) {
    __shared__ int s[THREADS];
    __shared__ bool isLastBlockDone;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = in[i];
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = s[0];                      // publish the data
        __threadfence();                                 // ... before the flag below
        unsigned int value = atomicInc(&count, gridDim.x);   // the flag: one more block done
        isLastBlockDone = (value == gridDim.x - 1);
    }
    __syncthreads();                                     // every thread reads isLastBlockDone
    if (isLastBlockDone) {
        int sum = 0;
        for (int b = tid; b < gridDim.x; b += blockDim.x) sum += partial[b];
        s[tid] = sum;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            *total = s[0];
            count = 0;                                   // ready for the next launch
        }
    }
}

// ---------------------------------------------------------------- part 4: the cost of a scope
// SCOPE 0 is no fence, 1 the block, 2 the device, 3 the system. `volatile` keeps every store.
#define FENCE_ITERS 1000
template <int SCOPE>
__global__ void fenceLoop(volatile int* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < FENCE_ITERS; ++k) {
        x[i] = k;
        if (SCOPE == 1) __threadfence_block();
        else if (SCOPE == 2) __threadfence();
        else if (SCOPE == 3) __threadfence_system();
    }
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

template <int SCOPE>
float fastestOf5(int* d_x, int blocks, cudaEvent_t start, cudaEvent_t stop) {
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(start);
        fenceLoop<SCOPE><<<blocks, THREADS>>>(d_x);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best) best = ms;
    }
    return best;
}

int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    int failures = 0;

    // Part 1.
    const int expected = BLOCKS * THREADS * ITERS;       // host reference
    int *d_c;
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(int)));
    raceKernel<<<BLOCKS, THREADS>>>(d_c);
    CUDA_CHECK(cudaGetLastError());
    int racy = 0;
    CUDA_CHECK(cudaMemcpy(&racy, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(int)));
    atomicKernel<<<BLOCKS, THREADS>>>(d_c);
    CUDA_CHECK(cudaGetLastError());
    int atom = 0;
    CUDA_CHECK(cudaMemcpy(&atom, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("1. one counter, %d threads x %d increments\n", BLOCKS * THREADS, ITERS);
    printf("  expected            = %d\n", expected);
    printf("  plain (*c = *c + 1) = %-9d %s\n", racy, racy == expected ? "matches" : "updates lost");
    printf("  atomicAdd           = %-9d %s\n", atom, atom == expected ? "ok" : "MISMATCH");
    failures += (atom != expected);
    CUDA_CHECK(cudaFree(d_c));

    // Part 2 and part 3 share one input: N ints from 0 to 6.
    const int NB = 16384, N = NB * THREADS, LAUNCHES = 20;
    int* h_in = (int*)malloc(N * sizeof(int));
    int* h_ref = (int*)malloc(NB * sizeof(int));
    int* h_out = (int*)malloc(NB * sizeof(int));
    long long grand = 0;
    for (int b = 0; b < NB; ++b) {
        h_ref[b] = 0;
        for (int t = 0; t < THREADS; ++t) {
            int v = (int)((b * 7919u + t * 104729u) % 7u);
            h_in[b * THREADS + t] = v;
            h_ref[b] += v;
        }
        grand += h_ref[b];
    }
    int *d_in, *d_out, *d_partial, *d_total;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, NB * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial, NB * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("2. the last six steps of a %d-thread block sum, %d blocks x %d launches\n", THREADS, NB, LAUNCHES);
    const char* names[4] = {"plain shared memory, no barrier", "volatile, no barrier",
                            "__syncwarp between read and write", "__shfl_down_sync"};
    for (int mode = 0; mode < 4; ++mode) {
        long long wrong = 0;
        for (int l = 0; l < LAUNCHES; ++l) {
            CUDA_CHECK(cudaMemset(d_out, 0, NB * sizeof(int)));
            if (mode == PLAIN)         blockSum<PLAIN><<<NB, THREADS>>>(d_in, d_out);
            else if (mode == VOLATILE) blockSum<VOLATILE><<<NB, THREADS>>>(d_in, d_out);
            else if (mode == SYNCWARP) blockSum<SYNCWARP><<<NB, THREADS>>>(d_in, d_out);
            else                       blockSum<SHUFFLE><<<NB, THREADS>>>(d_in, d_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NB * sizeof(int), cudaMemcpyDeviceToHost));
            for (int b = 0; b < NB; ++b) wrong += (h_out[b] != h_ref[b]);
        }
        bool keepsRules = (mode == SYNCWARP || mode == SHUFFLE);
        printf("  %-34s wrong block sums: %8lld of %lld  %s\n", names[mode], wrong, (long long)NB * LAUNCHES,
               keepsRules ? (wrong ? "MISMATCH" : "ok") : "(breaks the rule)");
        if (keepsRules) failures += (wrong != 0);
    }

    printf("3. the last-block sum of %d ints in %d blocks, %d launches\n", N, NB, LAUNCHES);
    int wrongTotals = 0, total = 0;
    for (int l = 0; l < LAUNCHES; ++l) {
        CUDA_CHECK(cudaMemset(d_partial, 0, NB * sizeof(int)));
        sumLastBlock<<<NB, THREADS>>>(d_in, d_partial, d_total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
        wrongTotals += (total != grand);
    }
    printf("  expected %lld, last launch %d, wrong totals: %d of %d  %s\n", grand, total, wrongTotals, LAUNCHES,
           wrongTotals ? "MISMATCH" : "ok");
    failures += (wrongTotals != 0);

    // Part 4: four blocks per SM, so every SM is busy.
    int fenceBlocks = 4 * prop.multiProcessorCount, *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, fenceBlocks * THREADS * sizeof(int)));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int w = 0; w < 400; ++w) fenceLoop<1><<<fenceBlocks, THREADS>>>(d_x);   // untimed: reach the clock
    CUDA_CHECK(cudaDeviceSynchronize());
    float mhzBefore = smClockMHz(start, stop);
    float t0 = fastestOf5<0>(d_x, fenceBlocks, start, stop);
    float t1 = fastestOf5<1>(d_x, fenceBlocks, start, stop);
    float t2 = fastestOf5<2>(d_x, fenceBlocks, start, stop);
    float t3 = fastestOf5<3>(d_x, fenceBlocks, start, stop);
    float mhzAfter = smClockMHz(start, stop);
    CUDA_CHECK(cudaGetLastError());
    printf("4. store then fence, %d threads x %d iterations, fastest of 5 launches\n", fenceBlocks * THREADS, FENCE_ITERS);
    printf("  scope                     ns per iteration\n");
    printf("  no fence                  %8.0f\n", t0 * 1e6f / FENCE_ITERS);
    printf("  __threadfence_block()     %8.0f\n", t1 * 1e6f / FENCE_ITERS);
    printf("  __threadfence()           %8.0f\n", t2 * 1e6f / FENCE_ITERS);
    printf("  __threadfence_system()    %8.0f\n", t3 * 1e6f / FENCE_ITERS);
    printf("  SM clock: %.0f MHz before the timed launches, %.0f MHz after\n", mhzBefore, mhzAfter);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial)); CUDA_CHECK(cudaFree(d_total));
    free(h_in); free(h_ref); free(h_out);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
