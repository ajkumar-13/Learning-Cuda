// async_pipeline.cu: post 03's tiled matrix multiply double-buffered with the cp.async pipeline intrinsics, checked and timed
// against the synchronous kernel it upgrades and against a double buffer staged through registers, which needs no cp.async.
//
// Build: nvcc -O3 -arch=sm_80 posts/15-async-copy-pipelining/snippets/async_pipeline.cu -o async_pipeline   (from the series root)
// Run:   ./async_pipeline        (on Windows: .\async_pipeline.exe)
//
// Built for compute capability 8.0 or later (Ampere), __pipeline_memcpy_async compiles to cp.async (SASS LDGSTS),
// __pipeline_commit to cp.async.commit_group and __pipeline_wait_prior(n) to cp.async.wait_group n. Built for an older card,
// cuda_pipeline.h compiles the same calls to an ordinary load into a register followed by a store to shared memory, and
// commit and wait to nothing: the program then checks the pipeline's logic but does not measure cp.async.
// async_pipeline_sm75.cu is this program built with -arch=sm_75. The first lines printed say which path the kernels use.
//
// A: [M, K], B: [K, N], C: [M, N], row-major floats; TILE x TILE threads per block, one output element C[row * N + col] each.
//   matmulTiled         post 03's synchronous loop: load tile t, barrier, compute tile t, barrier
//   matmulPipelined     two buffers per operand; the copy of tile t + 1 is issued before tile t is computed
//   matmulPrefetchRegs  two buffers per operand; tile t + 1 is loaded into registers before tile t is computed and
//                       stored into the other buffer after it: the pre-Ampere form of the same pipeline
// Check: at M = N = K = 1000 (not a multiple of TILE) every kernel's C is compared with the synchronous kernel's, and 4,096
// sampled elements with a double-precision reference. Time: at M = N = K = 1024, each kernel is launched with no extra shared
// memory, then with 40 KB of dynamic shared memory it does not use, which leaves room for one block per SM; each row is the
// fastest of 5 passes of 10 launches, after about half a second of untimed launches, with the SM clock measured before and
// after the timed rows (clock64() cycles over CUDA-event time). The program exits with status 1 if a check or a CUDA call fails.

#include <cuda_runtime.h>
#include <cuda_pipeline.h>            // __pipeline_memcpy_async, __pipeline_commit, __pipeline_wait_prior
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

#define TILE 16

// Post 03's kernel, unchanged: the load of tile t must land before any thread computes with it.
__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[TILE][TILE];                   // launched with blockDim = (TILE, TILE)
    __shared__ float s_B[TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;              // coalesced load of A
        s_A[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;              // coalesced load of B
        s_B[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                // tiles are loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();                                // done before the next load
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Issue this thread's copies of tile t into buffer `buf`: one float of A and one of B. An element outside the matrix is a
// plain zero store, which no outstanding copy targets.
__device__ void issueTile(float (*s_A)[TILE][TILE], float (*s_B)[TILE][TILE], int buf, int t,
                          const float* __restrict__ A, const float* __restrict__ B,
                          int row, int col, int M, int N, int K)
{
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    if (row < M && aCol < K) __pipeline_memcpy_async(&s_A[buf][threadIdx.y][threadIdx.x], &A[row * K + aCol], sizeof(float));
    else                     s_A[buf][threadIdx.y][threadIdx.x] = 0.0f;
    if (bRow < K && col < N) __pipeline_memcpy_async(&s_B[buf][threadIdx.y][threadIdx.x], &B[bRow * N + col], sizeof(float));
    else                     s_B[buf][threadIdx.y][threadIdx.x] = 0.0f;
}

__global__ void matmulPipelined(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[2][TILE][TILE];                // two ping-pong buffers per operand
    __shared__ float s_B[2][TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int numTiles = (K + TILE - 1) / TILE;
    float sum = 0.0f;

    issueTile(s_A, s_B, 0, 0, A, B, row, col, M, N, K); // prologue: tile 0 into buffer 0
    __pipeline_commit();                                // one group per tile

    for (int t = 0; t < numTiles; ++t) {
        int cur = t & 1, next = (t + 1) & 1;
        if (t + 1 < numTiles) {                         // steady state: issue tile t + 1 before computing tile t
            issueTile(s_A, s_B, next, t + 1, A, B, row, col, M, N, K);
            __pipeline_commit();
        }
        __pipeline_wait_prior(t + 1 < numTiles ? 1 : 0);// this thread's copies of tile t have landed
        __syncthreads();                                // and every other thread's

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[cur][threadIdx.y][k] * s_B[cur][k][threadIdx.x];
        __syncthreads();                                // done reading `cur` before tile t + 2 is issued into it
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

__global__ void matmulPrefetchRegs(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[2][TILE][TILE];
    __shared__ float s_B[2][TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int numTiles = (K + TILE - 1) / TILE;
    float sum = 0.0f;

    // prologue: tile 0 into buffer 0
    s_A[0][threadIdx.y][threadIdx.x] = (row < M && threadIdx.x < K) ? A[row * K + threadIdx.x] : 0.0f;
    s_B[0][threadIdx.y][threadIdx.x] = (threadIdx.y < K && col < N) ? B[threadIdx.y * N + col] : 0.0f;
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        int cur = t & 1;
        int aCol = (t + 1) * TILE + threadIdx.x;        // tile t + 1; past the last tile both loads are skipped
        int bRow = (t + 1) * TILE + threadIdx.y;
        float a = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;   // issued now; nothing waits for them yet
        float b = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE; ++k)                  // compute tile t while those loads are in flight
            sum += s_A[cur][threadIdx.y][k] * s_B[cur][k][threadIdx.x];
        s_A[cur ^ 1][threadIdx.y][threadIdx.x] = a;     // first use of a and b: the warp waits here, after the math
        s_B[cur ^ 1][threadIdx.y][threadIdx.x] = b;
        __syncthreads();                                // tile t + 1 is complete, and tile t is no longer read
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

__global__ void archProbe(int* out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;                               // 800 for an sm_80 build, 750 for an sm_75 build
#endif
}

// The SM clock: clock64() counts SM cycles; CUDA events time the same spin in milliseconds.
__global__ void clockSpin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

static float smClockMHz(cudaEvent_t start, cudaEvent_t stop) {
    const long long cycles = 20000000;
    cudaEventRecord(start);
    clockSpin<<<1, 1>>>(cycles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return cycles / (ms * 1e3f);                        // cycles per microsecond = MHz
}

typedef void (*Kernel)(const float*, const float*, float*, int, int, int);
static const Kernel KERNELS[3] = {matmulTiled, matmulPipelined, matmulPrefetchRegs};
static const char* NAMES[3] = {"synchronous (post 03)", "pipelined, __pipeline_*", "pipelined, via registers"};
static const int EXTRA_SMEM = 40 * 1024;                // unused dynamic shared memory: one block per SM
static const int RUNS = 5, ITERS = 10, SAMPLES = 4096;

static void launch(int w, int smem, const float* d_A, const float* d_B, float* d_C, int n)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    KERNELS[w]<<<grid, block, smem>>>(d_A, d_B, d_C, n, n, n);
}

int main()
{
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (compute capability %d.%d), %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    int *d_arch, arch = 0;
    CUDA_CHECK(cudaMalloc(&d_arch, sizeof(int)));
    archProbe<<<1, 1>>>(d_arch);
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("the kernels cannot run on this card (%s): build with -arch=sm_%d%d\n", cudaGetErrorString(e), p.major, p.minor);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(&arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost));
    printf("kernels compiled for compute capability %d.%d: ", arch / 100, (arch % 100) / 10);
    if (arch >= 800)
        printf("__pipeline_memcpy_async is cp.async, an asynchronous copy\n");
    else
        printf("__pipeline_memcpy_async is the software fallback,\n"
               "  a load into a register and a store to shared memory; commit and wait do nothing\n");
    for (int w = 0; w < 3; ++w)
        CUDA_CHECK(cudaFuncSetAttribute(KERNELS[w], cudaFuncAttributeMaxDynamicSharedMemorySize, EXTRA_SMEM));

    const int maxN = 1024;
    size_t maxBytes = (size_t)maxN * maxN * sizeof(float);
    std::vector<float> h_A(maxN * maxN), h_B(maxN * maxN), h_C(maxN * maxN), h_ref(maxN * maxN);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_B, maxBytes));
    CUDA_CHECK(cudaMalloc(&d_C, maxBytes));

    // Check at n = 1000: every kernel against the synchronous one, and the synchronous one against a double-precision reference.
    const int nc = 1000;
    srand(1);
    for (int i = 0; i < nc * nc; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < nc * nc; ++i) h_B[i] = (rand() % 100) / 100.0f;
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)nc * nc * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), (size_t)nc * nc * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    printf("check at M = N = K = %d (not a multiple of TILE = %d):\n", nc, TILE);
    for (int w = 0; w < 3; ++w) {
        CUDA_CHECK(cudaMemset(d_C, 0, (size_t)nc * nc * sizeof(float)));
        launch(w, 0, d_A, d_B, d_C, nc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(w == 0 ? h_ref.data() : h_C.data(), d_C, (size_t)nc * nc * sizeof(float), cudaMemcpyDeviceToHost));
        if (w == 0) {
            double worst = 0.0;
            unsigned s = 12345u;
            for (int q = 0; q < SAMPLES; ++q) {
                s = s * 1664525u + 1013904223u; int r = (int)(s % (unsigned)nc);
                s = s * 1664525u + 1013904223u; int c = (int)(s % (unsigned)nc);
                double ref = 0.0;
                for (int k = 0; k < nc; ++k) ref += (double)h_A[(size_t)r * nc + k] * h_B[(size_t)k * nc + c];
                worst = std::max(worst, fabs(h_ref[(size_t)r * nc + c] - ref) / fmax(fabs(ref), 1e-30));
            }
            bool good = worst < 1e-4;
            ok = ok && good;
            printf("  %-26s largest relative error on %d samples: %.1e  %s\n", NAMES[w], SAMPLES, worst, good ? "ok" : "MISMATCH");
        } else {
            int differ = 0;
            for (int i = 0; i < nc * nc; ++i) differ += (h_C[i] != h_ref[i]);
            ok = ok && differ == 0;
            printf("  %-26s elements that differ from the synchronous kernel: %d of %d  %s\n", NAMES[w], differ, nc * nc,
                   differ ? "MISMATCH" : "ok");
        }
    }

    // Time at n = 1024.
    const int n = 1024;
    for (int i = 0; i < n * n; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < n * n; ++i) h_B[i] = (rand() % 100) / 100.0f;
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), maxBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), maxBytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int r = 0; r < 70; ++r) launch(0, 0, d_A, d_B, d_C, n);    // untimed: about half a second, to reach the clock
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float mhzBefore = smClockMHz(start, stop);
    const int smems[2] = {0, EXTRA_SMEM};
    float best[2][3];
    int blocksPerSM[2][3];
    for (int c = 0; c < 2; ++c)
        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM[c][w], KERNELS[w], TILE * TILE, smems[c]));
            best[c][w] = 1e30f;
        }
    for (int q = 0; q < RUNS; ++q)                      // each pass times every row in turn, so a clock change hits all rows
        for (int c = 0; c < 2; ++c)
            for (int w = 0; w < 3; ++w) {
                CUDA_CHECK(cudaEventRecord(start));
                for (int r = 0; r < ITERS; ++r) launch(w, smems[c], d_A, d_B, d_C, n);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaGetLastError());
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                best[c][w] = std::min(best[c][w], ms / ITERS);
            }
    float mhzAfter = smClockMHz(start, stop);
    printf("time at M = N = K = %d: ms per launch, the fastest of %d passes of %d launches; GFLOP/s = 2 * M * N * K over the time\n",
           n, RUNS, ITERS);
    printf("  kernel                      blocks/SM  warps/SM   time (ms)  GFLOP/s  vs synchronous\n");
    double flop = 2.0 * n * n * n;
    for (int c = 0; c < 2; ++c) {
        printf("  %s\n", c == 0 ? "no extra shared memory:" : "40 KB of extra shared memory, one block per SM:");
        for (int w = 0; w < 3; ++w)
            printf("  %-26s %10d  %8d  %10.3f  %7.0f  %13.2fx\n", NAMES[w], blocksPerSM[c][w], blocksPerSM[c][w] * TILE * TILE / 32,
                   best[c][w], flop / (best[c][w] * 1e6), best[c][0] / best[c][w]);
    }
    printf("SM clock: %.0f MHz before the timed launches, %.0f MHz after\n", mhzBefore, mhzAfter);
    printf("every kernel matches the synchronous kernel: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_arch));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
