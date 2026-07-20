// async_pipeline.cu — tiled matrix multiply with cp.async double buffering.
//
// Build: nvcc -O3 -arch=sm_80 async_pipeline.cu -o async_pipeline   (from snippets/)
// Run:   ./async_pipeline
//
// This is the Ampere (sm_80+) upgrade of post 03's tiled matmul. Instead of
//
//     load tile -> __syncthreads -> compute tile -> __syncthreads -> repeat
//
// where the compute units WAIT while a tile crosses from global memory, we keep
// TWO shared-memory tile buffers and prefetch tile k+1 with cp.async WHILE the
// compute units chew on tile k. The copy bypasses the register file (global ->
// L2 -> shared, directly), so it costs no registers and runs in the background.
//
// The cp.async programming model used below:
//   __pipeline_memcpy_async(dst, src, bytes)  issue one async global->shared copy
//   __pipeline_commit()                        bundle issued copies into a group
//   __pipeline_wait_prior(n)                   block until <= n groups are in flight
// (these wrap the PTX cp.async / cp.async.commit_group / cp.async.wait_group.)
//
// Self-contained: own main(), CUDA_CHECK, a CPU reference, and a correctness
// check. Requires compute capability 8.0; older GPUs must emulate this with
// manual double buffering (load to registers, then store to shared). The
// committed pipeline_model.py is the VERIFIED latency-hiding artifact; this file
// is the runnable kernel that produces the win the model predicts.

#include <cuda_runtime.h>
#include <cuda_pipeline.h>            // __pipeline_memcpy_async, _commit, _wait_prior
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(err));                                  \
            return 1;                                                         \
        }                                                                     \
    } while (0)

#define TILE 16                       // 16x16 threads, one output element each

// Double-buffered tiled GEMM: C = A * B, with A[M,K], B[K,N], C[M,N].
// s_A / s_B hold TWO tiles each (the ping-pong buffers); `buf` selects which.
__global__ void matmulPipelined(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K)
{
    __shared__ float s_A[2][TILE][TILE];
    __shared__ float s_B[2][TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int numTiles = (K + TILE - 1) / TILE;
    float sum = 0.0f;

    // --- prologue: kick off the async load of tile 0 into buffer 0 ---
    {
        int aCol = 0 * TILE + threadIdx.x;
        int bRow = 0 * TILE + threadIdx.y;
        // each thread copies its one float global->shared, in the background.
        const float* aSrc = (row < M && aCol < K) ? &A[row * K + aCol] : nullptr;
        const float* bSrc = (bRow < K && col < N) ? &B[bRow * N + col] : nullptr;
        if (aSrc) __pipeline_memcpy_async(&s_A[0][threadIdx.y][threadIdx.x], aSrc, sizeof(float));
        else      s_A[0][threadIdx.y][threadIdx.x] = 0.0f;
        if (bSrc) __pipeline_memcpy_async(&s_B[0][threadIdx.y][threadIdx.x], bSrc, sizeof(float));
        else      s_B[0][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __pipeline_commit();              // tile 0's copies form group 0

    for (int t = 0; t < numTiles; ++t) {
        int cur  = t & 1;             // buffer we compute from this iteration
        int next = (t + 1) & 1;       // buffer we prefetch into

        // --- steady state: prefetch tile t+1 into `next` while we will compute t ---
        if (t + 1 < numTiles) {
            int aCol = (t + 1) * TILE + threadIdx.x;
            int bRow = (t + 1) * TILE + threadIdx.y;
            const float* aSrc = (row < M && aCol < K) ? &A[row * K + aCol] : nullptr;
            const float* bSrc = (bRow < K && col < N) ? &B[bRow * N + col] : nullptr;
            if (aSrc) __pipeline_memcpy_async(&s_A[next][threadIdx.y][threadIdx.x], aSrc, sizeof(float));
            else      s_A[next][threadIdx.y][threadIdx.x] = 0.0f;
            if (bSrc) __pipeline_memcpy_async(&s_B[next][threadIdx.y][threadIdx.x], bSrc, sizeof(float));
            else      s_B[next][threadIdx.y][threadIdx.x] = 0.0f;
            __pipeline_commit();      // tile t+1's copies form a new group
        }

        // wait until the CURRENT tile's copy has landed in shared memory. While a
        // prefetch is outstanding (keep = 1) it keeps running across this barrier;
        // on the last tile there is no prefetch, so keep = 0 drains its own copy.
        int keep = (t + 1 < numTiles) ? 1 : 0;
        __pipeline_wait_prior(keep);
        __syncthreads();             // every thread sees the current tile

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[cur][threadIdx.y][k] * s_B[cur][k][threadIdx.x];

        __syncthreads();             // done reading `cur` before it is reused
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

// CPU reference for the correctness check.
static void matmulCPU(const float* A, const float* B, float* C,
                      int M, int N, int K)
{
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
            C[r * N + c] = s;
        }
}

int main()
{
    const int M = 256, N = 256, K = 256;   // not assumed to be a multiple of TILE
    const size_t bA = (size_t)M * K * sizeof(float);
    const size_t bB = (size_t)K * N * sizeof(float);
    const size_t bC = (size_t)M * N * sizeof(float);

    // require Ampere: cp.async is an sm_80 instruction.
    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (prop.major < 8) {
        printf("needs compute capability 8.0+ (Ampere) for cp.async; found %d.%d\n",
               prop.major, prop.minor);
        return 0;
    }

    float* h_A = (float*)malloc(bA);
    float* h_B = (float*)malloc(bB);
    float* h_C = (float*)malloc(bC);
    float* h_ref = (float*)malloc(bC);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)((i % 13) - 6) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)((i % 7)  - 3) * 0.1f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bA));
    CUDA_CHECK(cudaMalloc(&d_B, bB));
    CUDA_CHECK(cudaMalloc(&d_C, bC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmulPipelined<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bC, cudaMemcpyDeviceToHost));
    matmulCPU(h_A, h_B, h_ref, M, N, K);

    double maxErr = 0.0;
    for (int i = 0; i < M * N; ++i)
        maxErr = fmax(maxErr, fabs((double)h_C[i] - (double)h_ref[i]));
    printf("pipelined matmul %dx%dx%d, TILE=%d\n", M, N, K, TILE);
    printf("  max abs error vs CPU = %.3e  (%s)\n",
           maxErr, maxErr < 1e-3 ? "PASS" : "FAIL");

    free(h_A); free(h_B); free(h_C); free(h_ref);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return 0;
}
