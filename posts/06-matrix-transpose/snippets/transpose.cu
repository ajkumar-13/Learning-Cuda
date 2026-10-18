// transpose.cu: transpose a 4096 x 4096 float matrix three ways and time each against a plain copy.
// copyKernel moves A to B unchanged and sets the ceiling. transposeNaive reads rows of A (coalesced)
// and writes columns of B (strided). transposeShared stages a 32 x 32 tile in shared memory so that
// both global accesses are rows, and reads a column of the tile, which hits one bank 32 times.
// transposePadded is the same kernel with one column of padding, [TILE][TILE + 1], which spreads that
// column over all 32 banks. Every kernel runs blocks of 32 x 8 threads; each thread moves 4 elements.
//
// Build: nvcc -O3 -arch=sm_75 posts/06-matrix-transpose/snippets/transpose.cu -o transpose   (from the series root)
// Run:   ./transpose        (on Windows: .\transpose.exe)
//
// Timing. A laptop GPU idles at a low clock, needs a few hundred milliseconds of work to leave it,
// and drops its clock again within a fraction of a second of idling. So the program first runs
// copyKernel untimed until it has kept the GPU busy for one second, and then times the four kernels
// back to back with no host work in between: ROUNDS rounds, each timing ITERS launches of every kernel
// in turn between two CUDA events, after one untimed launch of each. A row's time is the median over
// the rounds of the average launch time. Bandwidth is the bytes read plus the bytes written,
// 2 * 4096 * 4096 * 4, divided by that time, in decimal GB/s.
//
// Checking. After the timing, each kernel runs once more into a cleared B, and B is compared element by
// element with a transpose (or copy) computed on the host. A second, rectangular matrix of 1000 x 700,
// whose sides are not multiples of 32, checks every kernel's edge guards the same way. The program
// exits with status 1 if any check fails.

#include <cuda_runtime.h>
#include <algorithm>   // std::sort
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE 32          // a block's tile is TILE x TILE elements
#define BLOCK_ROWS 8     // a block has TILE x BLOCK_ROWS threads; each moves TILE / BLOCK_ROWS elements
#define ITERS 20
#define ROUNDS 5

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// A: [M, N] row-major input, B: [N, M] output (for the copy, B: [M, N]).

// the ceiling: read a row of A, write the same row of B
__global__ void copyKernel(float* B, const float* A, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < M && col < N)
            B[(row + j) * N + col] = A[(row + j) * N + col];
}

// naive: a warp reads 32 neighbors in a row of A (coalesced) and writes them down a column of B,
// M floats apart (strided)
__global__ void transposeNaive(float* B, const float* A, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < M && col < N)
            B[col * M + (row + j)] = A[(row + j) * N + col];   // strided write
}

// tiled: rows of A into the tile, rows of B out of it; the tile's column read is a 32-way bank conflict
__global__ void transposeShared(float* B, const float* A, int M, int N) {
    __shared__ float tile[TILE][TILE];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < M && col < N)
            tile[threadIdx.y + j][threadIdx.x] = A[(row + j) * N + col];
    __syncthreads();
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < N && col < M)
            B[(row + j) * M + col] = tile[threadIdx.x][threadIdx.y + j];
}

// tiled and padded: the same kernel; a row of the tile is TILE + 1 floats, so a column hits 32 banks
__global__ void transposePadded(float* B, const float* A, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];                 // one column of padding
    int col = blockIdx.x * blockDim.x + threadIdx.x;       // the column of A this thread reads
    int row = blockIdx.y * TILE + threadIdx.y;             // its first row of A; then row + 8, + 16, + 24
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < M && col < N)
            tile[threadIdx.y + j][threadIdx.x] = A[(row + j) * N + col];   // a warp reads a row of A
    __syncthreads();                                       // the whole tile is written before any is read
    col = blockIdx.y * TILE + threadIdx.x;                 // the block indices swap for B
    row = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        if (row + j < N && col < M)
            B[(row + j) * M + col] = tile[threadIdx.x][threadIdx.y + j];   // a warp writes a row of B
}

typedef void (*Kernel)(float*, const float*, int, int);

struct Run { const char* name; Kernel k; bool transposes; };
static const Run RUNS[] = {
    {"copy", copyKernel, false},
    {"naive", transposeNaive, true},
    {"shared", transposeShared, true},
    {"padded", transposePadded, true},
};
static const int NRUNS = 4;

static dim3 gridFor(int M, int N) { return dim3((N + TILE - 1) / TILE, (M + TILE - 1) / TILE); }

// launch k once on A: [M, N] into a cleared B and compare B with the host's transpose (or copy) of A
static bool check(const Run& r, float* d_B, const float* d_A, const std::vector<float>& h, int M, int N) {
    size_t bytes = (size_t)M * N * sizeof(float);
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    r.k<<<gridFor(M, N), dim3(TILE, BLOCK_ROWS)>>>(d_B, d_A, M, N);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> out((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(out.data(), d_B, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float want = h[(size_t)i * N + j];
            float got = r.transposes ? out[(size_t)j * M + i] : out[(size_t)i * N + j];
            if (got != want) return false;
        }
    return true;
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    // theoretical peak DRAM bandwidth (GB/s) = 2 transfers per clock * bus width in bytes * memory clock
    double peak = 2.0 * (p.memoryBusWidth / 8.0) * (p.memoryClockRate * 1e3) / 1e9;
    printf("device: %s (compute capability %d.%d), theoretical peak %.0f GB/s\n", p.name, p.major, p.minor, peak);

    const int M = 4096, N = 4096;
    const size_t bytes = (size_t)M * N * sizeof(float);    // 64 MiB per matrix
    std::vector<float> h((size_t)M * N);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (float)(i % 65521);   // distinct neighbors, exact in float

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h.data(), bytes, cudaMemcpyHostToDevice));
    dim3 block(TILE, BLOCK_ROWS), grid = gridFor(M, N);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // clock warm-up: copies until they have kept the GPU busy for one second
    for (float warm = 0.0f; warm < 1000.0f;) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int w = 0; w < 20; ++w) copyKernel<<<grid, block>>>(d_B, d_A, M, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        warm += ms;
    }

    // timing: the kernels in turn, back to back, ROUNDS times
    float ms[NRUNS][ROUNDS];
    for (int round = 0; round < ROUNDS; ++round)
        for (int r = 0; r < NRUNS; ++r) {
            RUNS[r].k<<<grid, block>>>(d_B, d_A, M, N);    // untimed
            CUDA_CHECK(cudaEventRecord(start));
            for (int it = 0; it < ITERS; ++it) RUNS[r].k<<<grid, block>>>(d_B, d_A, M, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float total = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
            ms[r][round] = total / ITERS;
        }

    printf("A: [%d, %d] floats, 64 MiB; blocks of %d x %d threads, %d elements per thread\n",
           M, N, TILE, BLOCK_ROWS, TILE / BLOCK_ROWS);
    printf("time: median of %d rounds, each the average of %d launches after warm-up\n", ROUNDS, ITERS);
    printf("%-8s %10s %7s %10s  %s\n", "kernel", "time (ms)", "GB/s", "% of copy", "check");
    bool allOk = true;
    double copyGBs = 0.0;
    for (int r = 0; r < NRUNS; ++r) {
        std::sort(ms[r], ms[r] + ROUNDS);
        double t = ms[r][ROUNDS / 2];
        double gbs = 2.0 * bytes / (t * 1e6);              // read plus write
        if (r == 0) copyGBs = gbs;
        bool ok = check(RUNS[r], d_B, d_A, h, M, N);
        printf("%-8s %10.3f %7.1f %9.1f%%  %s\n", RUNS[r].name, t, gbs, 100.0 * gbs / copyGBs,
               ok ? "ok" : "MISMATCH");
        allOk = allOk && ok;
    }

    // the edge guards: a matrix whose sides are not multiples of TILE
    const int m = 1000, n = 700;
    std::vector<float> hr((size_t)m * n);
    for (size_t i = 0; i < hr.size(); ++i) hr[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_A, hr.data(), hr.size() * sizeof(float), cudaMemcpyHostToDevice));
    printf("A: [%d, %d], sides not multiples of %d:", m, n, TILE);
    for (int r = 0; r < NRUNS; ++r) {
        bool ok = check(RUNS[r], d_B, d_A, hr, m, n);
        printf(" %s %s%s", RUNS[r].name, ok ? "ok" : "MISMATCH", r + 1 < NRUNS ? "," : "\n");
        allOk = allOk && ok;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return allOk ? 0 : 1;
}
