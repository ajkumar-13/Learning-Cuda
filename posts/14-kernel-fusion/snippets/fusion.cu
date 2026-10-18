// fusion.cu: kernel fusion, measured. Two experiments on one card.
//
// 1. An elementwise chain (bias, scale, ReLU, GELU) over N = 16,777,216 floats, for chains of 2, 3 and 4
//    operations: run as one kernel per operation, each reading the whole array from global memory and writing
//    the whole array back, against one fused kernel that reads x once, keeps the intermediate in a register and
//    writes z once.
// 2. Post 03's tiled matrix multiplication (TILE 16) at 1024 x 1024 x 1024 followed by a ReLU kernel over C,
//    against the same kernel with the ReLU applied in its epilogue, before its one write of C.
//
// It prints the card, each kernel's registers per thread and its theoretical occupancy for 256-thread blocks
// (cudaFuncGetAttributes, cudaOccupancyMaxActiveBlocksPerMultiprocessor), the SM clock measured before and after
// the timed rows (clock64() cycles over CUDA-event time), and the timings, taken after about half a second of
// untimed launches with the rows of each experiment timed in turn within every pass. For the chain: per row the
// fastest of 5 passes, the bytes the version moves to and from global memory, GB/s and the speed-up of fused
// over separate. For the multiplication: per row the fastest of 20 passes, and the time each ReLU adds to the
// multiplication alone in the same pass, as the median over the passes (the card's clock moves under its power
// cap by more than the rows differ). Every fused result is compared element by element with the separate one,
// and both with a host reference (every element of the chain; 4,096 sampled elements of the product).
//
// Build: nvcc -O3 -arch=sm_75 posts/14-kernel-fusion/snippets/fusion.cu -o fusion   (from the series root)
// Run:   ./fusion        (on Windows: .\fusion.exe)
//
// Add -Xptxas=-v to the build line to see the compiler's register count for every kernel.
// The program exits with status 1 if a check fails or a CUDA call fails.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define TILE 16
static const int N = 1 << 24;            // 16,777,216 elements, 64 MiB per array
static const int BLOCK = 256;
static const int MN = 1024;              // the matrix multiplication: A, B, C all [1024, 1024]
static const float BIAS = 0.5f, SCALE = 2.0f;
static const int PASSES = 5;                  // passes per row of the chain
static const int MM_PASSES = 20;              // passes per row of the matrix multiplication
static int g_threadsPerSM = 1024;           // read from the device in main()

// GELU, the tanh approximation, the same code on the host and the device.
__host__ __device__ inline float gelu(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// ---- separate: one kernel per operation; each reads its input array and writes its output array ----
__global__ void addBias(const float* in, float* out, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + b;               // 4 bytes read, 4 bytes written, 1 FLOP
}
__global__ void scale(const float* in, float* out, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * s;
}
__global__ void relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(in[i], 0.0f);
}
__global__ void geluKernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu(in[i]);
}

// ---- fused: one kernel; the intermediate lives and dies in a register ----
template <int OPS>                               // the chain's length, 2 to 4, fixed at compile time
__global__ void fusedChain(const float* x, float* z, float b, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + b;                      // op 1, bias: v is a register, never written to memory
        v = v * s;                               // op 2, scale
        if (OPS >= 3) v = fmaxf(v, 0.0f);        // op 3, ReLU
        if (OPS >= 4) v = gelu(v);               // op 4, GELU
        z[i] = v;                                // one read of x, one write of z
    }
}

// ---- a compute-bound producer: post 03's tiled multiply, with an optional ReLU epilogue ----
template <bool RELU>
__global__ void matmulTiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int n) {
    __shared__ float s_A[TILE][TILE];            // launched with blockDim = (TILE, TILE)
    __shared__ float s_B[TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;
        s_B[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();                         // tiles are loaded
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();                         // done before the next load
    }
    if (row < n && col < n)
        C[row * n + col] = RELU ? fmaxf(sum, 0.0f) : sum;   // the epilogue: ReLU on the register, one write
}

// ---- the SM clock: clock64() counts SM cycles; CUDA events time the same spin in milliseconds ----
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
    return cycles / (ms * 1e3f);                 // cycles per microsecond = MHz
}

static const int GRID = (N + BLOCK - 1) / BLOCK;

// The separate chain: op k reads op k-1's output array and writes its own, x -> t0 -> t1 -> t0 -> z.
static void runSeparate(int ops, const float* d_x, float* d_t0, float* d_t1, float* d_z) {
    const float* in = d_x;
    float* bufs[2] = {d_t0, d_t1};
    for (int k = 0; k < ops; ++k) {
        float* out = (k == ops - 1) ? d_z : bufs[k % 2];
        if (k == 0) addBias<<<GRID, BLOCK>>>(in, out, BIAS, N);
        if (k == 1) scale<<<GRID, BLOCK>>>(in, out, SCALE, N);
        if (k == 2) relu<<<GRID, BLOCK>>>(in, out, N);
        if (k == 3) geluKernel<<<GRID, BLOCK>>>(in, out, N);
        in = out;
    }
}
static void runFused(int ops, const float* d_x, float* d_z) {
    if (ops == 2) fusedChain<2><<<GRID, BLOCK>>>(d_x, d_z, BIAS, SCALE, N);
    if (ops == 3) fusedChain<3><<<GRID, BLOCK>>>(d_x, d_z, BIAS, SCALE, N);
    if (ops == 4) fusedChain<4><<<GRID, BLOCK>>>(d_x, d_z, BIAS, SCALE, N);
}

// One timed run of whatever f launches, in milliseconds. The rows of an experiment are timed in turn inside each
// pass, so a change of clock during the run (the laptop's power cap) falls on every row alike, and each row keeps
// the fastest of its passes.
template <typename F>
static float timeOnce(F f, cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template <typename K>
static void printOccupancy(const char* name, K kernel, int threads) {
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel));
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
    printf("  %-22s %4d  %9d  %8.0f%%\n", name, fa.numRegs, blocks, 100.0 * blocks * threads / g_threadsPerSM);
}

static unsigned int lcg = 12345u;                // a fixed pseudo-random sequence in [-4, 4)
static float nextValue(float range) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((lcg >> 8) / 16777216.0f * 2.0f - 1.0f) * range;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_threadsPerSM = prop.maxThreadsPerMultiProcessor;
    printf("device: %s (compute capability %d.%d), %d SMs, theoretical peak 160 GB/s\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("registers per thread and theoretical occupancy for 256-thread blocks, from the runtime\n");
    printf("  kernel                 regs  blocks/SM  occupancy\n");
    printOccupancy("addBias", addBias, BLOCK);
    printOccupancy("scale", scale, BLOCK);
    printOccupancy("relu", relu, BLOCK);
    printOccupancy("geluKernel", geluKernel, BLOCK);
    printOccupancy("fusedChain<2>", fusedChain<2>, BLOCK);
    printOccupancy("fusedChain<3>", fusedChain<3>, BLOCK);
    printOccupancy("fusedChain<4>", fusedChain<4>, BLOCK);
    printOccupancy("matmulTiled<false>", matmulTiled<false>, TILE * TILE);
    printOccupancy("matmulTiled<true>", matmulTiled<true>, TILE * TILE);

    // ---- data ----
    const size_t bytes = (size_t)N * sizeof(float);
    std::vector<float> h_x(N);
    for (int i = 0; i < N; ++i) h_x[i] = nextValue(4.0f);
    const size_t mbytes = (size_t)MN * MN * sizeof(float);
    std::vector<float> h_A((size_t)MN * MN), h_B((size_t)MN * MN);
    for (auto& v : h_A) v = nextValue(1.0f);
    for (auto& v : h_B) v = nextValue(1.0f);

    float *d_x, *d_t0, *d_t1, *d_zs, *d_zf, *d_A, *d_B, *d_C, *d_Cs, *d_Cf;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));  CUDA_CHECK(cudaMalloc(&d_t0, bytes));
    CUDA_CHECK(cudaMalloc(&d_t1, bytes)); CUDA_CHECK(cudaMalloc(&d_zs, bytes));
    CUDA_CHECK(cudaMalloc(&d_zf, bytes));
    CUDA_CHECK(cudaMalloc(&d_A, mbytes)); CUDA_CHECK(cudaMalloc(&d_B, mbytes));
    CUDA_CHECK(cudaMalloc(&d_C, mbytes)); CUDA_CHECK(cudaMalloc(&d_Cs, mbytes));
    CUDA_CHECK(cudaMalloc(&d_Cf, mbytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), mbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), mbytes, cudaMemcpyHostToDevice));
    dim3 mblock(TILE, TILE), mgrid((MN + TILE - 1) / TILE, (MN + TILE - 1) / TILE);
    const int mgridRelu = (MN * MN + BLOCK - 1) / BLOCK;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---- about half a second of untimed launches, so the card reaches its clock ----
    for (int w = 0; w < 150; ++w) runSeparate(4, d_x, d_t0, d_t1, d_zs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const float mhzBefore = smClockMHz(start, stop);

    // ---- experiment 1: the chain, timed; the outputs of each length are checked below ----
    float sepMs[5], fusMs[5];
    bool chainOk[5];
    double chainErr[5];
    long long chainSame[5];
    std::vector<float> h_zs(N), h_zf(N);
    for (int ops = 2; ops <= 4; ++ops) {
        sepMs[ops] = fusMs[ops] = 1e30f;
        for (int p = 0; p < PASSES; ++p) {
            float sep = timeOnce([&] { runSeparate(ops, d_x, d_t0, d_t1, d_zs); }, start, stop);
            float fus = timeOnce([&] { runFused(ops, d_x, d_zf); }, start, stop);
            sepMs[ops] = std::min(sepMs[ops], sep);
            fusMs[ops] = std::min(fusMs[ops], fus);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_zs.data(), d_zs, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_zf.data(), d_zf, bytes, cudaMemcpyDeviceToHost));
        long long same = 0;
        double err = 0.0;
        for (int i = 0; i < N; ++i) {
            double v = ((double)h_x[i] + BIAS) * SCALE;
            if (ops >= 3) v = std::max(v, 0.0);
            if (ops >= 4) v = 0.5 * v * (1.0 + std::tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
            same += (h_zs[i] == h_zf[i]);
            err = std::max(err, std::fabs(h_zf[i] - v));
        }
        chainSame[ops] = same;
        chainErr[ops] = err;
        chainOk[ops] = (same == N) && (err < 1e-5);
    }

    // ---- experiment 2: a compute-bound producer, timed ----
    // A sustained multiplication drives the laptop card to its power cap, and its clock falls by up to a fifth
    // while the rows differ by a few percent. So every pass times the four rows back to back (in reverse order on
    // odd passes), and the cost of each ReLU is also taken pass by pass, as the difference from the multiplication
    // alone in the same pass, and reported as the median over the passes.
    float t[4][MM_PASSES];                       // 0 matmul alone, 1 ReLU kernel alone, 2 then ReLU kernel, 3 epilogue
    auto row = [&](int r) {
        if (r == 0) return timeOnce([&] { matmulTiled<false><<<mgrid, mblock>>>(d_A, d_B, d_C, MN); }, start, stop);
        if (r == 1) return timeOnce([&] { relu<<<mgridRelu, BLOCK>>>(d_C, d_Cs, MN * MN); }, start, stop);
        if (r == 2) return timeOnce([&] {
            matmulTiled<false><<<mgrid, mblock>>>(d_A, d_B, d_C, MN);   // C written to global memory
            relu<<<mgridRelu, BLOCK>>>(d_C, d_Cs, MN * MN);             // C read back, ReLU(C) written
        }, start, stop);
        return timeOnce([&] { matmulTiled<true><<<mgrid, mblock>>>(d_A, d_B, d_Cf, MN); }, start, stop);
    };
    for (int p = 0; p < MM_PASSES; ++p)
        for (int k = 0; k < 4; ++k) {
            int r = (p % 2 == 0) ? k : 3 - k;
            t[r][p] = row(r);
        }
    float best[4];
    for (int r = 0; r < 4; ++r) best[r] = *std::min_element(t[r], t[r] + MM_PASSES);
    auto medianDiff = [&](int r) {               // median over the passes of (row r - matmul alone)
        std::vector<float> d(MM_PASSES);
        for (int p = 0; p < MM_PASSES; ++p) d[p] = t[r][p] - t[0][p];
        std::sort(d.begin(), d.end());
        return 0.5f * (d[MM_PASSES / 2 - 1] + d[MM_PASSES / 2]);
    };
    const float addSep = medianDiff(2), addFus = medianDiff(3);
    CUDA_CHECK(cudaGetLastError());
    const float mhzAfter = smClockMHz(start, stop);

    std::vector<float> h_Cs((size_t)MN * MN), h_Cf((size_t)MN * MN);
    CUDA_CHECK(cudaMemcpy(h_Cs.data(), d_Cs, mbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Cf.data(), d_Cf, mbytes, cudaMemcpyDeviceToHost));
    long long mmSame = 0;
    for (size_t e = 0; e < (size_t)MN * MN; ++e) mmSame += (h_Cs[e] == h_Cf[e]);
    double mmErr = 0.0;
    for (int s = 0; s < 4096; ++s) {             // a double-precision reference on 4,096 sampled elements
        int r = (int)((s * 2654435761u) % MN), c = (int)((s * 40503u + 17u) % MN);
        double ref = 0.0;
        for (int k = 0; k < MN; ++k) ref += (double)h_A[(size_t)r * MN + k] * h_B[(size_t)k * MN + c];
        ref = std::max(ref, 0.0);
        mmErr = std::max(mmErr, std::fabs(h_Cf[(size_t)r * MN + c] - ref));
    }
    const bool mmOk = (mmSame == (long long)MN * MN) && (mmErr < 1e-3);

    // ---- report ----
    const double mb = bytes / 1e6;               // one array, in MB (10^6 bytes)
    printf("elementwise chain over N = %d floats (%.1f MB per array); the fastest of %d passes\n", N, mb, PASSES);
    printf("  ops  version   launches  traffic (MB)  time (ms)    GB/s  speed-up  check\n");
    for (int ops = 2; ops <= 4; ++ops) {
        double sepMB = 2.0 * ops * mb, fusMB = 2.0 * mb;
        printf("  %3d  separate  %8d  %12.1f  %9.3f  %6.1f\n", ops, ops, sepMB, sepMs[ops],
               sepMB / 1e3 / (sepMs[ops] / 1e3));
        printf("  %3d  fused     %8d  %12.1f  %9.3f  %6.1f  %7.2fx  %s\n", ops, 1, fusMB, fusMs[ops],
               fusMB / 1e3 / (fusMs[ops] / 1e3), sepMs[ops] / fusMs[ops], chainOk[ops] ? "ok" : "MISMATCH");
    }
    printf("  traffic separate / fused = ops; fused results equal separate in %lld, %lld and %lld of %d elements;\n",
           chainSame[2], chainSame[3], chainSame[4], N);
    printf("  largest difference from a double-precision host reference: %.1e, %.1e, %.1e\n",
           chainErr[2], chainErr[3], chainErr[4]);
    printf("tiled matrix multiplication, TILE 16, %d x %d x %d, and ReLU over C (%.1f MB); the fastest of %d passes\n",
           MN, MN, MN, mbytes / 1e6, MM_PASSES);
    printf("  version                            fastest (ms)  median added to matmul alone (ms)\n");
    printf("  matmul alone                       %12.3f\n", best[0]);
    printf("  ReLU kernel alone                  %12.3f\n", best[1]);
    printf("  matmul, then ReLU kernel           %12.3f  %+10.3f\n", best[2], addSep);
    printf("  matmul with ReLU in its epilogue   %12.3f  %+10.3f  %s\n", best[3], addFus, mmOk ? "ok" : "MISMATCH");
    printf("  fused C equals separate C in %lld of %d elements; sampled error %.1e\n", mmSame, MN * MN, mmErr);
    printf("SM clock: %.0f MHz before the timed rows, %.0f MHz after\n", mhzBefore, mhzAfter);

    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_t0)); CUDA_CHECK(cudaFree(d_t1));
    CUDA_CHECK(cudaFree(d_zs)); CUDA_CHECK(cudaFree(d_zf));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_Cs)); CUDA_CHECK(cudaFree(d_Cf));
    const bool ok = chainOk[2] && chainOk[3] && chainOk[4] && mmOk;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
