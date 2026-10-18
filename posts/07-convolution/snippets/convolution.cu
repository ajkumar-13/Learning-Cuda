// convolution.cu: a K x K box filter over a 1920 x 1080 image, for K = 3 and K = 7, five ways, each
// timed and checked. The GPU kernels apply one fix at a time: a naive kernel that reads the filter
// from global memory; the same kernel with the filter in constant memory; a shared-memory tile with
// a halo of K / 2 pixels; and the separable form of the filter as a row pass and a column pass. A
// plain copy kernel gives the bandwidth ceiling, and the CPU rows give the baseline.
//
// Build: nvcc -O3 -arch=sm_75 -Xcompiler /openmp posts/07-convolution/snippets/convolution.cu -o convolution   (from the series root)
//        (on Linux the OpenMP flag is -Xcompiler -fopenmp; without it the OpenMP row is skipped)
// Run:   ./convolution        (on Windows: .\convolution.exe)
//
// Method: a laptop GPU idles at a low clock and needs a few hundred milliseconds of work to reach its
// boost clock, so the program first runs the copy kernel untimed for WARMUP_MS of wall time. It then
// times every GPU row before any CPU row, because the CPU and the GPU share one cooler and the CPU
// rows heat it. Each GPU row is one untimed launch followed by ROUNDS rounds of ITERS back-to-back
// launches between two CUDA events, and its pass time is the median over the rounds of each round's
// average. The whole set of GPU rows runs PASSES times and each row prints its fastest pass: this
// laptop now and then cuts the GPU's power budget for a second or two, and a row timed inside such
// an episode runs 2 to 8 times slower. Each CPU row is the best of CPU_REPS runs. Every row's output is compared with the
// one-thread CPU result (max absolute error, tolerance 1e-3 on pixel values from 0 to 255). GB/s
// counts the 8 bytes per pixel that any filter must move, one 4-byte read and one 4-byte write. The
// program exits with status 1 if any row fails its check.

#include <cuda_runtime.h>
#include <algorithm>   // std::sort
#include <chrono>
#include <cmath>       // fabsf
#include <cstdio>
#include <cstdlib>     // exit
#include <cstring>     // memcmp
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define TILE 16                    // output tile: 16 x 16 pixels per block
#define MAX_K 7                    // the largest filter the constant arrays hold
#define ITERS 20
#define ROUNDS 7
#define CPU_REPS 3
#define WARMUP_MS 500
#define PASSES 5                   // GPU passes; each row keeps its fastest

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__constant__ float c_mask[MAX_K * MAX_K];   // file scope, read-only, broadcast to a warp
__constant__ float c_row[MAX_K];            // the separable filter's two 1D factors
__constant__ float c_col[MAX_K];

// the bandwidth ceiling: read one float, write one float, per pixel
__global__ void copyKernel(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) out[row * W + col] = in[row * W + col];
}

// naive: the filter is read from global memory, and neighbors re-read overlapping pixels
template <int K>
__global__ void convNaive(const float* in, const float* mask, float* out, int W, int H) {
    constexpr int R = K / 2;                     // the filter radius
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j) {
                int r = row + i - R, c = col + j - R;
                if (r >= 0 && r < H && c >= 0 && c < W)
                    sum += in[r * W + c] * mask[i * K + j];
            }
        out[row * W + col] = sum;
    }
}

// constant memory: the same kernel, with the filter read from c_mask
template <int K>
__global__ void convConst(const float* in, float* out, int W, int H) {
    constexpr int R = K / 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j) {
                int r = row + i - R, c = col + j - R;
                if (r >= 0 && r < H && c >= 0 && c < W)
                    sum += in[r * W + c] * c_mask[i * K + j];   // one address for the whole warp
            }
        out[row * W + col] = sum;
    }
}

// tiled: BLK x BLK threads load an (output + halo) tile into shared memory once;
// only the inner TILE x TILE threads compute, reading neighbors from shared memory.
template <int K>
__global__ void convTiled(const float* in, float* out, int W, int H) {
    constexpr int R = K / 2, BLK = TILE + K - 1;     // BLK = 18 for a 3 x 3 filter
    __shared__ float s[BLK][BLK];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx - R;            // shifted back by the halo radius
    int row = blockIdx.y * TILE + ty - R;
    s[ty][tx] = (row >= 0 && row < H && col >= 0 && col < W) ? in[row * W + col] : 0.0f;
    __syncthreads();                                 // the whole tile is loaded before any thread reads it

    if (tx >= R && tx < BLK - R && ty >= R && ty < BLK - R) {   // inner threads only
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < K; ++i)
            #pragma unroll
            for (int j = 0; j < K; ++j)
                sum += s[ty + i - R][tx + j - R] * c_mask[i * K + j];
        int oc = blockIdx.x * TILE + (tx - R), orow = blockIdx.y * TILE + (ty - R);
        if (orow < H && oc < W) out[orow * W + oc] = sum;
    }
}

// separable, pass 1: each thread sums K pixels of its row into an intermediate image
template <int K>
__global__ void convRows(const float* in, float* tmp, int W, int H) {
    constexpr int R = K / 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            int c = col + j - R;
            if (c >= 0 && c < W) sum += in[row * W + c] * c_row[j];
        }
        tmp[row * W + col] = sum;
    }
}

// separable, pass 2: each thread sums K pixels of its column of the intermediate image
template <int K>
__global__ void convCols(const float* tmp, float* out, int W, int H) {
    constexpr int R = K / 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            int r = row + i - R;
            if (r >= 0 && r < H) sum += tmp[r * W + col] * c_col[i];
        }
        out[row * W + col] = sum;
    }
}

// the host reference: the same zero-padded sum, one output pixel per iteration
template <int K>
static void convCpu(const float* in, const float* mask, float* out, int W, int H, bool parallel) {
    constexpr int R = K / 2;
    #pragma omp parallel for if (parallel)
    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < K; ++j) {
                    int r = row + i - R, c = col + j - R;
                    if (r >= 0 && r < H && c >= 0 && c < W)
                        sum += in[r * W + c] * mask[i * K + j];
                }
            out[row * W + col] = sum;
        }
}

template <int K>
static double cpuBest(const float* in, const float* mask, float* out, int W, int H, bool parallel) {
    double best = 1e30;
    for (int rep = 0; rep < CPU_REPS; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        convCpu<K>(in, mask, out, W, H, parallel);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

static float maxAbsErr(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.0f;
    for (size_t k = 0; k < a.size(); ++k) {
        float d = fabsf(a[k] - b[k]);
        if (!(d <= e)) e = d;                   // a NaN (an unwritten pixel) sticks
    }
    return e;
}

enum Method { COPY, NAIVE, CONST, TILED, SEPARABLE };

template <int K>
static void launch(Method m, const float* d_in, const float* d_mask, float* d_tmp, float* d_out,
                   int W, int H) {
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);   // grid sized by OUTPUT tiles
    switch (m) {
    case COPY:  copyKernel<<<grid, block>>>(d_in, d_out, W, H); break;
    case NAIVE: convNaive<K><<<grid, block>>>(d_in, d_mask, d_out, W, H); break;
    case CONST: convConst<K><<<grid, block>>>(d_in, d_out, W, H); break;
    case TILED: convTiled<K><<<grid, dim3(TILE + K - 1, TILE + K - 1)>>>(d_in, d_out, W, H); break;
    case SEPARABLE:
        convRows<K><<<grid, block>>>(d_in, d_tmp, W, H);
        convCols<K><<<grid, block>>>(d_tmp, d_out, W, H);
        break;
    }
}

struct Row { const char* name; double ms; float err; bool ok; };

// the GPU rows of one filter size, checked against the one-thread CPU result ref
template <int K>
static void gpuRows(const std::vector<float>& h_in, const std::vector<float>& ref, const float* d_in,
                    float* d_mask, float* d_tmp, float* d_out, int W, int H, Row* rows) {
    const size_t n = (size_t)W * H, bytes = n * sizeof(float);
    std::vector<float> box(K * K, 1.0f / (K * K)), factor(K, 1.0f / K), out(n);
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, box.data(), K * K * sizeof(float)));   // not cudaMemcpy
    CUDA_CHECK(cudaMemcpyToSymbol(c_row, factor.data(), K * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_col, factor.data(), K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mask, box.data(), K * K * sizeof(float), cudaMemcpyHostToDevice));
    const char* names[5] = {"copy (ceiling)", "naive, global filter", "constant filter",
                            "tiled + halo", "separable, 2 passes"};
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int m = COPY; m <= SEPARABLE; ++m) {
        CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));   // NaN everywhere: an unwritten pixel fails
        launch<K>((Method)m, d_in, d_mask, d_tmp, d_out, W, H);   // untimed warm-up
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> rounds(ROUNDS);
        for (int rd = 0; rd < ROUNDS; ++rd) {
            CUDA_CHECK(cudaEventRecord(start));
            for (int it = 0; it < ITERS; ++it) launch<K>((Method)m, d_in, d_mask, d_tmp, d_out, W, H);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float total = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
            rounds[rd] = total / ITERS;
        }
        std::sort(rounds.begin(), rounds.end());
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        Row r = {names[m], rounds[ROUNDS / 2], 0.0f, false};
        if (m == COPY) r.ok = memcmp(out.data(), h_in.data(), bytes) == 0;
        else { r.err = maxAbsErr(out, ref); r.ok = r.err <= 1e-3f; }
        rows[m] = r;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// keep each row's fastest pass; a row fails if any pass failed, and reports the largest error
static void keepFastest(Row* best, const Row* now, bool first) {
    for (int m = COPY; m <= SEPARABLE; ++m) {
        bool ok = now[m].ok && (first || best[m].ok);
        float e = now[m].err;
        if (!first && e == e && !(best[m].err <= e)) e = best[m].err;   // the larger error; a NaN sticks
        if (first || now[m].ms < best[m].ms) best[m] = now[m];
        best[m].ok = ok;
        best[m].err = e;
    }
}

// one table: the CPU rows (timed now, after the GPU rows) and the GPU rows measured before
template <int K>
static bool printTable(const std::vector<float>& h_in, const std::vector<float>& ref, int W, int H,
                       const Row* rows) {
    const size_t n = (size_t)W * H, bytes = n * sizeof(float);
    std::vector<float> box(K * K, 1.0f / (K * K)), out(n);
    bool allOk = true;
    printf("%d x %d box filter, image %d x %d, %zu bytes moved per pass\n", K, K, W, H, 2 * bytes);
    printf("method                   time (ms)    GB/s   vs 1 CPU thread   max abs err\n");
    double cpu1 = cpuBest<K>(h_in.data(), box.data(), out.data(), W, H, false);
    printf("%-24s %9.3f %7.1f   %14.1fx   reference\n", "CPU, 1 thread", cpu1,
           2 * bytes / (cpu1 * 1e6), 1.0);
#ifdef _OPENMP
    double cpuN = cpuBest<K>(h_in.data(), box.data(), out.data(), W, H, true);
    char ompName[32];
    snprintf(ompName, sizeof(ompName), "CPU, OpenMP %d threads", omp_get_max_threads());
    float eN = maxAbsErr(out, ref);
    bool okN = eN <= 1e-3f;
    printf("%-24s %9.3f %7.1f   %14.1fx   %.1e %s\n", ompName, cpuN, 2 * bytes / (cpuN * 1e6),
           cpu1 / cpuN, eN, okN ? "ok" : "MISMATCH");
    allOk = allOk && okN;
#endif
    for (int m = COPY; m <= SEPARABLE; ++m) {
        const Row& r = rows[m];
        if (m == COPY)
            printf("%-24s %9.3f %7.1f   %15s   %s\n", r.name, r.ms, 2 * bytes / (r.ms * 1e6), "-",
                   r.ok ? "exact copy" : "MISMATCH");
        else
            printf("%-24s %9.3f %7.1f   %14.1fx   %.1e %s\n", r.name, r.ms, 2 * bytes / (r.ms * 1e6),
                   cpu1 / r.ms, r.err, r.ok ? "ok" : "MISMATCH");
        allOk = allOk && r.ok;
    }
    return allOk;
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    printf("zero padding; GPU rows: fastest of %d passes, each the median of %d rounds of %d launches; "
           "CPU rows: best of %d\n", PASSES, ROUNDS, ITERS, CPU_REPS);

    const int W = 1920, H = 1080;
    const size_t n = (size_t)W * H, bytes = n * sizeof(float);
    std::vector<float> h_in(n), ref3(n), ref7(n);
    unsigned int x = 12345u;                     // pixel values 0 to 255 from a fixed generator
    for (size_t k = 0; k < n; ++k) { x = 1664525u * x + 1013904223u; h_in[k] = (float)(x >> 24); }
    std::vector<float> box3(9, 1.0f / 9), box7(49, 1.0f / 49);
    convCpu<3>(h_in.data(), box3.data(), ref3.data(), W, H, false);   // the references, untimed
    convCpu<7>(h_in.data(), box7.data(), ref7.data(), W, H, false);

    float *d_in, *d_mask, *d_tmp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, MAX_K * MAX_K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // clock warm-up, then every GPU row before any CPU timing, so the CPU's heat does not slow the GPU rows
    auto w0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < WARMUP_MS) {
        for (int w = 0; w < 100; ++w) launch<3>(COPY, d_in, d_mask, d_tmp, d_out, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaGetLastError());
    Row rows3[5], rows7[5], pass3[5], pass7[5];
    for (int pass = 0; pass < PASSES; ++pass) {     // each row keeps its fastest pass
        gpuRows<3>(h_in, ref3, d_in, d_mask, d_tmp, d_out, W, H, pass3);
        gpuRows<7>(h_in, ref7, d_in, d_mask, d_tmp, d_out, W, H, pass7);
        keepFastest(rows3, pass3, pass == 0);
        keepFastest(rows7, pass7, pass == 0);
    }

    bool ok3 = printTable<3>(h_in, ref3, W, H, rows3);
    bool ok7 = printTable<7>(h_in, ref7, W, H, rows7);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_mask));
    return (ok3 && ok7) ? 0 : 1;
}
