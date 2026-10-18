// scan.cu: prefix sums (scans) on the GPU. It first runs the two block-level scans on the post's eight
// values in one block, Hillis-Steele (inclusive, N log N additions) and Blelloch (exclusive, up-sweep then
// down-sweep, 2(N - 1) additions). It then scans N = 16,777,216 floats five ways, each built from one
// block-level kernel by the three-phase scheme (scan each block and save its total, scan the totals, add
// each block's offset back), and times each against a plain copy of the same bytes:
//   Hillis-Steele in shared memory, 1,024 elements per block;
//   Blelloch with plain indices, 2,048 elements per block;
//   Blelloch with the conflict-free offset index + (index >> 5), 2,048 elements per block;
//   a warp-shuffle scan (Hillis-Steele inside each warp with __shfl_up_sync), 1,024 elements per block;
//   cub::DeviceScan::ExclusiveSum, NVIDIA's library scan.
//
// Build: nvcc -O3 -arch=sm_75 posts/08-parallel-scan/snippets/scan.cu -o scan   (from the series root)
// Run:   ./scan        (on Windows: .\scan.exe)
//
// A laptop GPU idles at a low clock, so the program first runs the Hillis-Steele scan 100 times untimed.
// Each row is then one untimed scan followed by 20 timed scans between two CUDA events, averaged. The input
// is 0s and 1s stored as floats, so every running total is an integer below 2^24 and exact in a float: each
// row's output is compared element by element with a prefix sum computed on the host. Bandwidth is the 8 * N
// bytes a scan must move at least (read N floats, write N floats), divided by the average time.

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BLOCK 1024                   // threads per block in the timed scans
#define WARP 32

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Hillis-Steele inclusive scan of one block's elements, one per thread: at each step every element adds
// the element `stride` places back. Two halves of buf take turns as input and output, so no thread
// overwrites a value another thread has still to read in the same step.
__global__ void hillisSteele(const float* in, float* out, float* sums, int n) {
    extern __shared__ float buf[];   // 2 * blockDim.x floats: two ping-pong halves
    int m = blockDim.x, t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pin = 1, pout = 0;           // the first swap makes half 0, which holds the input, the one read
    buf[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < m; stride *= 2) {
        pout = 1 - pout; pin = 1 - pin;
        buf[pout * m + t] = buf[pin * m + t] + (t >= stride ? buf[pin * m + t - stride] : 0.0f);
        __syncthreads();
    }
    if (i < n) out[i] = buf[pout * m + t];                   // inclusive scan
    if (sums && t == m - 1) sums[blockIdx.x] = buf[pout * m + t];   // the block's total
}

// The shared-memory index of element k: plain, or with one slot skipped after every 32 (32 banks).
template <bool PAD>
__device__ __forceinline__ int at(int k) { return PAD ? k + (k >> 5) : k; }

// Blelloch exclusive scan of one block's 2 * blockDim.x elements, two per thread: the up-sweep builds
// partial sums (a reduction tree), the root is cleared, and the down-sweep distributes them.
template <bool PAD>
__global__ void blelloch(const float* in, float* out, float* sums, int n) {
    extern __shared__ float s[];
    int t = threadIdx.x, m = 2 * blockDim.x, offset = 1;
    int base = blockIdx.x * m;
    int a = t, b = t + m / 2;                         // one element from each half of the tile
    s[at<PAD>(a)] = (base + a < n) ? in[base + a] : 0.0f;
    s[at<PAD>(b)] = (base + b < n) ? in[base + b] : 0.0f;

    for (int d = m >> 1; d > 0; d >>= 1) {            // up-sweep
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            s[at<PAD>(bi)] += s[at<PAD>(ai)];
        }
        offset *= 2;
    }
    if (t == 0) {
        if (sums) sums[blockIdx.x] = s[at<PAD>(m - 1)];   // the block's total
        s[at<PAD>(m - 1)] = 0.0f;                     // clear the root for an exclusive scan
    }
    for (int d = 1; d < m; d *= 2) {                  // down-sweep
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            float tmp = s[at<PAD>(ai)];               // save the left child
            s[at<PAD>(ai)] = s[at<PAD>(bi)];          // left <- parent
            s[at<PAD>(bi)] += tmp;                    // right <- parent + saved left
        }
    }
    __syncthreads();
    if (base + a < n) out[base + a] = s[at<PAD>(a)];
    if (base + b < n) out[base + b] = s[at<PAD>(b)];
}

// Hillis-Steele inside one warp, in registers: lane l adds the value of lane l - off, for off = 1, 2, 4,
// 8, 16. Every lane of the warp must call it (the full mask).
__device__ __forceinline__ float warpInclusiveScan(float v) {
    int lane = threadIdx.x % WARP;
    for (int off = 1; off < WARP; off *= 2) {
        float up = __shfl_up_sync(0xffffffff, v, off);   // the value of lane - off
        if (lane >= off) v += up;
    }
    return v;
}

// Inclusive scan of one block's elements with warp shuffles: each warp scans its 32 values, warp 0 scans
// the 32 warp totals, and each thread adds the totals of the warps before its own. Needs blockDim.x = 1024.
__global__ void warpScan(const float* in, float* out, float* sums, int n) {
    __shared__ float warpSums[WARP];
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = warpInclusiveScan((i < n) ? in[i] : 0.0f);
    if (lane == WARP - 1) warpSums[wid] = v;          // each warp's total
    __syncthreads();
    if (wid == 0) warpSums[lane] = warpInclusiveScan(warpSums[lane]);
    __syncthreads();
    if (wid > 0) v += warpSums[wid - 1];
    if (i < n) out[i] = v;
    if (sums && threadIdx.x == blockDim.x - 1) sums[blockIdx.x] = v;
}

// Phase 3: add to every element of block b the scanned total of the blocks before it. An inclusive scan
// of the totals holds that offset at b - 1, an exclusive one at b.
__global__ void addOffsets(float* out, const float* scanned, int n, int per, bool inclusive) {
    int b = blockIdx.x;
    float off = inclusive ? (b > 0 ? scanned[b - 1] : 0.0f) : scanned[b];
    for (int k = threadIdx.x; k < per; k += blockDim.x) {
        int idx = b * per + k;
        if (idx < n) out[idx] += off;
    }
}

__global__ void copyKernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

enum Algo { HILLIS_STEELE, BLELLOCH, BLELLOCH_PADDED, WARP_SHUFFLE };

int perBlock(Algo a) { return (a == BLELLOCH || a == BLELLOCH_PADDED) ? 2 * BLOCK : BLOCK; }
bool isInclusive(Algo a) { return a == HILLIS_STEELE || a == WARP_SHUFFLE; }

// The three-phase scan of n elements: (1) scan each block and save its total, (2) scan the totals with the
// same algorithm, recursively, (3) add each block's offset. The workspace is allocated once, by the caller.
void scanDevice(Algo a, const float* d_in, float* d_out, int n, float* const* work, int level) {
    int per = perBlock(a), blocks = (n + per - 1) / per;
    float* sums = (blocks > 1) ? work[2 * level] : nullptr;
    switch (a) {
    case HILLIS_STEELE:   hillisSteele<<<blocks, BLOCK, 2 * BLOCK * sizeof(float)>>>(d_in, d_out, sums, n); break;
    case BLELLOCH:        blelloch<false><<<blocks, BLOCK, 2 * BLOCK * sizeof(float)>>>(d_in, d_out, sums, n); break;
    case BLELLOCH_PADDED: blelloch<true><<<blocks, BLOCK, (2 * BLOCK + 2 * BLOCK / 32) * sizeof(float)>>>(d_in, d_out, sums, n); break;
    case WARP_SHUFFLE:    warpScan<<<blocks, BLOCK>>>(d_in, d_out, sums, n); break;
    }
    if (blocks > 1) {
        float* scanned = work[2 * level + 1];
        scanDevice(a, sums, scanned, blocks, work, level + 1);
        addOffsets<<<blocks, BLOCK>>>(d_out, scanned, n, per, isInclusive(a));
    }
}

int levels(Algo a, int n) { int k = 1; while (n > perBlock(a)) { n = (n + perBlock(a) - 1) / perBlock(a); ++k; } return k; }

template <class F>
float averageMs(F scan, cudaEvent_t start, cudaEvent_t stop, int iters) {
    scan();                                           // one untimed scan
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) scan();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / iters;
}

void demo() {
    const int n = 8;
    float h_in[n] = {3, 1, 7, 0, 4, 1, 6, 3};
    float *d_in, *d_inc, *d_exc;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inc, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    hillisSteele<<<1, n, 2 * n * sizeof(float)>>>(d_in, d_inc, nullptr, n);   // 8 threads, 8 elements
    blelloch<false><<<1, n / 2, n * sizeof(float)>>>(d_in, d_exc, nullptr, n); // 4 threads, 8 elements
    CUDA_CHECK(cudaGetLastError());
    float inc[n], exc[n];
    CUDA_CHECK(cudaMemcpy(inc, d_inc, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(exc, d_exc, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("one block, 8 values:  "); for (int i = 0; i < n; ++i) printf(" %g", h_in[i]); printf("\n");
    printf("  inclusive (Hillis-Steele):"); for (int i = 0; i < n; ++i) printf(" %g", inc[i]); printf("\n");
    printf("  exclusive (Blelloch)     :"); for (int i = 0; i < n; ++i) printf(" %g", exc[i]); printf("\n");
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_inc)); CUDA_CHECK(cudaFree(d_exc));
}

int main() {
    const int N = 16 * 1024 * 1024, ITERS = 20, WARMUP = 100;
    const size_t bytes = (size_t)N * sizeof(float);

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (compute capability %d.%d), CUB %d.%d.%d\n", p.name, p.major, p.minor,
           CUB_VERSION / 100000, CUB_VERSION / 100 % 1000, CUB_VERSION % 100);
    demo();

    // Input: 0s and 1s from a fixed linear congruential generator; host references in exact integers.
    std::vector<float> h_in(N), h_out(N);
    std::vector<long long> incl(N), excl(N);
    unsigned int x = 12345u;
    long long run = 0;
    for (int i = 0; i < N; ++i) {
        x = 1664525u * x + 1013904223u;
        h_in[i] = (float)(x >> 31);
        excl[i] = run; run += (long long)h_in[i]; incl[i] = run;
    }

    float *d_in, *d_out, *work[8] = {nullptr};
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    // Workspace for the block totals and their scans, two buffers per level, allocated once.
    for (int lv = 0, cnt = (N + BLOCK - 1) / BLOCK; lv < 4; ++lv, cnt = (cnt + BLOCK - 1) / BLOCK) {
        CUDA_CHECK(cudaMalloc(&work[2 * lv], (cnt + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&work[2 * lv + 1], (cnt + 1) * sizeof(float)));
    }
    void* d_temp = nullptr;
    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_in, d_out, N));   // asks for the size
    CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Clock warm-up: without it the first rows are timed while the GPU is still leaving its idle clock.
    for (int w = 0; w < WARMUP; ++w) scanDevice(HILLIS_STEELE, d_in, d_out, N, work, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("N = %d floats of 0 or 1, %d threads per block, average of %d timed scans after %d warm-up scans\n",
           N, BLOCK, ITERS, WARMUP);
    printf("method                    result     per block  levels  time (ms)    GB/s  check\n");

    bool allOk = true;
    auto check = [&](bool inclusive) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        const std::vector<long long>& ref = inclusive ? incl : excl;
        for (int i = 0; i < N; ++i)
            if ((long long)h_out[i] != ref[i]) { allOk = false; return false; }
        return true;
    };
    auto row = [&](const char* name, const char* result, int per, int lv, float ms, bool ok) {
        char perS[16] = "-", lvS[16] = "-";
        if (per) snprintf(perS, sizeof perS, "%d", per);
        if (lv) snprintf(lvS, sizeof lvS, "%d", lv);
        printf("%-25s %-10s %9s  %6s  %9.3f  %6.1f  %s\n", name, result, perS, lvS, ms,
               2.0 * bytes / (ms * 1e6), ok ? "ok" : "MISMATCH");
    };

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    float ms = averageMs([&] { copyKernel<<<N / BLOCK, BLOCK>>>(d_in, d_out, N); }, start, stop, ITERS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    bool copyOk = true;
    for (int i = 0; i < N; ++i) copyOk = copyOk && (h_out[i] == h_in[i]);
    allOk = allOk && copyOk;
    row("copy (the ceiling)", "copy", 0, 0, ms, copyOk);

    const char* names[4] = {"Hillis-Steele, shared", "Blelloch, plain index", "Blelloch, conflict-free",
                            "warp shuffle"};
    Algo algos[4] = {HILLIS_STEELE, BLELLOCH, BLELLOCH_PADDED, WARP_SHUFFLE};
    for (int k = 0; k < 4; ++k) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        ms = averageMs([&] { scanDevice(algos[k], d_in, d_out, N, work, 0); }, start, stop, ITERS);
        bool inc = isInclusive(algos[k]);
        row(names[k], inc ? "inclusive" : "exclusive", perBlock(algos[k]), levels(algos[k], N), ms, check(inc));
    }

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    ms = averageMs([&] { cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_in, d_out, N); }, start, stop, ITERS);
    row("CUB DeviceScan", "exclusive", 0, 0, ms, check(false));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int k = 0; k < 8; ++k) CUDA_CHECK(cudaFree(work[k]));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return allOk ? 0 : 1;
}
