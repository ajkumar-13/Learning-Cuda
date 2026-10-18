// streams.cu: post 12's benchmark. C = A + B over N = 2^24 floats (64 MiB per array), with the two inputs copied
// to the device and the result copied back, run serially and pipelined across streams, for two kernels: plain
// vector addition (extra = 0), whose copies take far longer than its arithmetic, and the same addition followed by
// EXTRA rounds of c = sqrtf(c * c), work that leaves c unchanged and makes the kernel as long as a copy.
//
// It prints:
//   - the card, its asyncEngineCount and whether it runs kernels of different streams concurrently;
//   - the stages alone, from pinned memory: the two input copies, each kernel, the result copy;
//   - one host-to-device and one device-to-host copy of 64 MiB, one after the other and then at the same time on
//     two streams (do the two directions overlap on this card?);
//   - the end-to-end table: serial from pageable memory, serial from pinned memory, then 1, 2, 4, 8, 16 and 32
//     streams from pinned memory and 4 streams from pageable memory; each cell is the fastest of 5 passes, with the
//     speed-up over the pageable serial row, the effective GB/s of vector addition (the 201 MB that cross the bus,
//     two arrays in and one out, over the time), and a check of every element of C;
//   - the 4-stream run of each kernel once more with a CUDA event after every operation (ms from the start);
//   - the SM clock before and after the timed rows (clock64() cycles over CUDA-event time).
// The program starts with about half a second of untimed kernels, so the card has left its idle clock, and every
// timed row starts with a sixth of a second more, because the host-side check between rows lets the clock drop.
//
// Build: nvcc -O3 -arch=sm_75 posts/12-cuda-streams/snippets/streams.cu -o streams   (from the series root)
// Run:   ./streams            (on Windows: .\streams.exe)
//
// The program exits with status 1 if a CUDA call fails or any element of C is wrong.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

const int N = 1 << 24;                                // 16,777,216 floats, 64 MiB per array
const size_t BYTES = (size_t)N * sizeof(float);
const int THREADS = 256;
const int PASSES = 5;
const int EXTRA = 64;                                 // rounds of extra work in the compute-heavy kernel

__global__ void vectorAdd(const float* A, const float* B, float* C, int n, int extra) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's element of the chunk
    if (i < n) {
        float c = A[i] + B[i];
        for (int k = 0; k < extra; ++k)               // extra = 0: plain vector addition
            c = sqrtf(c * c);                         // exact for these small integers, so c is unchanged
        C[i] = c;
    }
}

// The SM clock: clock64() counts SM cycles; CUDA events time the same spin in milliseconds.
__global__ void clockSpin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

float smClockMHz(cudaEvent_t start, cudaEvent_t stop) {
    const long long cycles = 20000000;
    CUDA_CHECK(cudaEventRecord(start));
    clockSpin<<<1, 1>>>(cycles);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return cycles / (ms * 1e3f);                     // cycles per microsecond = MHz
}

struct Host { float *A, *B, *C; };
struct Dev  { float *A, *B, *C; };

// Serial: everything on the default stream, one operation after another.
void serial(Host h, Dev d, int extra) {
    CUDA_CHECK(cudaMemcpy(d.A, h.A, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.B, h.B, BYTES, cudaMemcpyHostToDevice));
    vectorAdd<<<(N + THREADS - 1) / THREADS, THREADS>>>(d.A, d.B, d.C, N, extra);
    CUDA_CHECK(cudaMemcpy(h.C, d.C, BYTES, cudaMemcpyDeviceToHost));
}

// Pipelined: chunk i of the arrays goes in, is computed and comes out on stream i.
void pipelined(Host h, Dev d, int extra, cudaStream_t* s, int NS, cudaEvent_t (*ev)[4] = nullptr) {
    const int chunk = N / NS;                        // N is a multiple of every NS used
    const size_t chunkBytes = (size_t)chunk * sizeof(float);
    const int grid = (chunk + THREADS - 1) / THREADS;
    for (int i = 0; i < NS; ++i) {
        int off = i * chunk;
        cudaMemcpyAsync(&d.A[off], &h.A[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
        if (ev) cudaEventRecord(ev[i][0], s[i]);
        cudaMemcpyAsync(&d.B[off], &h.B[off], chunkBytes, cudaMemcpyHostToDevice, s[i]);
        if (ev) cudaEventRecord(ev[i][1], s[i]);
        vectorAdd<<<grid, THREADS, 0, s[i]>>>(&d.A[off], &d.B[off], &d.C[off], chunk, extra);
        if (ev) cudaEventRecord(ev[i][2], s[i]);
        cudaMemcpyAsync(&h.C[off], &d.C[off], chunkBytes, cudaMemcpyDeviceToHost, s[i]);
        if (ev) cudaEventRecord(ev[i][3], s[i]);
    }
    CUDA_CHECK(cudaGetLastError());
}

int wrongElements(const float* C) {
    int wrong = 0;
    for (int i = 0; i < N; ++i) wrong += (C[i] != (float)(i & 1023) + (float)((i >> 10) & 1023));
    return wrong;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (compute capability %d.%d), asyncEngineCount %d, concurrentKernels %d\n",
           prop.name, prop.major, prop.minor, prop.asyncEngineCount, prop.concurrentKernels);

    // Host arrays twice: pageable (std::vector, ordinary heap memory) and pinned (cudaMallocHost).
    std::vector<float> pA(N), pB(N), pC(N);
    Host pageable = {pA.data(), pB.data(), pC.data()}, pinned;
    CUDA_CHECK(cudaMallocHost(&pinned.A, BYTES));
    CUDA_CHECK(cudaMallocHost(&pinned.B, BYTES));
    CUDA_CHECK(cudaMallocHost(&pinned.C, BYTES));
    for (int i = 0; i < N; ++i) {                    // small integers, so every sum is exact in float
        pA[i] = pinned.A[i] = (float)(i & 1023);
        pB[i] = pinned.B[i] = (float)((i >> 10) & 1023);
    }
    Dev d;
    CUDA_CHECK(cudaMalloc(&d.A, BYTES));
    CUDA_CHECK(cudaMalloc(&d.B, BYTES));
    CUDA_CHECK(cudaMalloc(&d.C, BYTES));
    cudaStream_t s[32];
    for (int i = 0; i < 32; ++i) CUDA_CHECK(cudaStreamCreate(&s[i]));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int w = 0; w < 400; ++w) vectorAdd<<<N / THREADS, THREADS>>>(d.A, d.B, d.C, N, 0);   // untimed: reach the clock
    CUDA_CHECK(cudaDeviceSynchronize());
    float mhzBefore = smClockMHz(start, stop);

    // The fastest of PASSES runs of one piece of work, timed by events on the default stream, which waits for every
    // other stream: so the stop event completes only when all streams have finished. Each row starts with about a
    // sixth of a second of untimed kernels, because the host-side check between rows lets the card drop its clock.
    auto best = [&](auto work) {
        for (int w = 0; w < 120; ++w) vectorAdd<<<N / THREADS, THREADS>>>(d.A, d.B, d.C, N, 0);
        CUDA_CHECK(cudaMemset(d.C, 0xFF, BYTES));    // NaN again, so the timed work must write every element
        float bestMs = 1e30f;
        for (int r = 0; r < PASSES; ++r) {
            CUDA_CHECK(cudaEventRecord(start, 0));
            work();
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            bestMs = std::min(bestMs, ms);
        }
        return bestMs;
    };

    // The stages alone, from pinned memory.
    float h2dMs = best([&] {
        cudaMemcpyAsync(d.A, pinned.A, BYTES, cudaMemcpyHostToDevice, 0);
        cudaMemcpyAsync(d.B, pinned.B, BYTES, cudaMemcpyHostToDevice, 0); });
    float addMs = best([&] { vectorAdd<<<N / THREADS, THREADS>>>(d.A, d.B, d.C, N, 0); });
    float heavyMs = best([&] { vectorAdd<<<N / THREADS, THREADS>>>(d.A, d.B, d.C, N, EXTRA); });
    float d2hMs = best([&] { cudaMemcpyAsync(pinned.C, d.C, BYTES, cudaMemcpyDeviceToHost, 0); });
    // One copy each way, one after the other and then on two streams at once.
    float oneWayMs = best([&] {
        cudaMemcpyAsync(d.A, pinned.A, BYTES, cudaMemcpyHostToDevice, s[0]);
        cudaMemcpyAsync(pinned.C, d.C, BYTES, cudaMemcpyDeviceToHost, s[0]); });
    float bothWaysMs = best([&] {
        cudaMemcpyAsync(d.A, pinned.A, BYTES, cudaMemcpyHostToDevice, s[0]);
        cudaMemcpyAsync(pinned.C, d.C, BYTES, cudaMemcpyDeviceToHost, s[1]); });

    // The end-to-end table, for both kernels.
    struct Row { const char* name; Host h; int streams; float ms[2]; int wrong; };
    std::vector<Row> rows = {
        {"serial, pageable", pageable, 0}, {"serial, pinned", pinned, 0},
        {"1 stream, pinned", pinned, 1}, {"2 streams, pinned", pinned, 2}, {"4 streams, pinned", pinned, 4},
        {"8 streams, pinned", pinned, 8}, {"16 streams, pinned", pinned, 16}, {"32 streams, pinned", pinned, 32},
        {"4 streams, pageable", pageable, 4}};
    const int extras[2] = {0, EXTRA};
    for (Row& r : rows) {
        r.wrong = 0;
        for (int k = 0; k < 2; ++k) {
            memset(r.h.C, 0xFF, BYTES);              // NaN in every element, so a stale result cannot pass
            int extra = extras[k];
            r.ms[k] = r.streams == 0 ? best([&] { serial(r.h, d, extra); })
                                     : best([&] { pipelined(r.h, d, extra, s, r.streams); });
            r.wrong += wrongElements(r.h.C);
        }
    }

    // The 4-stream run of each kernel with an event after each operation: the timelines.
    cudaEvent_t ev[4][4];
    for (auto& e : ev) for (auto& x : e) CUDA_CHECK(cudaEventCreate(&x));
    float t[2][4][4];                                // ms from the start to the end of each operation
    for (int k = 0; k < 2; ++k) {
        for (int w = 0; w < 120; ++w) vectorAdd<<<N / THREADS, THREADS>>>(d.A, d.B, d.C, N, 0);
        float bestTotal = 1e30f;                     // keep the fastest of PASSES runs, as in the table
        for (int r = 0; r < PASSES; ++r) {
            CUDA_CHECK(cudaEventRecord(start, 0));
            pipelined(pinned, d, extras[k], s, 4, ev);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float total = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
            if (total >= bestTotal) continue;
            bestTotal = total;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) CUDA_CHECK(cudaEventElapsedTime(&t[k][i][j], start, ev[i][j]));
        }
    }
    float mhzAfter = smClockMHz(start, stop);

    const double mb = 1e-6 * BYTES;                  // MB per array; MB per ms is GB/s
    printf("N = %d floats, %.1f MB per array; fastest of %d passes; GB/s = MB over the time in ms\n", N, mb, PASSES);
    printf("stages alone, pinned: copy A and B in %.2f ms (%.1f GB/s), copy C out %.2f ms (%.1f GB/s)\n",
           h2dMs, 2 * mb / h2dMs, d2hMs, mb / d2hMs);
    printf("kernels alone: vector add %.2f ms (%.1f GB/s on the device), with %d extra rounds %.2f ms\n",
           addMs, 3 * mb / addMs, EXTRA, heavyMs);
    printf("one copy each way, %.1f MB: one after the other %.2f ms, at the same time on two streams %.2f ms\n",
           mb, oneWayMs, bothWaysMs);
    printf("                      vector add                    %d extra rounds\n", EXTRA);
    printf("method                time (ms)   GB/s  speed-up    time (ms)  speed-up  check\n");
    int wrongTotal = 0;
    for (const Row& r : rows) {
        printf("%-21s %9.2f  %5.1f  %7.2fx    %9.2f  %7.2fx  %s\n", r.name, r.ms[0], 3 * mb / r.ms[0],
               rows[0].ms[0] / r.ms[0], r.ms[1], rows[0].ms[1] / r.ms[1], r.wrong ? "MISMATCH" : "ok");
        wrongTotal += r.wrong;
    }
    for (int k = 0; k < 2; ++k) {
        printf("timeline, 4 streams, %s, ms from the start to the end of: A in, B in, kernel, C out\n",
               k == 0 ? "vector add" : "extra rounds");
        for (int i = 0; i < 4; ++i)
            printf("  stream %d: %6.2f %6.2f %6.2f %6.2f\n", i, t[k][i][0], t[k][i][1], t[k][i][2], t[k][i][3]);
    }
    printf("SM clock: %.0f MHz before the timed rows, %.0f MHz after\n", mhzBefore, mhzAfter);

    for (auto& e : ev) for (auto& x : e) CUDA_CHECK(cudaEventDestroy(x));
    for (int i = 0; i < 32; ++i) CUDA_CHECK(cudaStreamDestroy(s[i]));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(pinned.A)); CUDA_CHECK(cudaFreeHost(pinned.B)); CUDA_CHECK(cudaFreeHost(pinned.C));
    CUDA_CHECK(cudaFree(d.A)); CUDA_CHECK(cudaFree(d.B)); CUDA_CHECK(cudaFree(d.C));
    return wrongTotal ? EXIT_FAILURE : EXIT_SUCCESS;
}
