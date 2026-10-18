// graphs.cu: what a kernel launch costs, what a CUDA graph replay costs instead, and what cudaMalloc and
// cudaFree cost against a stream-ordered memory pool (post 13).
//
// It prints, on the card it runs on:
//   1. the launch sweep: kernels of a set length W (one warp spinning for W microseconds of SM cycles; W = 0 is an
//      empty kernel), issued as 2,000 direct launches (100 steps of 20 kernels) and as one 20-kernel graph
//      replayed 100 times; for each, the host's issue time per kernel and the time per kernel until the GPU has
//      finished, and the one-time cost of capturing and instantiating the graph;
//   2. allocation, 4 MiB from the default pool: a cudaMalloc plus cudaFree pair against a cudaMallocAsync plus
//      cudaFreeAsync pair; the host time each call blocks while a 20 ms kernel runs on another stream; and a step
//      of allocate, launch, free and synchronize with cudaMalloc and with the pool, its release threshold at 0
//      (the default) and at its maximum, split into the time spent in each call;
//   3. what a capture accepts: cudaMalloc refused, the pool's calls recorded as graph nodes;
// and the SM clock measured before and after (clock64() cycles over CUDA-event time).
// Every timed row is the median of 7 passes, timed on the host with std::chrono, after about half a second of
// untimed launches. These are host-side costs: they depend on the CPU, the driver and the operating system as
// much as on the GPU.
//
// Build: nvcc -O3 -arch=sm_75 posts/13-cuda-graphs/snippets/graphs.cu -o graphs   (from the series root)
// Run:   ./graphs            (on Windows: .\graphs.exe)
// Needs CUDA 12 or later (the 3-argument cudaGraphInstantiate; CUDA 11 has a 5-argument form) and a device that
// supports memory pools (cudaMallocAsync, CUDA 11.2 or later). Exits with status 1 if a CUDA call fails.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

using Clock = std::chrono::steady_clock;
static double usSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

const int KERNELS = 20;                             // kernels per step, and nodes per graph
const int STEPS = 100;                              // steps per pass: 2,000 kernels
const int PASSES = 7;

__global__ void empty() {}

// One warp spins for a set number of SM cycles: a kernel whose device time W is chosen, not measured.
__global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

void launch(cudaStream_t s, long long cycles) {
    if (cycles == 0) empty<<<1, 32, 0, s>>>();
    else spin<<<1, 32, 0, s>>>(cycles);
}

// Capture one step of KERNELS launches into a graph and instantiate it: paid once, before any replay.
cudaGraphExec_t buildGraph(cudaStream_t s, long long cycles) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < KERNELS; ++k) launch(s, cycles);   // recorded as nodes, not run
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));     // CUDA 12 form: graph, flags
    CUDA_CHECK(cudaGraphDestroy(graph));                   // the executable graph keeps its own copy
    return exec;
}

// One stream-ordered allocation, used and freed on stream s: no device-wide synchronization.
void poolRoundTrip(cudaStream_t s, size_t bytes) {
    void* d = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d, bytes, s));   // from the pool, ordered in s
    empty<<<1, 32, 0, s>>>();                     // the work that uses d
    CUDA_CHECK(cudaFreeAsync(d, s));             // back to the pool once s reaches this point
}

float smClockMHz(cudaEvent_t start, cudaEvent_t stop) {
    const long long cycles = 20000000;
    cudaEventRecord(start);
    spin<<<1, 1>>>(cycles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return cycles / (ms * 1e3f);                    // cycles per microsecond = MHz
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0, pools = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0));
    printf("device: %s (compute capability %d.%d), CUDA driver API %d, runtime %d, memory pools %s\n",
           prop.name, prop.major, prop.minor, driver, runtime, pools ? "supported" : "NOT supported");
    if (!pools) return EXIT_FAILURE;

    cudaStream_t s, other;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Untimed: about half a second of launches, so the card and the driver reach their working state.
    for (int i = 0; i < 100000; ++i) empty<<<1, 32, 0, s>>>();
    spin<<<1, 32, 0, s>>>(500000000LL);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaGetLastError());
    const float mhzBefore = smClockMHz(e0, e1);
    const double cyclesPerUs = mhzBefore;           // W microseconds become W * MHz cycles

    // ---- 1. Direct launches against graph replay, for kernels of length W ----
    const double W_US[] = {0, 2, 4, 8, 16, 32};
    const int NW = sizeof(W_US) / sizeof(W_US[0]);
    printf("launch sweep: %d kernels per pass (%d steps of %d), one warp each; median of %d passes; us per kernel\n",
           KERNELS * STEPS, STEPS, KERNELS, PASSES);
    printf("  W (us)   direct issue   direct done   graph issue   graph done   speedup\n");
    double buildUs = 0.0;
    for (int w = 0; w < NW; ++w) {
        const long long cycles = (long long)(W_US[w] * cyclesPerUs);
        std::vector<double> dIssue, dDone, gIssue, gDone, build;
        for (int p = 0; p < PASSES; ++p) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            Clock::time_point t0 = Clock::now();
            for (int st = 0; st < STEPS; ++st)
                for (int k = 0; k < KERNELS; ++k) launch(s, cycles);
            dIssue.push_back(usSince(t0) / (KERNELS * STEPS));
            CUDA_CHECK(cudaStreamSynchronize(s));
            dDone.push_back(usSince(t0) / (KERNELS * STEPS));
        }
        for (int p = 0; p < PASSES; ++p) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            Clock::time_point tb = Clock::now();
            cudaGraphExec_t exec = buildGraph(s, cycles);
            build.push_back(usSince(tb));
            CUDA_CHECK(cudaGraphLaunch(exec, s));                    // the first replay uploads the graph
            CUDA_CHECK(cudaStreamSynchronize(s));
            Clock::time_point t0 = Clock::now();
            for (int st = 0; st < STEPS; ++st) CUDA_CHECK(cudaGraphLaunch(exec, s));
            gIssue.push_back(usSince(t0) / (KERNELS * STEPS));
            CUDA_CHECK(cudaStreamSynchronize(s));
            gDone.push_back(usSince(t0) / (KERNELS * STEPS));
            CUDA_CHECK(cudaGraphExecDestroy(exec));
        }
        CUDA_CHECK(cudaGetLastError());
        if (w == 0) buildUs = median(build);
        printf("  %6.0f   %12.2f   %11.2f   %11.2f   %10.2f   %6.2fx\n", W_US[w], median(dIssue), median(dDone),
               median(gIssue), median(gDone), median(dDone) / median(gDone));
    }
    printf("graph build (capture %d empty kernels + instantiate): %.1f us, once\n", KERNELS, buildUs);

    // ---- 2. cudaMalloc and cudaFree against the stream-ordered pool ----
    const size_t bytes = 4u << 20;                  // 4 MiB
    const int ALLOCS = 200;
    std::vector<double> mPair, aPair;
    for (int p = 0; p < PASSES; ++p) {
        CUDA_CHECK(cudaDeviceSynchronize());
        Clock::time_point t0 = Clock::now();
        for (int i = 0; i < ALLOCS; ++i) {
            void* d = nullptr;
            CUDA_CHECK(cudaMalloc(&d, bytes));
            CUDA_CHECK(cudaFree(d));
        }
        mPair.push_back(usSince(t0) / ALLOCS);
        t0 = Clock::now();
        for (int i = 0; i < ALLOCS; ++i) {
            void* d = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d, bytes, s));
            CUDA_CHECK(cudaFreeAsync(d, s));
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
        aPair.push_back(usSince(t0) / ALLOCS);
    }
    printf("allocation, 4 MiB, GPU idle; median of %d passes of %d pairs; us per pair\n", PASSES, ALLOCS);
    printf("  cudaMalloc + cudaFree           : %8.2f\n", median(mPair));
    printf("  cudaMallocAsync + cudaFreeAsync : %8.2f\n", median(aPair));

    // How long each call holds the host while a 20 ms kernel runs on another stream. The pool's release
    // threshold decides whether it still holds memory after a synchronize: 0 (the default) gives it all back.
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t keepNone = 0, keepAll = UINT64_MAX;
    const long long busy = (long long)(20000.0 * cyclesPerUs);
    std::vector<double> tMalloc, tFree, tTrim, tKeep, tKernel;
    for (int p = 0; p < PASSES; ++p) {
        void* d = nullptr;
        CUDA_CHECK(cudaDeviceSynchronize());
        spin<<<1, 32, 0, other>>>(busy);
        Clock::time_point t0 = Clock::now();
        CUDA_CHECK(cudaMalloc(&d, bytes));
        tMalloc.push_back(usSince(t0));
        CUDA_CHECK(cudaDeviceSynchronize());
        tKernel.push_back(usSince(t0));

        spin<<<1, 32, 0, other>>>(busy);
        t0 = Clock::now();
        CUDA_CHECK(cudaFree(d));
        tFree.push_back(usSince(t0));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepNone));
        spin<<<1, 32, 0, other>>>(busy);
        t0 = Clock::now();
        poolRoundTrip(s, bytes);
        tTrim.push_back(usSince(t0));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepAll));
        poolRoundTrip(s, bytes);                                    // fill the pool once, untimed
        CUDA_CHECK(cudaDeviceSynchronize());
        spin<<<1, 32, 0, other>>>(busy);
        t0 = Clock::now();
        poolRoundTrip(s, bytes);
        tKeep.push_back(usSince(t0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("host time a call blocks while a %.1f ms kernel runs on another stream; median of %d; us\n",
           median(tKernel) / 1000.0, PASSES);
    printf("  cudaMalloc                     : %8.1f\n", median(tMalloc));
    printf("  cudaFree                       : %8.1f\n", median(tFree));
    printf("  poolRoundTrip, pool trimmed    : %8.1f\n", median(tTrim));
    printf("  poolRoundTrip, pool kept       : %8.1f\n", median(tKeep));

    // A step that allocates, launches, frees and synchronizes, three ways, with the time inside each call.
    printf("step = allocate, kernel, free, cudaStreamSynchronize; median of %d passes of %d steps; us per step\n",
           PASSES, ALLOCS);
    printf("                                          step   allocate       free   synchronize\n");
    for (int way = 0; way < 3; ++way) {
        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, way == 2 ? &keepAll : &keepNone));
        std::vector<double> step, alloc, freed, sync;
        for (int p = 0; p < PASSES; ++p) {
            double a = 0, f = 0, y = 0;
            Clock::time_point t0 = Clock::now();
            for (int i = 0; i < ALLOCS; ++i) {
                void* d = nullptr;
                Clock::time_point t = Clock::now();
                if (way == 0) CUDA_CHECK(cudaMalloc(&d, bytes));
                else CUDA_CHECK(cudaMallocAsync(&d, bytes, s));
                a += usSince(t);
                empty<<<1, 32, 0, s>>>();
                t = Clock::now();
                if (way == 0) CUDA_CHECK(cudaFree(d));
                else CUDA_CHECK(cudaFreeAsync(d, s));
                f += usSince(t);
                t = Clock::now();
                CUDA_CHECK(cudaStreamSynchronize(s));
                y += usSince(t);
            }
            step.push_back(usSince(t0) / ALLOCS);
            alloc.push_back(a / ALLOCS);
            freed.push_back(f / ALLOCS);
            sync.push_back(y / ALLOCS);
        }
        const char* name[] = {"cudaMalloc + cudaFree               ",
                              "pool, release threshold 0 (default)",
                              "pool, release threshold maximum    "};
        printf("  %s %8.2f   %8.2f   %8.2f   %11.2f\n", name[way], median(step), median(alloc), median(freed),
               median(sync));
    }
    CUDA_CHECK(cudaGetLastError());

    // ---- 3. What a capture accepts ----
    {
        void* d = nullptr;
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        cudaError_t mallocErr = cudaMalloc(&d, bytes);
        cudaStreamEndCapture(s, &graph);                    // the refusal invalidates the capture; end it
        cudaGetLastError();                                 // and clear the error it left
        if (mallocErr == cudaSuccess) CUDA_CHECK(cudaFree(d));
        if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
        printf("capture: cudaMalloc inside a global-mode capture returns \"%s\"\n", cudaGetErrorString(mallocErr));

        size_t nodes = 0;
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        poolRoundTrip(s, bytes);
        CUDA_CHECK(cudaStreamEndCapture(s, &graph));
        CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaGraphLaunch(exec, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        printf("capture: poolRoundTrip becomes a graph of %zu nodes (allocate, kernel, free), replayed ok\n", nodes);
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    const float mhzAfter = smClockMHz(e0, e1);
    printf("SM clock: %.0f MHz before the timed rows, %.0f MHz after\n", mhzBefore, mhzAfter);

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaStreamDestroy(other));
    return EXIT_SUCCESS;
}
