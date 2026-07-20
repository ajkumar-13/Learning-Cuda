// graphs.cu — cut per-launch host overhead with CUDA Graphs, and stop the
// device-wide stalls of cudaMalloc with a stream-ordered memory pool.
//
// Three timed sections on one trivial kernel:
//   (i)   N direct launches           -> pays the host launch cost N times
//   (ii)  one captured graph, replayed -> pays the launch cost once, then replays
//   (iii) cudaMallocAsync from a pool  -> stream-ordered allocation, no device sync
//
// The printed speedup depends entirely on your GPU and driver: it is largest for
// tiny kernels (host-overhead bound) and shrinks toward 1x as the kernel grows.
//
// Build: nvcc -O3 -arch=sm_75 snippets/graphs.cu -o graphs   (from the post directory)
// Run:   ./graphs
// Needs CUDA 12+ (the 3-arg cudaGraphInstantiate below; on CUDA 11 use the
// 5-arg form cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)).
// cudaMallocAsync / stream-ordered pools need CUDA 11.4+.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define N_ITERS 5000        // many small launches: the regime where graphs win
#define ELEMS   (1 << 12)   // deliberately small kernel, so host overhead dominates

// A trivial, cheap kernel: one FMA per element. Small on purpose.
__global__ void bump(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 1.0009765625f + 1.0f;   // exact in float, keeps values finite
}

int main() {
    const int block = 256, grid = (ELEMS + block - 1) / block;
    size_t bytes = (size_t)ELEMS * sizeof(float);

    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMemset(d_x, 0, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // ---- (i) N DIRECT LAUNCHES: pay the host launch overhead every iteration ----
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < N_ITERS; ++i)
        bump<<<grid, block, 0, stream>>>(d_x, ELEMS);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_direct = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_direct, t0, t1));

    // ---- (ii) CAPTURE ONCE, REPLAY N TIMES: one enqueue, then cudaGraphLaunch ----
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    bump<<<grid, block, 0, stream>>>(d_x, ELEMS);      // record one node into the graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));   // CUDA 12+ 3-arg form

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < N_ITERS; ++i)
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));  // replay: no per-node re-launch
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_graph = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph, t0, t1));

    printf("kernel: %d small launches of a %d-element kernel\n", N_ITERS, ELEMS);
    printf("  direct launches : %8.3f ms\n", ms_direct);
    printf("  graph replay    : %8.3f ms\n", ms_graph);
    printf("  speedup         : %8.2fx   (depends on your GPU + driver)\n",
           ms_graph > 0.0f ? ms_direct / ms_graph : 0.0f);

    // ---- (iii) STREAM-ORDERED ALLOCATION from a pool: no device-wide stall ----
    // cudaMallocAsync draws from a memory pool and is ordered in the stream, so
    // it does NOT synchronize the whole device the way cudaMalloc/cudaFree do.
    // Reused across iterations, the pool returns the same block for ~free.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int poolSupported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&poolSupported,
                                      cudaDevAttrMemoryPoolsSupported, device));
    if (poolSupported) {
        for (int i = 0; i < 3; ++i) {
            float* d_tmp = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_tmp, bytes, stream));   // pool alloc, stream-ordered
            bump<<<grid, block, 0, stream>>>(d_tmp, ELEMS);
            CUDA_CHECK(cudaFreeAsync(d_tmp, stream));             // returns block to the pool
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("  cudaMallocAsync : pool alloc/free x3 ok (no device-wide sync per call)\n");
    } else {
        printf("  cudaMallocAsync : not supported on this device/driver; skipped\n");
    }

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_x));
    return 0;
}
