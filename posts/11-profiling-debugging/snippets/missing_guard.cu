// missing_guard.cu: post 02's vector addition with and without its guard, for compute-sanitizer's memcheck.
//
// N = 6 elements in blocks of 4 threads: a grid of 2 blocks, 8 threads, 2 of them surplus (threads 6 and 7).
// By default the guarded kernel runs. With --no-guard the kernel without `if (i < N)` runs instead: threads 6
// and 7 read A[6], A[7], B[6], B[7] and write C[6], C[7], past the end of 6-element arrays. The program
// itself usually notices nothing, since the bytes past a small allocation are padding no one checks.
//
// Build: nvcc -O3 -arch=sm_75 -lineinfo posts/11-profiling-debugging/snippets/missing_guard.cu -o missing_guard   (from the series root)
// Run:   ./missing_guard                                                      (guarded; exits 0)
//        compute-sanitizer --tool memcheck ./missing_guard --no-guard          (the bug, reported)
//        (on Windows: .\missing_guard.exe)
//
// The program exits with status 1 if a CUDA call fails or an element of C is wrong.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)                       // the guard: threads past the end do nothing
        C[i] = A[i] + B[i];
}

__global__ void vectorAddNoGuard(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];              // BUG: no guard; threads 6 and 7 read and write past the end
}

int main(int argc, char** argv) {
    const bool noGuard = (argc > 1 && strcmp(argv[1], "--no-guard") == 0);
    const int N = 6, threads = 4, blocks = (N + threads - 1) / threads;   // 2 blocks, 8 threads
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; ++i) { h_A[i] = (float)i; h_B[i] = 10.0f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));

    if (noGuard) vectorAddNoGuard<<<blocks, threads>>>(d_A, d_B, d_C, N);
    else         vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));

    int right = 0;
    for (int i = 0; i < N; ++i) right += (h_C[i] == h_A[i] + h_B[i]);
    printf("%s<<<%d, %d>>> over N = %d: %d threads, %d surplus\n", noGuard ? "vectorAddNoGuard" : "vectorAdd",
           blocks, threads, N, blocks * threads, blocks * threads - N);
    printf("elements equal to A + B: %d of %d\n", right, N);

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return right == N ? EXIT_SUCCESS : EXIT_FAILURE;
}
