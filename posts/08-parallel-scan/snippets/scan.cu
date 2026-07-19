// scan.cu — block-level prefix sum (scan) two ways: Hillis-Steele (simple,
// work-inefficient) and Blelloch (up-sweep + down-sweep, work-efficient).
//
// Build: nvcc -O3 -arch=sm_75 snippets/scan.cu -o scan   (paths relative to the post dir)
// Run:   ./scan

#include <cuda_runtime.h>
#include <cstdio>

#define N 8                          // one block, power of two, for clarity

// Hillis-Steele inclusive scan: each step adds the element `stride` back.
// O(N log N) work, O(log N) depth. Double-buffered to avoid races.
__global__ void hillisSteele(float* out, const float* in, int n) {
    extern __shared__ float buf[];   // size 2*n: two ping-pong halves
    int t = threadIdx.x, pin = 0, pout = 1;
    buf[t] = in[t];
    __syncthreads();
    for (int stride = 1; stride < n; stride *= 2) {
        pout = 1 - pout; pin = 1 - pin;
        buf[pout * n + t] = buf[pin * n + t] + (t >= stride ? buf[pin * n + t - stride] : 0.0f);
        __syncthreads();
    }
    out[t] = buf[pout * n + t];      // inclusive scan
}

// Blelloch exclusive scan: up-sweep builds partial sums (a reduction tree),
// down-sweep distributes them. O(N) work, ~2 log N parallel steps.
__global__ void blelloch(float* out, const float* in, int n) {
    __shared__ float s[N];
    int t = threadIdx.x, offset = 1;
    s[2 * t] = in[2 * t]; s[2 * t + 1] = in[2 * t + 1];

    for (int d = n >> 1; d > 0; d >>= 1) {            // up-sweep
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            s[bi] += s[ai];
        }
        offset *= 2;
    }
    if (t == 0) s[n - 1] = 0.0f;                      // clear root for exclusive scan
    for (int d = 1; d < n; d *= 2) {                  // down-sweep
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            float tmp = s[ai];                        // save left child
            s[ai] = s[bi];                            // left <- parent
            s[bi] += tmp;                             // right <- parent + saved
        }
    }
    __syncthreads();
    out[2 * t] = s[2 * t]; out[2 * t + 1] = s[2 * t + 1];
}

int main() {
    float h_in[N] = {3, 1, 7, 0, 4, 1, 6, 3};
    float *d_in, *d_inc, *d_exc;
    cudaMalloc(&d_in, N * 4); cudaMalloc(&d_inc, N * 4); cudaMalloc(&d_exc, N * 4);
    cudaMemcpy(d_in, h_in, N * 4, cudaMemcpyHostToDevice);

    hillisSteele<<<1, N, 2 * N * sizeof(float)>>>(d_inc, d_in, N);
    blelloch<<<1, N / 2>>>(d_exc, d_in, N);
    cudaDeviceSynchronize();

    float inc[N], exc[N];
    cudaMemcpy(inc, d_inc, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(exc, d_exc, N * 4, cudaMemcpyDeviceToHost);
    printf("inclusive (Hillis-Steele): "); for (int i = 0; i < N; ++i) printf("%g ", inc[i]); printf("\n");
    printf("exclusive (Blelloch)     : "); for (int i = 0; i < N; ++i) printf("%g ", exc[i]); printf("\n");

    cudaFree(d_in); cudaFree(d_inc); cudaFree(d_exc);
    return 0;
}
