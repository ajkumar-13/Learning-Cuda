// precision.cu — the numerical WHY behind mixed precision, on the GPU.
//
// Two kernels compute the SAME length-K dot product of two FP16 vectors. One
// accumulates the running sum in FP16 (fast, but only ~3 digits); the other
// multiplies in FP16 and accumulates in FP32 (the Tensor Core recipe of post
// 17). Both are checked against a host FP64 reference computed on the same
// FP16-rounded operands, so the only variable is the accumulator's precision.
//
// Build: nvcc -O3 -arch=sm_75 snippets/precision.cu -o precision   (from the post dir)
// Run:   ./precision
//
// Self-contained: own main(), CUDA_CHECK, FP64 reference, tolerance check.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define K     4096          // dot-product length
#define BLOCK 256           // one block; each thread strides over K/BLOCK elements

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                   \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Accumulate the dot product entirely in FP16: multiply AND sum in half. The
// running sum grows past what FP16's ~3 digits can track, so small terms are
// lost. One block; a shared-memory tree reduction, also in FP16.
__global__ void dotFP16(const __half* a, const __half* b, float* out, int n) {
    __shared__ __half s[BLOCK];
    __half acc = __float2half(0.0f);
    for (int i = threadIdx.x; i < n; i += BLOCK)
        acc = __hadd(acc, __hmul(a[i], b[i]));           // FP16 multiply, FP16 add
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] = __hadd(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = __half2float(s[0]);
}

// Mixed precision: FP16 inputs, FP16 multiply, but accumulate in FP32. Same
// data, same tree, only the accumulator's precision differs.
__global__ void dotMixed(const __half* a, const __half* b, float* out, int n) {
    __shared__ float s[BLOCK];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK)
        acc += __half2float(a[i]) * __half2float(b[i]);  // FP16 in, FP32 accumulate
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s[0];
}

int main() {
    // Deterministic operands, small so the FP16 products are well-scaled.
    __half *h_a = (__half*)malloc(K * sizeof(__half));
    __half *h_b = (__half*)malloc(K * sizeof(__half));
    double ref = 0.0;                                    // FP64 reference on the FP16 operands
    for (int i = 0; i < K; ++i) {
        h_a[i] = __float2half((float)((i % 13) - 6) * 0.1f);
        h_b[i] = __float2half((float)((i % 7)  - 3) * 0.1f);
        ref += (double)__half2float(h_a[i]) * (double)__half2float(h_b[i]);
    }

    __half *d_a, *d_b; float *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, K * sizeof(__half), cudaMemcpyHostToDevice));

    float r16 = 0.0f, r32 = 0.0f;
    dotFP16<<<1, BLOCK>>>(d_a, d_b, d_out, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&r16, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    dotMixed<<<1, BLOCK>>>(d_a, d_b, d_out, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&r32, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    double e16 = fabs((double)r16 - ref);
    double e32 = fabs((double)r32 - ref);
    const double TOL = 1e-3;                              // FP32-accumulate should meet this
    printf("dot product length K=%d, FP64 reference = %.6f\n", K, ref);
    printf("  FP16 accumulate : %.6f   abs err %.3e\n", r16, e16);
    printf("  FP32 accumulate : %.6f   abs err %.3e   (%s within %.0e)\n",
           r32, e32, e32 < TOL ? "PASS" : "FAIL", TOL);
    printf("  -> multiply in FP16 for speed, accumulate in FP32 for accuracy\n");

    free(h_a); free(h_b);
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    return e32 < TOL ? 0 : 1;
}
