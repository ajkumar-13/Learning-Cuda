// precision.cu: what floating point does on the GPU, checked against precision_model.py.
//
// Four experiments, with the operands, the order of operations and the block size of the CPU
// model, so the model predicts every FP32 and FP16 result below bit for bit:
//   (b) sumBlocks: 2^20 floats summed with 1, 14, 56 and 224 blocks; each count is its own tree.
//   (c) the device's own conversions to FP16 and BF16 at the edges of their range.
//   (d) naive and stable softmax of [2, 3, 1000, 1001] in FP32.
//   (e) a length-4096 dot product of FP16 values, accumulated in FP16 (fused, as nvcc compiles
//       __hadd(acc, __hmul(a, b)), and rounded apart with the _rn intrinsics) and in FP32,
//       each checked against an FP64 reference with an error bound sized to its format.
// No timing: every number printed is a result of the arithmetic, not a speed.
//
// Build: nvcc -O3 -arch=sm_75 posts/16-numerical-precision/snippets/precision.cu -o precision   (from the series root)
// Run:   ./precision        (on Windows: .\precision.exe; exits 1 if a result leaves its bound)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define BLOCK 256                    // threads per block, as in the model
#define N_SUM (1 << 20)              // floats in experiment (b)
#define K     4096                   // dot-product length of experiment (e)

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// n floats in [0, 1) with 24 random bits each, the same integer hash as the model's hash01().
static float hash01(unsigned i, unsigned seed) {
    unsigned h = i * 2654435761u + seed * 0x9E3779B9u;
    h ^= h >> 16; h *= 0x45D9F3Bu; h ^= h >> 16;
    return (float)(h >> 8) / 16777216.0f;
}

// (b) Thread i adds x[i], x[i + T], ... with T the total thread count, then the block reduces its
// 256 sums in shared memory (post 04's tree) and writes one partial sum per block.
__global__ void sumBlocks(const float* x, float* partial, int n) {
    __shared__ float s[BLOCK];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += x[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

// (c) The device's conversions: FP16 and BF16 values of three FP32 inputs.
__global__ void convert(const float* in, float* h, float* bf, int n) {
    int i = threadIdx.x;
    if (i < n) {
        h[i]  = __half2float(__float2half(in[i]));
        bf[i] = __bfloat162float(__float2bfloat16(in[i]));
    }
}

// (d) One thread computes a short row both ways, so the two paths share every input.
__global__ void softmax(const float* x, float* naive, float* stable, int n) {
    float m = x[0], sn = 0.0f, ss = 0.0f;
    for (int j = 1; j < n; ++j) m = fmaxf(m, x[j]);
    for (int j = 0; j < n; ++j) { sn += expf(x[j]); ss += expf(x[j] - m); }
    for (int j = 0; j < n; ++j) { naive[j] = expf(x[j]) / sn; stable[j] = expf(x[j] - m) / ss; }
}

// (e) The same dot product three ways; one block, terms t, t + 256, ... per thread, then the tree.
// FP16 accumulator, written as a multiply then an add: nvcc fuses the pair into one HFMA2.
__global__ void dotFP16(const __half* a, const __half* b, float* out, int n) {
    __shared__ __half s[BLOCK];
    __half acc = __float2half(0.0f);
    for (int i = threadIdx.x; i < n; i += BLOCK)
        acc = __hadd(acc, __hmul(a[i], b[i]));           // contracted: a * b + acc, rounded once
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] = __hadd(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = __half2float(s[0]);
}

// FP16 accumulator, product and sum rounded apart: the _rn intrinsics are never contracted.
__global__ void dotFP16rn(const __half* a, const __half* b, float* out, int n) {
    __shared__ __half s[BLOCK];
    __half acc = __float2half(0.0f);
    for (int i = threadIdx.x; i < n; i += BLOCK)
        acc = __hadd_rn(acc, __hmul_rn(a[i], b[i]));     // two roundings to FP16
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] = __hadd(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = __half2float(s[0]);
}

// Mixed precision: FP16 inputs, FP32 accumulator. The product of two FP16 values (11 significant
// bits each) is exact in FP32's 24, so only the additions round.
__global__ void dotMixed(const __half* a, const __half* b, float* out, int n) {
    __shared__ float s[BLOCK];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK)
        acc += __half2float(a[i]) * __half2float(b[i]);
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s[0];
}

static float sumWithBlocks(const float* d_x, float* d_part, int grid) {
    static float h_part[1024];
    sumBlocks<<<grid, BLOCK>>>(d_x, d_part, N_SUM);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_part, d_part, grid * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;                                  // the host adds the block sums in order
    for (int b = 0; b < grid; ++b) total += h_part[b];
    return total;
}

static float dot(void (*kernel)(const __half*, const __half*, float*, int),
                 const __half* d_a, const __half* d_b, float* d_out) {
    float r;
    kernel<<<1, BLOCK>>>(d_a, d_b, d_out, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return r;
}

static double gamma(int n, double u) { return n * u / (1.0 - n * u); }

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    bool ok = true;

    // ---- (b) block count ----
    float* h_x = (float*)malloc(N_SUM * sizeof(float));
    double ref = 0.0;                                    // exact: 2^20 values of 24 bits fit in 53
    for (int i = 0; i < N_SUM; ++i) { h_x[i] = hash01(i, 7); ref += h_x[i]; }
    float *d_x, *d_part;
    CUDA_CHECK(cudaMalloc(&d_x, N_SUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_part, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N_SUM * sizeof(float), cudaMemcpyHostToDevice));
    printf("(b) %d floats in [0, 1), exact sum %.4f\n", N_SUM, ref);
    printf("    blocks x 256 threads   fp32 sum         abs error\n");
    const int grids[4] = {1, 14, 56, 224};
    for (int g = 0; g < 4; ++g) {
        float s = sumWithBlocks(d_x, d_part, grids[g]);
        printf("    %6d x 256          %.4f     %.4f\n", grids[g], s, fabs((double)s - ref));
    }
    float r1 = sumWithBlocks(d_x, d_part, 56), r2 = sumWithBlocks(d_x, d_part, 56);
    printf("    56 blocks twice more: %.4f and %.4f, %s\n", r1, r2,
           memcmp(&r1, &r2, sizeof(float)) == 0 ? "the same bits" : "DIFFERENT bits");

    // ---- (c) conversions ----
    const float vals[3] = {100000.0f, 1e-5f, 1e-8f};
    float *d_v, *d_h, *d_bf, h_h[3], h_bf[3];
    CUDA_CHECK(cudaMalloc(&d_v, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bf, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_v, vals, sizeof(vals), cudaMemcpyHostToDevice));
    convert<<<1, 32>>>(d_v, d_h, d_bf, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_h, d_h, sizeof(h_h), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bf, d_bf, sizeof(h_bf), cudaMemcpyDeviceToHost));
    printf("(c) value       fp16          bf16\n");
    for (int i = 0; i < 3; ++i)
        printf("    %-9g   %-12.6g  %.6g\n", vals[i], h_h[i], h_bf[i]);

    // ---- (d) softmax ----
    const float row[4] = {2.0f, 3.0f, 1000.0f, 1001.0f};
    float *d_row, *d_nv, *d_st, h_nv[4], h_st[4];
    CUDA_CHECK(cudaMalloc(&d_row, sizeof(row)));
    CUDA_CHECK(cudaMalloc(&d_nv, sizeof(row)));
    CUDA_CHECK(cudaMalloc(&d_st, sizeof(row)));
    CUDA_CHECK(cudaMemcpy(d_row, row, sizeof(row), cudaMemcpyHostToDevice));
    softmax<<<1, 1>>>(d_row, d_nv, d_st, 4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_nv, d_nv, sizeof(h_nv), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_st, d_st, sizeof(h_st), cudaMemcpyDeviceToHost));
    printf("(d) naive  softmax: %.4f %.4f %.4f %.4f\n", h_nv[0], h_nv[1], h_nv[2], h_nv[3]);
    printf("    stable softmax: %.4f %.4f %.4f %.4f\n", h_st[0], h_st[1], h_st[2], h_st[3]);

    // ---- (e) mixed precision ----
    __half *h_a = (__half*)malloc(K * sizeof(__half)), *h_b = (__half*)malloc(K * sizeof(__half));
    double dref = 0.0, mag = 0.0;                        // FP64 reference on the FP16 operands
    for (int i = 0; i < K; ++i) {
        h_a[i] = __float2half((hash01(i, 1) - 0.5f) * 0.5f);
        h_b[i] = __float2half((hash01(i, 2) - 0.5f) * 0.5f);
        double t = (double)__half2float(h_a[i]) * (double)__half2float(h_b[i]);
        dref += t; mag += fabs(t);
    }
    __half *d_a, *d_b; float* d_out;
    CUDA_CHECK(cudaMalloc(&d_a, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, K * sizeof(__half), cudaMemcpyHostToDevice));
    const int d = K / BLOCK + 8;                         // additions on the longest path: 16 + 8
    struct { const char* name; void (*k)(const __half*, const __half*, float*, int); double bound; } rows[3] = {
        {"fp16, fused (as compiled)", dotFP16,   gamma(d, ldexp(1.0, -11)) * mag},
        {"fp16, rounded apart (_rn)", dotFP16rn, gamma(d + 1, ldexp(1.0, -11)) * mag},
        {"fp32 (mixed precision)",    dotMixed,  gamma(d, ldexp(1.0, -24)) * mag}};
    printf("(e) dot product, K = %d, FP64 reference %.9f\n", K, dref);
    printf("    accumulator                  result         abs error   bound\n");
    for (int r = 0; r < 3; ++r) {
        float v = dot(rows[r].k, d_a, d_b, d_out);
        double e = fabs((double)v - dref);
        bool in = e <= rows[r].bound;
        ok = ok && in;
        printf("    %-28s %.9f  %.3e   %.3e  %s\n", rows[r].name, v, e, rows[r].bound, in ? "within" : "OUTSIDE");
    }
    printf("%s\n", ok ? "every result within its bound: PASS" : "a result left its bound: FAIL");

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_part)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_h)); CUDA_CHECK(cudaFree(d_bf)); CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_nv)); CUDA_CHECK(cudaFree(d_st)); CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    free(h_x); free(h_a); free(h_b);
    return ok ? 0 : 1;
}
