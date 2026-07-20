// fusion.cu — the same elementwise chain (scale, bias, ReLU) done two ways:
// separate kernels that round-trip the intermediate through global memory, and
// one fused kernel that keeps it in a register and writes once.
//
// Build: nvcc -O3 -arch=sm_75 snippets/fusion.cu -o fusion   (from the post directory)
// Run:   ./fusion

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>   // malloc, free
#include <cmath>     // fmaxf

#define N (1 << 24)

// --- separate: two kernels, an intermediate y written to and read from HBM ---
__global__ void scaleBias(const float* x, float* y, float s, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s + b;        // reads x, writes y (HBM)
}
__global__ void relu(const float* y, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = fmaxf(y[i], 0.0f);   // reads y (HBM again!), writes z
}

// --- fused: one kernel; the intermediate lives and dies in a register ---
__global__ void scaleBiasReluFused(const float* x, float* z, float s, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] * s + b;            // in a register, never written to HBM
        z[i] = fmaxf(v, 0.0f);             // one read of x, one write of z
    }
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h[i] = (float)(i % 7) - 3.0f;

    float *d_x, *d_y, *d_z1, *d_z2;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z1, bytes); cudaMalloc(&d_z2, bytes);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    float s = 2.0f, b = 0.5f;

    scaleBias<<<grid, block>>>(d_x, d_y, s, b, N);     // separate: 2 launches,
    relu<<<grid, block>>>(d_y, d_z1, N);               // y round-trips through HBM
    scaleBiasReluFused<<<grid, block>>>(d_x, d_z2, s, b, N);  // fused: 1 launch
    cudaDeviceSynchronize();

    float a, c;
    cudaMemcpy(&a, d_z1 + 5, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, d_z2 + 5, 4, cudaMemcpyDeviceToHost);
    printf("separate vs fused match: %s  (%.1f == %.1f)\n", a == c ? "yes" : "no", a, c);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z1); cudaFree(d_z2); free(h);
    return 0;
}
