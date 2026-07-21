// tensor_core.cu — a matrix multiply on Tensor Cores via the WMMA API.
// One warp cooperatively computes a 16x16x16 tile: D = A*B + C, with FP16
// inputs and an FP32 accumulator (mixed precision).
//
// Build: nvcc -O3 -arch=sm_75 snippets/tensor_core.cu -o tensor_core
// Run:   ./tensor_core   (needs a Volta-or-newer GPU, compute capability 7.0+)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each warp produces one 16x16 output tile. A is row-major FP16, B col-major
// FP16, the accumulator C/D is FP32. The mma_sync line is 16*16*16 = 4096 FMAs.
__global__ void tcMatmul(const half* A, const half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;   // warp row
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;          // warp col
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);                          // FP32 accumulator

    for (int k = 0; k < K; k += WMMA_K) {
        const half* aTile = A + warpM * WMMA_M * K + k;
        const half* bTile = B + k * N + warpN * WMMA_N;
        wmma::load_matrix_sync(a_frag, aTile, K);              // all 32 lanes load
        wmma::load_matrix_sync(b_frag, bTile, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);        // one instruction, 4096 FMAs
    }
    float* cTile = C + warpM * WMMA_M * N + warpN * WMMA_N;
    wmma::store_matrix_sync(cTile, c_frag, N, wmma::mem_row_major);
}

// Fill an FP16 buffer with a constant, so the product has a known reference.
__global__ void fillHalf(half* out, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(v);
}

int main() {
    const int M = 4096, N = 4096, K = 4096;         // all multiples of 16
    size_t e = (size_t)M * K;
    half *dA, *dB; float *dC;
    cudaMalloc(&dA, e * 2); cudaMalloc(&dB, e * 2); cudaMalloc(&dC, (size_t)M * N * 4);

    // Known pattern: A = B = 0.5 everywhere, so every C[i] = K * 0.5 * 0.5.
    const float a_val = 0.5f, b_val = 0.5f;
    fillHalf<<<(e + 255) / 256, 256>>>(dA, a_val, e);
    fillHalf<<<(e + 255) / 256, 256>>>(dB, b_val, e);

    dim3 block(128, 4), grid((M + 16 * 4 - 1) / (16 * 4), (N + 16 * 4 - 1) / (16 * 4));
    tcMatmul<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // Copy the first output element back and check it against the reference.
    float c0 = 0.0f;
    cudaMemcpy(&c0, dC, sizeof(float), cudaMemcpyDeviceToHost);
    float ref = (float)K * a_val * b_val;           // 4096 * 0.5 * 0.5 = 1024
    printf("tensor-core matmul %dx%dx%d (FP16 in, FP32 accumulate)\n", M, N, K);
    printf("  C[0] = %.3f, reference = %.3f, abs error = %.3e\n",
           c0, ref, fabsf(c0 - ref));

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
