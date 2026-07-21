// flash_attention.cu — an illustrative FlashAttention forward kernel EXCERPT.
// One thread block handles a block of Q rows; it loops over K/V blocks,
// computing scores in shared memory and folding them into a running
// (max, sum, output) with an online softmax. The N-by-N score matrix is never
// written to global memory.
//
// NOTE: this is an EXCERPT, not a runnable program. It has no main(), and the
// cooperative load of Kj/Vj into shared memory is left as a stub (see below),
// so it will not link into an executable. Compile-check only, from snippets/:
//   nvcc -arch=sm_80 -c snippets/flash_attention.cu -o /dev/null
// The runnable, verified artifact for this post is the CPU model:
//   python snippets/flash_attention.py   (needs NumPy)
// (Illustrative: a production kernel uses Tensor Cores for the two matmuls;
//  this excerpt uses plain scalar FMA loops for clarity.)

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

#define BM 64      // rows of Q per block
#define BN 64      // rows of K/V per inner step
#define BD 64      // head dimension

__global__ void flashAttention(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               const float* __restrict__ V,
                               float* __restrict__ O, int N, int d, float scale) {
    __shared__ float Kj[BN][BD], Vj[BN][BD], Sij[BM][BN];
    int row = blockIdx.x * BM + threadIdx.y;          // this thread's Q row
    if (row >= N) return;

    float mi = -FLT_MAX, li = 0.0f;                   // running max and sum (online softmax)
    float Oi[BD];
    for (int k = 0; k < d; ++k) Oi[k] = 0.0f;         // running output, in registers

    for (int j = 0; j < N; j += BN) {                 // loop over K/V blocks
        // cooperatively load Kj, Vj into shared memory (omitted for brevity)
        __syncthreads();

        float mij = -FLT_MAX;                          // block: scores, then block max
        for (int c = 0; c < BN; ++c) {
            float s = 0.0f;
            for (int k = 0; k < d; ++k) s += Q[row * d + k] * Kj[c][k];
            Sij[threadIdx.y][c] = s * scale;
            mij = fmaxf(mij, Sij[threadIdx.y][c]);
        }
        float mNew = fmaxf(mi, mij);                   // online-softmax update
        float corr = __expf(mi - mNew);                // rescale old accumulator
        float lij = 0.0f;
        for (int c = 0; c < BN; ++c) {
            float p = __expf(Sij[threadIdx.y][c] - mNew);
            Sij[threadIdx.y][c] = p;
            lij += p;
        }
        li = corr * li + lij;
        for (int k = 0; k < d; ++k) {                  // O = corr*O + P @ Vj
            float pv = 0.0f;
            for (int c = 0; c < BN; ++c) pv += Sij[threadIdx.y][c] * Vj[c][k];
            Oi[k] = corr * Oi[k] + pv;
        }
        mi = mNew;
        __syncthreads();
    }
    for (int k = 0; k < d; ++k) O[row * d + k] = Oi[k] / li;   // normalize once, write O
}
