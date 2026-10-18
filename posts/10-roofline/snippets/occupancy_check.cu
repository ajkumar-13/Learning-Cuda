// occupancy_check.cu: post 10's occupancy rule against the CUDA runtime's own calculator.
//
// Build: nvcc -O3 -arch=sm_75 posts/10-roofline/snippets/occupancy_check.cu -o occupancy_check   (from the series root)
// Run:   ./occupancy_check        (on Windows: .\occupancy_check.exe)
//
// The rule is roofline_model.py's occupancy() for compute capability 7.5: at most 32 resident warps
// and 16 resident blocks per SM; registers allocated per warp, rounded up to 256, from a register
// file of four quarters of 16,384 that each hold whole warps; shared memory allocated per block,
// rounded up to 256 bytes, from 64 KiB per SM. Twenty-three copies of one kernel keep 2 to 130
// values live, so the compiler gives them a range of register counts. For each copy, 40 block sizes
// and 27 dynamic shared-memory sizes (chosen on both sides of a 256-byte unit), the program compares
// the rule's blocks per SM with cudaOccupancyMaxActiveBlocksPerMultiprocessor, prints every
// mismatch, the cases per copy and the total, and counts the cases in which dividing the whole
// register file, without the quarters, would give another answer. No kernel is launched and
// nothing is timed. It exits with status 1 if a CUDA call fails or, on a card of compute
// capability 7.5, if any case mismatches; on another card the rule's constants are not that
// card's, and it only reports.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

// K values stay live until the end, so the register count grows with K. Never launched.
template <int K>
__global__ void pressure(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float dyn[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v[K];
#pragma unroll
    for (int k = 0; k < K; ++k) v[k] = in[(i + k * 977) % n];
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int k = 0; k < K; ++k) acc += v[k] * v[(k * 7 + r) % K];
    if (n < 0) dyn[threadIdx.x] = acc;
    out[i] = acc;
}

// The rule, as roofline_model.py's occupancy() states it for compute capability 7.5.
// quarters = 4 is the rule; quarters = 1 divides the whole register file, the error post 10 warns of.
int ruleBlocksPerSM(int regs, int smem, int threads, int quarters = 4) {
    if (threads > 1024) return 0;
    int warps = (threads + 31) / 32;
    int byBlock = 32 / warps < 16 ? 32 / warps : 16;
    int regsPerWarp = (regs * 32 + 255) / 256 * 256;
    int warpsByRegs = (65536 / quarters) / regsPerWarp * quarters;
    int byRegs = warpsByRegs / warps < 16 ? warpsByRegs / warps : 16;
    int bySmem = 16;
    if (smem > 0) {
        int fit = 65536 / ((smem + 255) / 256 * 256);
        bySmem = fit < 16 ? fit : 16;
    }
    int blocks = byBlock < byRegs ? byBlock : byRegs;
    return blocks < bySmem ? blocks : bySmem;
}

typedef void (*Kernel)(const float*, float*, int);

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    bool cc75 = p.major == 7 && p.minor == 5;
    printf("GPU: %s (compute capability %d.%d): %d registers, %zu bytes of shared memory, "
           "%d threads and %d blocks per SM\n", p.name, p.major, p.minor, p.regsPerMultiprocessor,
           p.sharedMemPerMultiprocessor, p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor);
    if (!cc75) printf("note: the rule is for compute capability 7.5; mismatches here are expected\n");

    const Kernel kernels[] = {pressure<2>, pressure<4>, pressure<6>, pressure<8>, pressure<10>,
                              pressure<12>, pressure<14>, pressure<16>, pressure<18>, pressure<22>,
                              pressure<24>, pressure<26>, pressure<28>, pressure<32>, pressure<48>,
                              pressure<56>, pressure<64>, pressure<72>, pressure<80>, pressure<88>,
                              pressure<100>, pressure<122>, pressure<130>};
    const int live[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 22, 24, 26, 28, 32, 48, 56, 64, 72, 80, 88,
                        100, 122, 130};
    int threads[40], nThreads = 0;
    for (int t = 32; t <= 1024; t += 32) threads[nThreads++] = t;
    const int odd[] = {1, 48, 100, 200, 333, 500, 700, 1000};
    for (int t : odd) threads[nThreads++] = t;
    const int smem[] = {0, 1, 255, 256, 257, 1024, 2048, 4096, 4224, 4352, 4353, 8192, 13100, 13312,
                        13313, 16384, 20608, 21800, 21845, 21846, 32768, 32769, 49152, 49153,
                        65280, 65281, 65536};

    int cases = 0, mismatches = 0, wholeFileDiffers = 0;
    printf("live values  registers  static smem  cases  mismatches\n");
    for (int k = 0; k < (int)(sizeof(kernels) / sizeof(kernels[0])); ++k) {
        cudaFuncAttributes a;
        CUDA_CHECK(cudaFuncGetAttributes(&a, kernels[k]));
        CUDA_CHECK(cudaFuncSetAttribute(kernels[k], cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        65536 - (int)a.sharedSizeBytes));
        int kc = 0, km = 0;
        for (int t = 0; t < nThreads; ++t)
            for (int s : smem) {
                int api = -1;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, kernels[k], threads[t], s));
                int rule = ruleBlocksPerSM(a.numRegs, (int)a.sharedSizeBytes + s, threads[t]);
                ++kc;
                if (ruleBlocksPerSM(a.numRegs, (int)a.sharedSizeBytes + s, threads[t], 1) != rule)
                    ++wholeFileDiffers;
                if (rule != api) {
                    ++km;
                    printf("  MISMATCH: %d registers, %d threads, %d bytes: rule %d, runtime %d\n",
                           a.numRegs, threads[t], s, rule, api);
                }
            }
        printf("%11d  %9d  %11zu  %5d  %10d\n", live[k], a.numRegs, a.sharedSizeBytes, kc, km);
        cases += kc;
        mismatches += km;
    }
    printf("%d cases, %d mismatches between the rule and cudaOccupancyMaxActiveBlocksPerMultiprocessor\n",
           cases, mismatches);
    printf("in %d of the cases, dividing the whole register file would give another answer\n",
           wholeFileDiffers);
    return (cc75 && mismatches) ? EXIT_FAILURE : EXIT_SUCCESS;
}
