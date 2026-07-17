// transpose.cu — transpose a matrix three ways: naive (strided writes),
// shared-memory tiled (coalesced both ways, but bank conflicts), and tiled with
// a padded tile (conflict-free). Pure data movement: no arithmetic to optimize.
//
// Build: nvcc -O3 -arch=sm_75 snippets/transpose.cu -o transpose   (from the post dir)
// Run:   ./transpose

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TILE 32

// naive: reads a row coalesced, but writes a column strided (each thread's
// store is `height` floats from its neighbour's) -> up to 32 transactions/warp.
__global__ void transposeNaive(float* out, const float* in, int W, int H) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H)
        out[x * H + y] = in[y * W + x];          // strided write: BAD
}

// tiled: load a tile coalesced into shared memory, then write it out
// transposed, also coalesced. The pad of +1 column makes the column read of
// shared memory hit 32 different banks instead of all bank 0.
__global__ void transposeTiled(float* out, const float* in, int W, int H) {
    __shared__ float tile[TILE][TILE + 1];       // +1 pad: conflict-free
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H)
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];   // coalesced read
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;         // transposed block coords
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < H && y < W)
        out[y * H + x] = tile[threadIdx.x][threadIdx.y];  // coalesced write
}

int main() {
    const int W = 4096, H = 4096;
    size_t bytes = (size_t)W * H * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < W * H; ++i) h[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    transposeTiled<<<grid, block>>>(d_out, d_in, W, H);   // swap for transposeNaive
    cudaDeviceSynchronize();

    float* hb = (float*)malloc(bytes);
    cudaMemcpy(hb, d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = hb[1 * H + 0] == h[0 * W + 1];   // out[1][0] should equal in[0][1]
    printf("transpose %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out); free(h); free(hb);
    return 0;
}
