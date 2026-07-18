// convolution.cu — 2D convolution (image stencil) three ways: naive global,
// constant memory for the filter, and shared-memory tiling with a halo.
//
// Build: nvcc -O3 -arch=sm_75 snippets/convolution.cu -o convolution   (from the post directory)
// Run:   ./convolution

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>   // malloc, free

#define MASK 3
#define R    (MASK / 2)            // filter radius = 1 for a 3x3 filter
#define TILE 16
#define BLK  (TILE + MASK - 1)     // 18: output tile plus the halo

__constant__ float c_mask[MASK * MASK];   // file scope, broadcast read-only

// naive: filter read from global, neighbours re-read overlapping pixels
__global__ void convNaive(const float* in, const float* mask, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < MASK; ++i)
            for (int j = 0; j < MASK; ++j) {
                int r = row + i - R, c = col + j - R;
                if (r >= 0 && r < H && c >= 0 && c < W)
                    sum += in[r * W + c] * mask[i * MASK + j];
            }
        out[row * W + col] = sum;
    }
}

// constant memory: the filter is broadcast from a cached read-only space
__global__ void convConst(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < MASK; ++i)
            for (int j = 0; j < MASK; ++j) {
                int r = row + i - R, c = col + j - R;
                if (r >= 0 && r < H && c >= 0 && c < W)
                    sum += in[r * W + c] * c_mask[i * MASK + j];  // broadcast
            }
        out[row * W + col] = sum;
    }
}

// tiled: BLK x BLK threads load an (output + halo) tile into shared memory once;
// only the inner TILE x TILE threads compute, reading neighbours from shared.
__global__ void convTiled(const float* in, float* out, int W, int H) {
    __shared__ float s[BLK][BLK];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx - R;       // shifted by the halo radius
    int row = blockIdx.y * TILE + ty - R;
    s[ty][tx] = (row >= 0 && row < H && col >= 0 && col < W) ? in[row * W + col] : 0.0f;
    __syncthreads();

    if (tx >= R && tx < BLK - R && ty >= R && ty < BLK - R) {  // inner threads only
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < MASK; ++i)
            #pragma unroll
            for (int j = 0; j < MASK; ++j)
                sum += s[ty + i - R][tx + j - R] * c_mask[i * MASK + j];
        int oc = blockIdx.x * TILE + (tx - R), orow = blockIdx.y * TILE + (ty - R);
        if (orow < H && oc < W) out[orow * W + oc] = sum;
    }
}

int main() {
    const int W = 1920, H = 1080;
    size_t bytes = (size_t)W * H * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < W * H; ++i) h[i] = (float)(i % 256);
    float box[MASK * MASK]; for (int i = 0; i < MASK * MASK; ++i) box[i] = 1.0f / 9.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, box, sizeof(box));        // not regular cudaMemcpy

    dim3 block(BLK, BLK);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);  // grid sized by OUTPUT tiles
    convTiled<<<grid, block>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    printf("convolution launched: %dx%d, 3x3 box blur, tiled+halo+constant\n", W, H);

    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
