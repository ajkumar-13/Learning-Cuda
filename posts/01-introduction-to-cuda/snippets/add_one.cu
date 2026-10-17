// add_one.cu: the first kernel of the series. Each thread adds 1 to one element, and the guard
// i < N stops the surplus threads of the last block.
//
// Build: nvcc -O3 -arch=sm_75 posts/01-introduction-to-cuda/snippets/add_one.cu -o add_one   (from the series root)
// Run:   ./add_one        (on Windows: .\add_one.exe)
//
// The host side (cudaMalloc, cudaMemcpy, cudaFree) is the subject of post 02; here it only moves
// N floats to the device and back, so the program can check that every element was touched once.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void addOne(float *data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < N)                                       // guard the surplus threads
        data[i] += 1.0f;                             // one thread, one element
}

int main()
{
    const int N = 1000, threads = 256;
    const int blocks = (N + threads - 1) / threads;   // ceil(N / threads)

    float h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = (float)i;

    float *d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));           // exactly N floats: no room for the surplus
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    addOne<<<blocks, threads>>>(d_data, N);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    int wrong = 0;
    for (int i = 0; i < N; ++i) wrong += (h_data[i] != (float)(i + 1));
    printf("addOne<<<%d, %d>>> over N = %d: %d threads, %d surplus\n",
           blocks, threads, N, blocks * threads, blocks * threads - N);
    printf("elements incremented exactly once: %d of %d\n", N - wrong, N);
    return wrong == 0 ? 0 : 1;
}
