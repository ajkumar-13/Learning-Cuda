#include <cuda_runtime.h>
#include <cstdio>

// Simple kernel: C[i] = factor * A[i]
__global__ void multiplyKernel(const float* a, float* c, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = factor * a[i];
    }
}

int main()
{
    const int N = 1 << 20; // 1,048,576 elements
    const float factor = 2.0f;

    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_a = new float[N];
    float* h_c = new float[N];

    // Initialize input data on host
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_c[i] = 0.0f;
    }

    // Allocate device memory
    float *d_a = nullptr, *d_c = nullptr;
    if (cudaMalloc(&d_a, size) != cudaSuccess || cudaMalloc(&d_c, size) != cudaSuccess)
    {
        std::printf("cudaMalloc failed\n");
        delete[] h_a;
        delete[] h_c;
        return 1;
    }

    // Copy input data to device
    if (cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::printf("cudaMemcpy H2D failed\n");
        cudaFree(d_a);
        cudaFree(d_c);
        delete[] h_a;
        delete[] h_c;
        return 1;
    }

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    multiplyKernel<<<gridSize, blockSize>>>(d_a, d_c, factor, N);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        std::printf("Kernel execution failed\n");
        cudaFree(d_a);
        cudaFree(d_c);
        delete[] h_a;
        delete[] h_c;
        return 1;
    }

    // Copy results back to host
    if (cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::printf("cudaMemcpy D2H failed\n");
        cudaFree(d_a);
        cudaFree(d_c);
        delete[] h_a;
        delete[] h_c;
        return 1;
    }

    // Verify results (check a few elements)
    bool ok = true;
    for (int i = 0; i < 10; ++i)
    {
        float expected = factor * h_a[i];
        if (h_c[i] != expected)
        {
            ok = false;
            break;
        }
    }

    if (ok)
    {
        std::printf("multiply.cu: PASS\n");
    }
    else
    {
        std::printf("multiply.cu: FAIL\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_c;

    return ok ? 0 : 1;
}
