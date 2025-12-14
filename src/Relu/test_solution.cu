#include <cuda_runtime.h>
#include <cstdio>

// Declaration of solution function from solution.cu
__global__ void relu_kernel(const float* input, float* output, int N);

int main()
{
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Host arrays
    float* hinput = new float[N];
    float* houtput = new float[N];


    // Initialize inputs
    for (int i = 0; i < N; ++i)
    {
        if (i < N / 3) {
            hinput[i] = -static_cast<float>(i + 1);   
        } else if (i == N / 3) {
            hinput[i] = 0.0f;                         
        } else {
            hinput[i] = static_cast<float>(i - N / 3); 
        }

        houtput[i] = 0.0f;
    }

    // Device arrays
    float *dinput = nullptr, *doutput = nullptr;
    if (cudaMalloc(&dinput, size) != cudaSuccess ||
        cudaMalloc(&doutput, size) != cudaSuccess)
    {
        std::printf("cudaMalloc failed\n");
        delete[] hinput;
        delete[] houtput;
        return 1;
    }

    // Copy host data to device
    if (cudaMemcpy(dinput, hinput, size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(doutput, houtput, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::printf("cudaMemcpy H2D failed\n");
        cudaFree(dinput);
        cudaFree(doutput);
        delete[] hinput;
        delete[] houtput;
        return 1;
    }

    // Calling GPU relu_kernel implementation
    relu_kernel<<<(N + 255) / 256, 256>>>(dinput, doutput, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    if (cudaMemcpy(houtput, doutput, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::printf("cudaMemcpy D2H failed\n");
        cudaFree(dinput);
        cudaFree(doutput);
        delete[] hinput;
        delete[] houtput;
        return 1;
    }

    // Verify: output[i] should be max(input[i], 0)
    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = hinput[i] > 0 ? hinput[i] : 0.0f;
        if (houtput[i] != expected)
        {
            std::printf("Mismatch at i=%d: got %f, expected %f\n", i, houtput[i], expected);
            ok = false;
            break;
        }
    }

    if (ok)
    {
        std::printf("solution.cu: PASS\n");
    }
    else
    {
        std::printf("solution.cu: FAIL\n");
    }

    cudaFree(dinput);
    cudaFree(doutput);
    delete[] hinput;
    delete[] houtput;


    return ok ? 0 : 1;
}
