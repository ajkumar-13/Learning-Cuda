#include <cuda_runtime.h>
#include <cstdio>

// Declaration of solution function from solution.cu
extern "C" void solve(const float* A, const float* B, float* C, int N);

int main()
{
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Host arrays
    float* hA = new float[N];
    float* hB = new float[N];
    float* hC = new float[N];

    // Initialize inputs
    for (int i = 0; i < N; ++i)
    {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
        hC[i] = 0.0f;
    }

    // Device arrays
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (cudaMalloc(&dA, size) != cudaSuccess ||
        cudaMalloc(&dB, size) != cudaSuccess ||
        cudaMalloc(&dC, size) != cudaSuccess)
    {
        std::printf("cudaMalloc failed\n");
        delete[] hA;
        delete[] hB;
        delete[] hC;
        return 1;
    }

    // Copy host data to device
    if (cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::printf("cudaMemcpy H2D failed\n");
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        delete[] hA;
        delete[] hB;
        delete[] hC;
        return 1;
    }

    // Calling GPU solve implementation
    solve(dA, dB, dC, N);

    // Copy result back to host
    if (cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::printf("cudaMemcpy D2H failed\n");
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        delete[] hA;
        delete[] hB;
        delete[] hC;
        return 1;
    }

    // Verify: C[i] should equal A[i] + B[i]
    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = hA[i] + hB[i];
        if (hC[i] != expected)
        {
            std::printf("Mismatch at i=%d: got %f, expected %f\n", i, hC[i], expected);
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

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return ok ? 0 : 1;
}
