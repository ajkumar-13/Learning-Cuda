#include <cuda_runtime.h>
#include <cstdio>

// Declaration of matrix-add solution from solution.cu
extern "C" void solve(const float* A, const float* B, float* C, int N);

int main()
{
    // N is the matrix dimension (N x N)
    const int N = 32;
    const size_t size = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

    // Host arrays (flattened N x N matrices)
    float* hA = new float[N * N];
    float* hB = new float[N * N];
    float* hC = new float[N * N];

    // Initialize inputs with a simple pattern: A[row,col] = row, B[row,col] = col
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            int idx = row * N + col;
            hA[idx] = static_cast<float>(row);
            hB[idx] = static_cast<float>(col);
            hC[idx] = 0.0f;
        }
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

    // Verify: C[row,col] should equal A[row,col] + B[row,col]
    bool ok = true;
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            int idx = row * N + col;
            float expected = hA[idx] + hB[idx];
            if (hC[idx] != expected)
            {
                std::printf("Mismatch at (row=%d, col=%d): got %f, expected %f\n", row, col, hC[idx], expected);
                ok = false;
                break;
            }
        }
        if (!ok)
        {
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
