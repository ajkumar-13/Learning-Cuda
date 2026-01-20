#include <cuda_runtime.h>
#include <cstdio>

// Declaration of matrix-add solution from solution.cu
extern "C" void solve(const float* A, const float* B, float* C, int N);

// Optional file-based test for square matrices.
// Each test case format:
//   N
//   A[0..N*N-1]
//   B[0..N*N-1]
//   C_expected[0..N*N-1]


static bool run_test_from_file(const char* path)
{
    std::FILE* f = std::fopen(path, "r");
    if (!f)
    {
        std::printf("Could not open test file: %s\n", path);
        return false;
    }

    bool allOk = true;
    int caseIndex = 0;

    while (true)
    {
        int N = 0;
        if (std::fscanf(f, "%d", &N) != 1)
        {
            break; // no more cases
        }
        if (N <= 0)
        {
            std::printf("Invalid N (%d) in case %d in %s\n", N, caseIndex, path);
            allOk = false;
            break;
        }

        const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
        const size_t size = elems * sizeof(float);
        float* hA = new float[elems];
        float* hB = new float[elems];
        float* hC = new float[elems];
        float* hExpected = new float[elems];

        bool readOk = true;
        for (size_t i = 0; i < elems; ++i)
        {
            if (std::fscanf(f, "%f", &hA[i]) != 1)
            {
                std::printf("Failed to read A[%zu] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
        }
        for (size_t i = 0; readOk && i < elems; ++i)
        {
            if (std::fscanf(f, "%f", &hB[i]) != 1)
            {
                std::printf("Failed to read B[%zu] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
        }
        for (size_t i = 0; readOk && i < elems; ++i)
        {
            if (std::fscanf(f, "%f", &hExpected[i]) != 1)
            {
                std::printf("Failed to read C_expected[%zu] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
            hC[i] = 0.0f;
        }

        if (!readOk)
        {
            delete[] hA; delete[] hB; delete[] hC; delete[] hExpected;
            allOk = false;
            break;
        }

        // Device arrays
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        if (cudaMalloc(&dA, size) != cudaSuccess ||
            cudaMalloc(&dB, size) != cudaSuccess ||
            cudaMalloc(&dC, size) != cudaSuccess)
        {
            std::printf("cudaMalloc failed in case %d\n", caseIndex);
            delete[] hA; delete[] hB; delete[] hC; delete[] hExpected;
            allOk = false;
            break;
        }

        if (cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            std::printf("cudaMemcpy H2D failed in case %d\n", caseIndex);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            delete[] hA; delete[] hB; delete[] hC; delete[] hExpected;
            allOk = false;
            break;
        }

        solve(dA, dB, dC, N);

        if (cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::printf("cudaMemcpy D2H failed in case %d\n", caseIndex);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            delete[] hA; delete[] hB; delete[] hC; delete[] hExpected;
            allOk = false;
            break;
        }

        bool ok = true;
        for (size_t i = 0; i < elems; ++i)
        {
            if (hC[i] != hExpected[i])
            {
                std::printf("File test mismatch in case %d at flat index %zu: got %f, expected %f\n",
                    caseIndex, i, hC[i], hExpected[i]);
                ok = false;
                break;
            }
        }

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC; delete[] hExpected;

        if (!ok)
        {
            allOk = false;
            break;
        }

        ++caseIndex;
    }

    std::fclose(f);

    if (caseIndex == 0)
    {
        std::printf("No test cases found in %s\n", path);
        return false;
    }

    std::printf(allOk ? "File tests: ALL CASES PASS\n" : "File tests: AT LEAST ONE CASE FAILED\n");
    return allOk;
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        if (!run_test_from_file(argv[1]))
        {
            return 1;
        }
        return 0;
    }

    // N is the matrix dimension (N x N)
    const int N = 32;
    const size_t size = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

    // Host arrays (flattened N x N matrices)
    float* hA = new float[N * N];
    float* hB = new float[N * N];
    float* hC = new float[N * N];

    // Initialize inputs: A[row,col] = row, B[row,col] = col (built-in test)
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
