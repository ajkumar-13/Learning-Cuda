#include <cuda_runtime.h>
#include <cstdio>

// Declaration of solution function from solution.cu
extern "C" void solve(const float* A, const float* B, float* C, int N);


// Optional: run one or more tests loaded from a text file.
// Each test case has the format:
//   N
//   A[0] A[1] ... A[N-1]
//   B[0] B[1] ... B[N-1]
//   C_expected[0] ... C_expected[N-1]
// Multiple test cases can be concatenated back-to-back in the same file with each test case separated by newline.
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
            break; // no more test cases
        }
        if (N <= 0)
        {
            std::printf("Invalid N (%d) in case %d in %s\n", N, caseIndex, path);
            allOk = false;
            break;
        }

        const size_t size = static_cast<size_t>(N) * sizeof(float);
        float* hA = new float[N];
        float* hB = new float[N];
        float* hC = new float[N];
        float* hExpected = new float[N];

        bool readOk = true;
        for (int i = 0; i < N; ++i)
        {
            if (std::fscanf(f, "%f", &hA[i]) != 1)
            {
                std::printf("Failed to read A[%d] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
        }
        for (int i = 0; readOk && i < N; ++i)
        {
            if (std::fscanf(f, "%f", &hB[i]) != 1)
            {
                std::printf("Failed to read B[%d] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
        }
        for (int i = 0; readOk && i < N; ++i)
        {
            if (std::fscanf(f, "%f", &hExpected[i]) != 1)
            {
                std::printf("Failed to read C_expected[%d] in case %d from %s\n", i, caseIndex, path);
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
        for (int i = 0; i < N; ++i)
        {
            if (hC[i] != hExpected[i])
            {
                std::printf("File test mismatch in case %d at i=%d: got %f, expected %f\n",
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
            // we are stoping on first failure
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
    // If a test file is provided, we will try that first.
    if (argc > 1)
    {
        if (!run_test_from_file(argv[1]))
        {
            return 1;
        }
        return 0;
    }

    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Host arrays
    float* hA = new float[N];
    float* hB = new float[N];
    float* hC = new float[N];

    // Initialize inputs (built-in test)
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
