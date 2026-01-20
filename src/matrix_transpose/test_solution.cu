#include <cuda_runtime.h>
#include <cstdio>

// Declaration of transpose solution from solution.cu
extern "C" void solve(const float* input, float* output, int rows, int cols);

// Optional file-based test.
// Each test case format:
//   rows cols
//   input[0..rows*cols-1]
//   expected[0..rows*cols-1]   (flattened transpose, size rows*cols)
// Multiple test cases can be concatenated in the same file.
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
        int rows = 0, cols = 0;
        if (std::fscanf(f, "%d %d", &rows, &cols) != 2)
        {
            break; // no more cases
        }
        if (rows <= 0 || cols <= 0)
        {
            std::printf("Invalid rows/cols (%d,%d) in case %d in %s\n", rows, cols, caseIndex, path);
            allOk = false;
            break;
        }

        const int elems = rows * cols;
        const size_t size = static_cast<size_t>(elems) * sizeof(float);
        float* hInput = new float[elems];
        float* hOutput = new float[elems];
        float* hExpected = new float[elems];

        bool readOk = true;
        for (int i = 0; i < elems; ++i)
        {
            if (std::fscanf(f, "%f", &hInput[i]) != 1)
            {
                std::printf("Failed to read input[%d] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
        }
        for (int i = 0; readOk && i < elems; ++i)
        {
            if (std::fscanf(f, "%f", &hExpected[i]) != 1)
            {
                std::printf("Failed to read expected[%d] in case %d from %s\n", i, caseIndex, path);
                readOk = false;
                break;
            }
            hOutput[i] = 0.0f;
        }

        if (!readOk)
        {
            delete[] hInput; delete[] hOutput; delete[] hExpected;
            allOk = false;
            break;
        }

        float *dInput = nullptr, *dOutput = nullptr;
        if (cudaMalloc(&dInput, size) != cudaSuccess ||
            cudaMalloc(&dOutput, size) != cudaSuccess)
        {
            std::printf("cudaMalloc failed in case %d\n", caseIndex);
            delete[] hInput; delete[] hOutput; delete[] hExpected;
            allOk = false;
            break;
        }

        if (cudaMemcpy(dInput, hInput, size, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            std::printf("cudaMemcpy H2D failed in case %d\n", caseIndex);
            cudaFree(dInput); cudaFree(dOutput);
            delete[] hInput; delete[] hOutput; delete[] hExpected;
            allOk = false;
            break;
        }

        solve(dInput, dOutput, rows, cols);

        if (cudaMemcpy(hOutput, dOutput, size, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::printf("cudaMemcpy D2H failed in case %d\n", caseIndex);
            cudaFree(dInput); cudaFree(dOutput);
            delete[] hInput; delete[] hOutput; delete[] hExpected;
            allOk = false;
            break;
        }

        bool ok = true;
        for (int i = 0; i < elems; ++i)
        {
            if (hOutput[i] != hExpected[i])
            {
                std::printf("File test mismatch in case %d at flat index %d: got %f, expected %f\n",
                    caseIndex, i, hOutput[i], hExpected[i]);
                ok = false;
                break;
            }
        }

        cudaFree(dInput); cudaFree(dOutput);
        delete[] hInput; delete[] hOutput; delete[] hExpected;

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

    const int rows = 3;
    const int cols = 4;
    const int inElems = rows * cols;
    const size_t size = static_cast<size_t>(inElems) * sizeof(float);

    // Host arrays (flattened row-major matrices)
    float* hInput = new float[inElems];
    float* hOutput = new float[inElems]; 

    // Initialize input matrix (built-in test)
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int idx = r * cols + c;
            hInput[idx] = static_cast<float>(r * 10 + c);
            hOutput[idx] = 0.0f;
        }
    }

    // Device arrays
    float *dInput = nullptr, *dOutput = nullptr;
    if (cudaMalloc(&dInput, size) != cudaSuccess ||
        cudaMalloc(&dOutput, size) != cudaSuccess)
    {
        std::printf("cudaMalloc failed\n");
        delete[] hInput;
        delete[] hOutput;
        return 1;
    }

    // Copy host data to device
    if (cudaMemcpy(dInput, hInput, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::printf("cudaMemcpy H2D failed\n");
        cudaFree(dInput);
        cudaFree(dOutput);
        delete[] hInput;
        delete[] hOutput;
        return 1;
    }

    // Call GPU transpose implementation
    solve(dInput, dOutput, rows, cols);

    // Copy result back to host
    if (cudaMemcpy(hOutput, dOutput, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::printf("cudaMemcpy D2H failed\n");
        cudaFree(dInput);
        cudaFree(dOutput);
        delete[] hInput;
        delete[] hOutput;
        return 1;
    }

    // Verify: output[c,r] should equal input[r,c]
    bool ok = true;
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int inIdx  = r * cols + c;
            int outIdx = c * rows + r; // transposed index
            float expected = hInput[inIdx];
            if (hOutput[outIdx] != expected)
            {
                std::printf("Mismatch at (row=%d, col=%d): got %f, expected %f\n",
                    c, r, hOutput[outIdx], expected);
                ok = false;
                break;
            }
        }
        if (!ok)
        {
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

    cudaFree(dInput);
    cudaFree(dOutput);
    delete[] hInput;
    delete[] hOutput;

    return ok ? 0 : 1;
}
