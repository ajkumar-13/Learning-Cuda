#include <cuda_runtime.h>
#include <cstdio>

// Declaration of transpose solution from solution.cu
extern "C" void solve(const float* input, float* output, int rows, int cols);

int main()
{
    const int rows = 3;
    const int cols = 4;
    const int inElems = rows * cols;
    const size_t size = static_cast<size_t>(inElems) * sizeof(float);

    // Host arrays (flattened row-major matrices)
    float* hInput = new float[inElems];
    float* hOutput = new float[inElems]; 

    // Initialize input matrix
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
