#include <cuda_runtime.h>
#include <cstdio>

// Declaration of reverse-array solution from solution.cu
extern "C" void solve(float* input, int N);

// Optional file-based test.
// Each test case format:
//   N
//   input[0..N-1]
//   expected[0..N-1]
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

		const size_t size = static_cast<size_t>(N) * sizeof(float);
		float* hInput = new float[N];
		float* hExpected = new float[N];

		bool readOk = true;
		for (int i = 0; i < N; ++i)
		{
			if (std::fscanf(f, "%f", &hInput[i]) != 1)
			{
				std::printf("Failed to read input[%d] in case %d from %s\n", i, caseIndex, path);
				readOk = false;
				break;
			}
		}
		for (int i = 0; readOk && i < N; ++i)
		{
			if (std::fscanf(f, "%f", &hExpected[i]) != 1)
			{
				std::printf("Failed to read expected[%d] in case %d from %s\n", i, caseIndex, path);
				readOk = false;
				break;
			}
		}

		if (!readOk)
		{
			delete[] hInput; delete[] hExpected;
			allOk = false;
			break;
		}

		float* dInput = nullptr;
		if (cudaMalloc(&dInput, size) != cudaSuccess)
		{
			std::printf("cudaMalloc failed in case %d\n", caseIndex);
			delete[] hInput; delete[] hExpected;
			allOk = false;
			break;
		}

		if (cudaMemcpy(dInput, hInput, size, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			std::printf("cudaMemcpy H2D failed in case %d\n", caseIndex);
			cudaFree(dInput);
			delete[] hInput; delete[] hExpected;
			allOk = false;
			break;
		}

		solve(dInput, N);

		if (cudaMemcpy(hInput, dInput, size, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			std::printf("cudaMemcpy D2H failed in case %d\n", caseIndex);
			cudaFree(dInput);
			delete[] hInput; delete[] hExpected;
			allOk = false;
			break;
		}

		bool ok = true;
		for (int i = 0; i < N; ++i)
		{
			if (hInput[i] != hExpected[i])
			{
				std::printf("File test mismatch in case %d at i=%d: got %f, expected %f\n",
					caseIndex, i, hInput[i], hExpected[i]);
				ok = false;
				break;
			}
		}

		cudaFree(dInput);
		delete[] hInput; delete[] hExpected;

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
	// If a file path is provided, run the file-based test
	if (argc > 1)
	{
		if (!run_test_from_file(argv[1]))
		{
			return 1;
		}
		return 0;
	}

	const int N = 7; // small odd length to test center element behavior
	const size_t size = static_cast<size_t>(N) * sizeof(float);

	// Host array
	float* hInput = new float[N];

	// Initialize input with a simple pattern: [0.0, 1.0, 2.0, ...]
	for (int i = 0; i < N; ++i)
	{
		hInput[i] = static_cast<float>(i);
	}

	// Device array
	float* dInput = nullptr;
	if (cudaMalloc(&dInput, size) != cudaSuccess)
	{
		std::printf("cudaMalloc failed\n");
		delete[] hInput;
		return 1;
	}

	// Copy host data to device
	if (cudaMemcpy(dInput, hInput, size, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::printf("cudaMemcpy H2D failed\n");
		cudaFree(dInput);
		delete[] hInput;
		return 1;
	}

	// Call GPU reverse implementation (in-place on device array)
	solve(dInput, N);

	// Copy result back to host
	if (cudaMemcpy(hInput, dInput, size, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::printf("cudaMemcpy D2H failed\n");
		cudaFree(dInput);
		delete[] hInput;
		return 1;
	}

	// Verify: input[i] should now equal original value at index N-1-i
	bool ok = true;
	for (int i = 0; i < N; ++i)
	{
		float expected = static_cast<float>(N - 1 - i);
		if (hInput[i] != expected)
		{
			std::printf("Mismatch at i=%d: got %f, expected %f\n", i, hInput[i], expected);
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

	cudaFree(dInput);
	delete[] hInput;

	return ok ? 0 : 1;
}

