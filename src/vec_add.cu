#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(){
    int n = 1<<20; // 1M elements
    size_t bytes = n * sizeof(float);
    std::vector<float> hA(n, 1.0f), hB(n, 2.0f), hC(n);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (cudaMalloc(&dA, bytes) != cudaSuccess) { std::cerr << "cudaMalloc dA failed\n"; return 1; }
    if (cudaMalloc(&dB, bytes) != cudaSuccess) { std::cerr << "cudaMalloc dB failed\n"; cudaFree(dA); return 1; }
    if (cudaMalloc(&dC, bytes) != cudaSuccess) { std::cerr << "cudaMalloc dC failed\n"; cudaFree(dA); cudaFree(dB); return 1; }

    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost);

    // quick verify
    bool ok = true;
    for (int i=0;i<10;i++){
        if (hC[i] != 3.0f) { ok=false; break; }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok ? 0 : 1;
}
