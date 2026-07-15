#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}