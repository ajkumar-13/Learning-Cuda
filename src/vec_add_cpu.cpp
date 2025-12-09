#include <iostream>
#include <vector>

int main(){
    int n = 1<<20; // 1M elements
    std::vector<float> A(n, 1.0f), B(n, 2.0f), C(n);
    for (int i=0;i<n;i++) C[i] = A[i] + B[i];

    bool ok = true;
    for (int i=0;i<10;i++) if (C[i] != 3.0f) { ok = false; break; }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
