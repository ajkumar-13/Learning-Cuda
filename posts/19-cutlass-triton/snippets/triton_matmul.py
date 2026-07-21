"""triton_matmul.py — a tiled matrix multiply in Triton (reference).

This is the Triton version of the tiled GEMM from post 03. You program at the
*block* level: tl.program_id picks an output tile, tl.arange builds vectors of
offsets, and tl.dot does the tile multiply (on Tensor Cores when available).
There is no threadIdx, no __syncthreads, no manual shared-memory staging — the
Triton compiler generates all of that. @triton.autotune times several block
configs and keeps the fastest.

Needs a CUDA GPU and `pip install triton`; it does not run in this blog's test
environment. The runnable companion is tools_model.py, which reproduces this
exact block-tiled accumulation in NumPy and checks it.
"""
import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)        # accumulate in registers

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)                                     # Tensor Cores if available
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk

    c_ptrs = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](A, B, C, M, N, K,
                        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                        C.stride(0), C.stride(1))
    return C
