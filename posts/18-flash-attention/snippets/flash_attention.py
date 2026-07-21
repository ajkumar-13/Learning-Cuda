"""flash_attention.py — a CPU model of FlashAttention's online softmax.

CUDA cannot run in this blog's test environment, so this NumPy model implements
both standard attention (which materializes the full N-by-N score matrix) and
FlashAttention (which tiles over K, V blocks and keeps a running max and sum, an
online softmax, so the N-by-N matrix never exists). It checks they give the same
result, then contrasts the memory that the N-by-N matrix would cost against the
O(N) running statistics FlashAttention keeps.  Run: python flash_attention.py  (needs NumPy)
"""
import numpy as np

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

N, d, Bc = 256, 32, 64
rng = np.random.default_rng(0)
Q = rng.standard_normal((N, d)) * 0.5
K = rng.standard_normal((N, d)) * 0.5
V = rng.standard_normal((N, d)) * 0.5
scale = 1.0 / np.sqrt(d)

# standard: build the whole N-by-N score matrix in memory
O_std = softmax((Q @ K.T) * scale) @ V

# FlashAttention: tile over K/V blocks, keep running max m and sum l (online softmax)
O = np.zeros((N, d)); m = np.full(N, -np.inf); l = np.zeros(N)
for j in range(0, N, Bc):
    Sij = (Q @ K[j:j + Bc].T) * scale            # (N, Bc) block of scores, in SRAM
    mij = Sij.max(-1)
    m_new = np.maximum(m, mij)
    P = np.exp(Sij - m_new[:, None])             # block softmax numerator
    corr = np.exp(m - m_new)                     # rescale the old accumulator
    l = corr * l + P.sum(-1)
    O = corr[:, None] * O + P @ V[j:j + Bc]
    m = m_new
O = O / l[:, None]                               # normalize once, at the end

def nn_mem(n, bytes_per=2):                       # N-by-N matrix size in MB (FP16)
    return n * n * bytes_per / 1e6

print("flash attention model")
print(f"  N={N}, d={d}, block={Bc}")
print(f"  flash == standard   : {bool(np.allclose(O, O_std, atol=1e-5))}  (max err {np.max(np.abs(O - O_std)):.2e})")
print( "  the N-by-N score matrix FlashAttention never materializes:")
for n in (2048, 8192, 32768):
    print(f"    seq {n:>6}: standard {nn_mem(n):>8.1f} MB   vs   flash O(N) running stats")
print("  double the sequence length -> 4x standard memory, but only 2x for flash")
