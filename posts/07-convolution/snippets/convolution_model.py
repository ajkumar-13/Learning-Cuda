"""convolution_model.py: a CPU model of the convolution kernels of post 07, and of the counts behind them.

The model runs the shared-memory tile with its halo on the CPU and checks it against a direct
zero-padded reference, then counts what the GPU timing cannot show:

1. Global reads for one 16 x 16 output tile. The naive kernel reads a K x K neighborhood for every
   output pixel, TILE^2 * K^2 reads; the tiled kernel loads the (TILE + K - 1)^2 input tile once into
   shared memory. The ratio is the reuse factor.
2. The threads of a tiled block: it launches one thread per input pixel of the tile, and only the
   inner TILE x TILE of them compute an output.
3. Multiply-adds per pixel: K^2 for the 2D filter, 2K for its separable form (a row pass and a
   column pass), and the separable form of the box filter checked against the 2D one.
4. Constant memory: how many 4-byte filter weights fit in its 64 KB, and the largest odd K x K filter.

Run from the series root:

    uv run python posts/07-convolution/snippets/convolution_model.py
"""
import numpy as np

TILE = 16
CONSTANT_BYTES = 64 * 1024                   # the constant memory of every CUDA device


def reference(img, mask):
    """Direct cross-correlation with zero padding: out[y][x] = sum in[y+i-r][x+j-r] * mask[i][j]."""
    kh, kw = mask.shape                      # a 1 x K or K x 1 mask is one pass of a separable filter
    pad = np.pad(img, ((kh // 2, kh // 2), (kw // 2, kw // 2)))
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += pad[i:i + img.shape[0], j:j + img.shape[1]] * mask[i, j]
    return out


def tiled(img, mask):
    """convTiled on the CPU: per block, load a (TILE + K - 1)^2 tile with zeros outside the image, then
    let only the inner TILE x TILE threads compute, reading their neighbors from the tile."""
    k = mask.shape[0]
    r = k // 2
    blk = TILE + k - 1
    h, w = img.shape
    out = np.zeros_like(img)
    for by in range((h + TILE - 1) // TILE):
        for bx in range((w + TILE - 1) // TILE):
            s = np.zeros((blk, blk), dtype=img.dtype)
            for ty in range(blk):                # every thread of the block loads one pixel
                for tx in range(blk):
                    row, col = by * TILE + ty - r, bx * TILE + tx - r
                    if 0 <= row < h and 0 <= col < w:
                        s[ty, tx] = img[row, col]
            for ty in range(r, blk - r):         # inner threads only
                for tx in range(r, blk - r):
                    orow, ocol = by * TILE + ty - r, bx * TILE + tx - r
                    if orow < h and ocol < w:
                        out[orow, ocol] = np.sum(s[ty - r:ty + r + 1, tx - r:tx + r + 1] * mask)
    return out


rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(40, 70)).astype(np.float32)   # not a multiple of 16: edge tiles
box3 = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)

print("1. the tiled kernel against a direct reference (3x3 box filter, zero padding)")
print(f"  image {img.shape[1]}x{img.shape[0]}, tile {TILE}x{TILE}: tiled == reference: "
      f"{bool(np.allclose(tiled(img, box3), reference(img, box3), atol=1e-4))}")
for name, m in (("box filter", box3), ("asymmetric filter", np.arange(9, dtype=np.float32).reshape(3, 3))):
    same = np.allclose(reference(img, m[::-1, ::-1]), reference(img, m), atol=1e-4)
    print(f"  {name}: true convolution (filter flipped) == cross-correlation: {bool(same)}")

print("2. one 16x16 output tile: global reads, and the threads of its block (one per tile pixel)")
print("  filter  naive reads  tile reads     reuse  threads computing")
for k in (3, 5, 7):
    naive = TILE * TILE * k * k
    blk = TILE + k - 1
    print(f"  {k}x{k}   {naive:>11,}  {blk * blk:>4,} ({blk}x{blk})  {naive / blk ** 2:5.1f}x"
          f"  {TILE * TILE} of {blk * blk} ({100 * TILE * TILE / blk ** 2:.0f}%)")

print("3. multiply-adds per output pixel")
for k in (3, 5, 7):
    print(f"  {k}x{k}: {k * k} as one 2D filter, {2 * k} as a row pass and a column pass")
row = np.full(3, 1.0 / 3.0, dtype=np.float32)
two_pass = reference(reference(img, row[None, :]), row[:, None])
print(f"  3x3 box = (1/3 row) then (1/3 column): {bool(np.allclose(two_pass, reference(img, box3), atol=1e-4))}")

print("4. constant memory")
floats = CONSTANT_BYTES // 4
k_max = max(k for k in range(1, 256, 2) if k * k <= floats)
print(f"  64 KB holds {floats:,} floats; the largest odd filter is {k_max}x{k_max} ({k_max * k_max:,} weights)")
