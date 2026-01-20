# Matrix Transpose: The Ultimate Memory Coalescing Benchmark

> **Why reading rows is fast, writing columns is slow, and how Shared Memory fixes it**

---

## 1. Introduction

### The Hook

In our previous posts, we optimized computation ([Vector Add](../vector%20addition/)), algorithmic complexity ([Reduction](../Reduction/02_reduction.md)), and parallel primitives ([Scan](../scan/04_parallel_prefix_sum.md)). Now, we face the **Memory System**.

Matrix Transpose ($B = A^T$) is a mathematically trivial operation:

$$B[col][row] = A[row][col]$$

There is **no math to optimize**. No FLOPs. It is purely a **data movement problem**. Yet, a naive GPU implementation will run at **5-10% of peak bandwidth**. Why? Because the GPU hates jumping around in memory.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Matrix Transpose: The Operation                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input Matrix A (4×4)              Output Matrix B = Aᵀ (4×4)         │
│                                                                         │
│   ┌────┬────┬────┬────┐             ┌────┬────┬────┬────┐              │
│   │ 1  │ 2  │ 3  │ 4  │             │ 1  │ 5  │ 9  │ 13 │              │
│   ├────┼────┼────┼────┤             ├────┼────┼────┼────┤              │
│   │ 5  │ 6  │ 7  │ 8  │     →       │ 2  │ 6  │ 10 │ 14 │              │
│   ├────┼────┼────┼────┤             ├────┼────┼────┼────┤              │
│   │ 9  │ 10 │ 11 │ 12 │             │ 3  │ 7  │ 11 │ 15 │              │
│   ├────┼────┼────┼────┤             ├────┼────┼────┼────┤              │
│   │ 13 │ 14 │ 15 │ 16 │             │ 4  │ 8  │ 12 │ 16 │              │
│   └────┴────┴────┴────┘             └────┴────┴────┴────┘              │
│                                                                         │
│   Row 0 of A becomes Column 0 of B                                     │
│   Row 1 of A becomes Column 1 of B                                     │
│   ...and so on                                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Problem: Coalescing vs. Strided Access

GPUs fetch global memory in **128-byte transactions** (cache lines). The efficiency of your kernel depends entirely on how threads access memory.

| Access Pattern | Description | Efficiency |
|----------------|-------------|------------|
| **Coalesced** | 32 threads read 32 consecutive floats → 1 transaction (128 bytes) | **100%** |
| **Strided** | 32 threads read floats separated by stride → up to 32 transactions | **~3%** |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Coalesced vs. Strided Memory Access                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   COALESCED ACCESS (Reading a Row)                                      │
│   ════════════════════════════════                                      │
│                                                                         │
│   Memory:  [0][1][2][3][4][5][6][7][8][9]...                           │
│             ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑                                     │
│            T0 T1 T2 T3 T4 T5 T6 T7  (Thread IDs)                       │
│                                                                         │
│   All 32 threads access consecutive addresses                          │
│   → GPU fetches ONE 128-byte cache line                                │
│   → 100% of fetched data is used                                       │
│   → ✅ EFFICIENT                                                        │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   STRIDED ACCESS (Reading a Column, stride = width = 1024)             │
│   ════════════════════════════════════════════════════════════         │
│                                                                         │
│   Memory Layout (Row-Major, Width=1024):                               │
│                                                                         │
│   Row 0:  [0][1][2]...[1023]                                           │
│            ↑                                                            │
│           T0                                                            │
│   Row 1:  [1024][1025]...[2047]                                        │
│            ↑                                                            │
│           T1                                                            │
│   Row 2:  [2048][2049]...[3071]                                        │
│            ↑                                                            │
│           T2                                                            │
│   ...                                                                   │
│                                                                         │
│   Thread 0 accesses index 0      → Fetch 128-byte line (use 4 bytes)  │
│   Thread 1 accesses index 1024   → Fetch 128-byte line (use 4 bytes)  │
│   Thread 2 accesses index 2048   → Fetch 128-byte line (use 4 bytes)  │
│   ...                                                                   │
│                                                                         │
│   32 threads → 32 separate cache lines fetched!                        │
│   → 4096 bytes fetched, only 128 bytes used                            │
│   → 3.125% efficiency                                                  │
│   → ❌ INEFFICIENT                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Transpose Dilemma

In a Matrix Transpose, you can pick your poison:

| Strategy | Read | Write | Problem |
|----------|------|-------|---------|
| **Option A** | Rows (Coalesced ✓) | Columns (Strided ✗) | Slow writes |
| **Option B** | Columns (Strided ✗) | Rows (Coalesced ✓) | Slow reads |

**You cannot have both... unless you use Shared Memory.**

> **Note:** As mentioned in the CUDA Programming Guide, the GPU is designed for massive throughput. Strided access kills this throughput by starving the memory controller with inefficient requests.

---

## 2. The Naive Implementation (The Baseline)

Let's implement the "Read Coalesced, Write Strided" approach.

```cpp
__global__ void transpose_naive(float* out, float* in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // ═══════════════════════════════════════════════════════
        // READ from Input (COALESCED)
        // ═══════════════════════════════════════════════════════
        // Consecutive threads (x, x+1, x+2...) read consecutive addresses
        // Thread 0: in[y * width + 0]
        // Thread 1: in[y * width + 1]
        // Thread 2: in[y * width + 2]
        // These are adjacent in memory → GOOD!
        int in_index = y * width + x;
        
        // ═══════════════════════════════════════════════════════
        // WRITE to Output (STRIDED)
        // ═══════════════════════════════════════════════════════
        // Thread 0: out[0 * height + y] = out[y]
        // Thread 1: out[1 * height + y] = out[height + y]
        // Thread 2: out[2 * height + y] = out[2*height + y]
        // Addresses are 'height' apart → BAD!
        int out_index = x * height + y;

        out[out_index] = in[in_index];
    }
}
```

### Why It Fails

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Naive Transpose Memory Pattern                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input Matrix (1024 × 1024)                                           │
│   ══════════════════════════                                            │
│                                                                         │
│   Row 0: [A₀₀][A₀₁][A₀₂]...[A₀,₁₀₂₃]                                   │
│           ↑   ↑   ↑         ↑                                           │
│          T0  T1  T2       T31     ← 32 threads read ROW (coalesced)    │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Output Matrix (1024 × 1024)                                          │
│   ═══════════════════════════                                           │
│                                                                         │
│   Row 0:   [A₀₀] ...                                                   │
│             ↑                                                           │
│            T0 writes here                                               │
│                                                                         │
│   Row 1:   [A₀₁] ...                                                   │
│             ↑                                                           │
│            T1 writes here (1024 floats = 4096 bytes away!)             │
│                                                                         │
│   Row 2:   [A₀₂] ...                                                   │
│             ↑                                                           │
│            T2 writes here (another 4096 bytes away!)                   │
│                                                                         │
│   If height = 1024:                                                     │
│   Thread 0 writes to address A                                         │
│   Thread 1 writes to address A + 4096 bytes                            │
│   Thread 2 writes to address A + 8192 bytes                            │
│   ...                                                                   │
│                                                                         │
│   This is a Memory Divergence NIGHTMARE!                               │
│   Each thread triggers a separate memory transaction.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Performance Impact:**

| Metric | Naive Transpose | Theoretical Max |
|--------|-----------------|-----------------|
| Bandwidth | ~60 GB/s | ~900 GB/s |
| Efficiency | ~7% | 100% |

We're leaving **93% of performance** on the table!

---

## 3. Optimization 1: Shared Memory Tiling

### The Strategy

We need to **decouple the Read Pattern from the Write Pattern**. Shared Memory is our buffer.

1. **Read:** Load a $32 \times 32$ tile from Global Memory (A) into Shared Memory. (Coalesced)
2. **Sync:** `__syncthreads()` — ensure all threads have finished loading
3. **Write:** Read from Shared Memory and write to Global Memory (B) in a transposed order.

Crucially, because Shared Memory is **random-access** (no coalescing requirements), we can read/write it in any order we want. We write to Global Memory row-by-row (Coalesced), effectively "turning" the tile inside Shared Memory.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       The Tiled Transpose Dance                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   STEP 1: Load Tile from Global Memory (COALESCED)                     │
│   ════════════════════════════════════════════════                      │
│                                                                         │
│   Global Memory (Input A)         Shared Memory (tile[32][32])         │
│   ┌────────────────────┐          ┌────────────────────┐               │
│   │ Row 0: T0 T1...T31─┼─────────►│ tile[0][0..31]     │               │
│   │ Row 1: T0 T1...T31─┼─────────►│ tile[1][0..31]     │               │
│   │ ...                │          │ ...                │               │
│   │ Row 31:T0 T1...T31─┼─────────►│ tile[31][0..31]    │               │
│   └────────────────────┘          └────────────────────┘               │
│                                                                         │
│   Threads read consecutive addresses → COALESCED ✓                     │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   STEP 2: __syncthreads()                                              │
│   ═══════════════════════                                               │
│   Wait for all threads to finish loading the tile                      │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   STEP 3: Write Transposed Tile to Global Memory (COALESCED)           │
│   ═══════════════════════════════════════════════════════════          │
│                                                                         │
│   Shared Memory (read transposed)   Global Memory (Output B)           │
│   ┌────────────────────┐            ┌────────────────────┐             │
│   │ Col 0: tile[0..31][0]──────────►│ Row 0: T0 T1...T31 │             │
│   │ Col 1: tile[0..31][1]──────────►│ Row 1: T0 T1...T31 │             │
│   │ ...                │            │ ...                │             │
│   │ Col 31:tile[0..31][31]─────────►│ Row 31:T0 T1...T31 │             │
│   └────────────────────┘            └────────────────────┘             │
│                                                                         │
│   Threads WRITE consecutive addresses → COALESCED ✓                    │
│   Threads READ from shared memory (random access OK)                   │
│                                                                         │
│   ═══════════════════════════════════════════════════════════          │
│   RESULT: Both Global Memory accesses are COALESCED!                   │
│   ═══════════════════════════════════════════════════════════          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Code

```cpp
#define TILE_DIM 32

__global__ void transpose_shared(float* out, float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // ═══════════════════════════════════════════════════════
    // Calculate INPUT coordinates (for reading)
    // ═══════════════════════════════════════════════════════
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // ═══════════════════════════════════════════════════════
    // STEP 1: Read Global (Coalesced) → Write Shared
    // ═══════════════════════════════════════════════════════
    // Threads in a warp have consecutive threadIdx.x values
    // They read consecutive memory addresses → COALESCED
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // ═══════════════════════════════════════════════════════
    // STEP 2: Recalculate coordinates for OUTPUT
    // ═══════════════════════════════════════════════════════
    // The output tile location is TRANSPOSED:
    // - Output column comes from Input row (blockIdx.y)
    // - Output row comes from Input column (blockIdx.x)
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Note: blockIdx.y!
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // Note: blockIdx.x!

    // ═══════════════════════════════════════════════════════
    // STEP 3: Read Shared (Transposed) → Write Global (Coalesced)
    // ═══════════════════════════════════════════════════════
    if (x < height && y < width) {
        // Key insight: swap indices when reading from tile!
        // tile[threadIdx.x][threadIdx.y] instead of tile[threadIdx.y][threadIdx.x]
        // This reads the COLUMN from shared memory
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Understanding the Index Swap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Why We Swap Block and Thread Indices                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INPUT Matrix A (width × height)      OUTPUT Matrix B (height × width)│
│                                                                         │
│   Block (1, 2) processes:              Block (1, 2) writes to:         │
│   ┌───────────────────────┐            ┌───────────────────────┐       │
│   │  Tile at column 1,    │    →       │  Tile at column 2,    │       │
│   │  row 2 in A           │            │  row 1 in B           │       │
│   └───────────────────────┘            └───────────────────────┘       │
│                                                                         │
│   Input:  blockIdx.x = 1, blockIdx.y = 2                               │
│   Output: x uses blockIdx.y (= 2), y uses blockIdx.x (= 1)             │
│                                                                         │
│   This "flips" the tile position for the transposed output!            │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   INSIDE the tile:                                                      │
│                                                                         │
│   Load:  tile[threadIdx.y][threadIdx.x] = in[...]                      │
│   Store: out[...] = tile[threadIdx.x][threadIdx.y]  ← SWAPPED!         │
│                                                                         │
│   Thread (0,0) loads  A[row][col]   → tile[0][0]                       │
│   Thread (0,0) stores tile[0][0]    → B[col][row]  ✓ Transpose!        │
│                                                                         │
│   Thread (5,3) loads  A[row+3][col+5] → tile[3][5]                     │
│   Thread (5,3) stores tile[5][3]      → B[col+5][row+3]  ✓ Transpose!  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Optimization 2: Bank Conflict Avoidance

We fixed Global Memory coalescing, but we introduced a new problem in **Shared Memory**.

### The Conflict

Shared memory is divided into **32 Banks**. Successive 32-bit words map to successive banks:

- `tile[0][0]` is Bank 0
- `tile[0][1]` is Bank 1
- `tile[0][31]` is Bank 31
- `tile[1][0]` is Bank 0 (wraps around when TILE_DIM = 32)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Bank Conflict in Transpose                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Shared Memory tile[32][32] - Bank Assignment:                        │
│   ══════════════════════════════════════════════                        │
│                                                                         │
│              Col 0   Col 1   Col 2  ...  Col 31                        │
│            ┌───────┬───────┬───────┬───┬───────┐                       │
│   Row 0    │ Bk 0  │ Bk 1  │ Bk 2  │...│ Bk 31 │                       │
│            ├───────┼───────┼───────┼───┼───────┤                       │
│   Row 1    │ Bk 0  │ Bk 1  │ Bk 2  │...│ Bk 31 │  ← Same banks!       │
│            ├───────┼───────┼───────┼───┼───────┤                       │
│   Row 2    │ Bk 0  │ Bk 1  │ Bk 2  │...│ Bk 31 │                       │
│            ├───────┼───────┼───────┼───┼───────┤                       │
│   ...      │  ...  │  ...  │  ...  │...│  ...  │                       │
│            ├───────┼───────┼───────┼───┼───────┤                       │
│   Row 31   │ Bk 0  │ Bk 1  │ Bk 2  │...│ Bk 31 │                       │
│            └───────┴───────┴───────┴───┴───────┘                       │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   LOADING (No Conflict):                                                │
│   ══════════════════════                                                │
│   tile[threadIdx.y][threadIdx.x] = in[...]                             │
│                                                                         │
│   Thread 0: writes tile[row][0]  → Bank 0                              │
│   Thread 1: writes tile[row][1]  → Bank 1                              │
│   Thread 2: writes tile[row][2]  → Bank 2                              │
│   ...                                                                   │
│   All threads hit DIFFERENT banks → ✓ No Conflict                      │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   READING (32-Way Conflict!):                                          │
│   ═══════════════════════════                                           │
│   out[...] = tile[threadIdx.x][threadIdx.y]                            │
│                                                                         │
│   If threadIdx.y = 0 (first row of threads in block):                  │
│   Thread 0: reads tile[0][0]  → Bank 0                                 │
│   Thread 1: reads tile[1][0]  → Bank 0  ⚠️                             │
│   Thread 2: reads tile[2][0]  → Bank 0  ⚠️                             │
│   ...                                                                   │
│   Thread 31: reads tile[31][0] → Bank 0  ⚠️                            │
│                                                                         │
│   ALL 32 THREADS ACCESS BANK 0!                                        │
│   → Hardware serializes: 32× slower!                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Fix: Padding

We simply add one "dummy" column to the shared memory array:

```cpp
__shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 32 × 33
```

Now, the stride is **33 elements (floats)** instead of 32. Since each `float` is 4 bytes, the byte stride is $33 \times 4 = 132$ bytes, but for bank conflict analysis, we count in 4-byte words (elements):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Padding Eliminates Bank Conflicts                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   WITHOUT PADDING: tile[32][32], stride = 32                           │
│   ══════════════════════════════════════════                            │
│                                                                         │
│   tile[0][0] → address 0   → Bank 0 % 32 = Bank 0                      │
│   tile[1][0] → address 32  → Bank 32 % 32 = Bank 0  ← CONFLICT!        │
│   tile[2][0] → address 64  → Bank 64 % 32 = Bank 0  ← CONFLICT!        │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   WITH PADDING: tile[32][33], stride = 33                              │
│   ════════════════════════════════════════                              │
│                                                                         │
│   tile[0][0] → address 0   → Bank 0 % 32 = Bank 0                      │
│   tile[1][0] → address 33  → Bank 33 % 32 = Bank 1   ← Different!      │
│   tile[2][0] → address 66  → Bank 66 % 32 = Bank 2   ← Different!      │
│   tile[3][0] → address 99  → Bank 99 % 32 = Bank 3   ← Different!      │
│   ...                                                                   │
│   tile[31][0] → address 1023 → Bank 1023 % 32 = Bank 31               │
│                                                                         │
│   Column accesses are now perfectly DIAGONAL across banks!             │
│   → ✓ CONFLICT FREE                                                    │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   VISUALIZATION:                                                        │
│                                                                         │
│   Before (32×32):          After (32×33):                              │
│   Col 0 = All Bank 0       Col 0 = Banks 0,1,2,3...31                  │
│                                                                         │
│   │ Bk0 │                  │ Bk0 │                                     │
│   │ Bk0 │  ← Conflict!     │ Bk1 │  ← No conflict!                     │
│   │ Bk0 │                  │ Bk2 │                                     │
│   │ ... │                  │ ... │                                     │
│   │ Bk0 │                  │Bk31 │                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Cost of Padding

| Configuration | Shared Memory per Block | Overhead |
|---------------|------------------------|----------|
| `tile[32][32]` | 4,096 bytes | 0% |
| `tile[32][33]` | 4,224 bytes | +3% |

**We waste 3% of shared memory to gain 32× speedup on reads. Worth it!**

---

## 5. The Final Kernel (Copy-Paste Ready)

```cpp
#define TILE_DIM 32

// ⚠️ PRO TIP: 32×32 = 1024 threads, which is the CUDA hard maximum per block.
// If your kernel uses many registers, the compiler may fail or spill to slow local memory.
// For complex kernels, consider 16×16 tiles (256 threads) to maintain high occupancy.
// For this simple transpose (few registers), 32×32 maximizes memory throughput.

__global__ void transpose_optimized(float* out, float* in, int width, int height) {
    // ═══════════════════════════════════════════════════════
    // PADDED shared memory to avoid bank conflicts
    // ═══════════════════════════════════════════════════════
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // ═══════════════════════════════════════════════════════
    // Calculate INPUT coordinates
    // ═══════════════════════════════════════════════════════
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // ═══════════════════════════════════════════════════════
    // LOAD: Global → Shared (Coalesced read)
    // ═══════════════════════════════════════════════════════
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // ═══════════════════════════════════════════════════════
    // TRANSPOSE block coordinates for output
    // ═══════════════════════════════════════════════════════
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // ═══════════════════════════════════════════════════════
    // STORE: Shared → Global (Coalesced write, no bank conflicts)
    // ═══════════════════════════════════════════════════════
    if (x < height && y < width) {
        // Swap indices when reading from shared memory
        // This transposes the tile!
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Host launch code
void launch_transpose(float* d_out, float* d_in, int width, int height) {
    dim3 block(TILE_DIM, TILE_DIM);
    
    // ⚠️ CRITICAL: Grid dimensions must match how the kernel uses blockIdx!
    // Kernel uses: x = blockIdx.x * TILE + threadIdx.x  → for INPUT columns (width)
    //              y = blockIdx.y * TILE + threadIdx.y  → for INPUT rows (height)
    // Therefore: grid.x must cover width, grid.y must cover height
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,   // grid.x → blockIdx.x → width
              (height + TILE_DIM - 1) / TILE_DIM); // grid.y → blockIdx.y → height
    
    transpose_optimized<<<grid, block>>>(d_out, d_in, width, height);
}
```

### Handling Non-Square Matrices

For rectangular matrices, the grid dimension logic is the **#1 source of bugs**. Here's how to reason about it:

```cpp
// For matrix A[height][width] → B[width][height]
//
// The kernel's INPUT coordinate calculation:
//   x = blockIdx.x * TILE_DIM + threadIdx.x;  // x ranges over width
//   y = blockIdx.y * TILE_DIM + threadIdx.y;  // y ranges over height
//
// Therefore:
//   - blockIdx.x must cover all columns → grid.x = ceil(width / TILE_DIM)
//   - blockIdx.y must cover all rows    → grid.y = ceil(height / TILE_DIM)
//
// ⚠️ COMMON BUG: Swapping width/height here because you're thinking about
// the OUTPUT dimensions. Always think about INPUT when setting grid dims!

dim3 grid((width + TILE_DIM - 1) / TILE_DIM,   // grid.x → blockIdx.x → INPUT columns
          (height + TILE_DIM - 1) / TILE_DIM); // grid.y → blockIdx.y → INPUT rows
```

---

## 6. Benchmarks

### Test Configuration
- **GPU:** NVIDIA RTX 3080
- **Matrix Size:** 4096 × 4096 floats (64 MB)
- **Iterations:** 1000 (averaged)

| Implementation | Bandwidth (GB/s) | % of Peak | Notes |
|----------------|------------------|-----------|-------|
| Naive Copy | 350 | 100% | `out[i] = in[i]` (Reference) |
| Naive Transpose | 60 | 17% | Strided writes kill performance |
| Shared Memory | 220 | 63% | Coalesced global, bank conflicts |
| **Shared + Padding** | **310** | **88%** | Near peak memory bandwidth |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Transpose Performance Comparison                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Naive Copy        ████████████████████████████████████████  350 GB/s │
│   (Reference)                                                           │
│                                                                         │
│   Shared + Padding  ████████████████████████████████         310 GB/s │
│   (Optimized)       ↑ 5.2× faster than naive!                          │
│                                                                         │
│   Shared Memory     ██████████████████████████               220 GB/s │
│   (Bank Conflicts)                                                      │
│                                                                         │
│   Naive Transpose   ██████                                    60 GB/s │
│   (Strided Writes)                                                      │
│                                                                         │
│   0       50      100     150     200     250     300     350 (GB/s)   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Insights

| Optimization | Speedup | What It Fixed |
|--------------|---------|---------------|
| Shared Memory Tiling | **3.7×** | Global memory coalescing |
| Bank Conflict Padding | **1.4×** | Shared memory serialization |
| **Combined** | **5.2×** | Both issues |

We recovered **88% of the raw copy bandwidth** just by managing memory access patterns!

---

## 7. Common Pitfalls

### 1. Forgetting to Swap Block Indices

```cpp
// ❌ WRONG: Using same block indices for input and output
x = blockIdx.x * TILE_DIM + threadIdx.x;  // Same as input!
y = blockIdx.y * TILE_DIM + threadIdx.y;

// ✅ CORRECT: Swap blockIdx.x and blockIdx.y for output
x = blockIdx.y * TILE_DIM + threadIdx.x;  // Note: blockIdx.y
y = blockIdx.x * TILE_DIM + threadIdx.y;  // Note: blockIdx.x
```

### 2. Forgetting to Swap Shared Memory Indices

```cpp
// ❌ WRONG: Reading in same order as writing
out[...] = tile[threadIdx.y][threadIdx.x];

// ✅ CORRECT: Swap to transpose within the tile
out[...] = tile[threadIdx.x][threadIdx.y];
```

### 3. Wrong Output Dimensions

```cpp
// For A[height][width] → B[width][height]

// ❌ WRONG: Using input width for output stride
out[y * width + x] = ...;

// ✅ CORRECT: Output has 'height' columns (input's rows)
out[y * height + x] = ...;
```

### 4. Insufficient Shared Memory Padding

```cpp
// ❌ WRONG: No padding (bank conflicts)
__shared__ float tile[32][32];

// ✅ CORRECT: Add 1 column of padding
__shared__ float tile[32][33];
```

---

## 8. Challenge for the Reader

### Challenge 1: Rectangular Matrices

The code above works best when `TILE_DIM` divides width and height evenly.

**Challenge:** Modify the kernel to handle a matrix of size **1000 × 50**.

**Hints:**
- The logic `x = blockIdx.y * TILE_DIM` implies the grid layout flips
- Ensure your grid launch dimensions match the input interpretation
- Handle boundary conditions when tiles extend past matrix edges

### Challenge 2: In-Place Transpose

For square matrices, can you transpose **without** allocating a separate output buffer?

**Hints:**
- Elements on the diagonal stay in place
- Elements `A[i][j]` and `A[j][i]` swap
- Be careful about which blocks process which elements (avoid double-swapping!)

### Challenge 3: Profile with Nsight

Use Nsight Compute to verify:
1. **Global Memory Efficiency** — should be near 100% with tiling
2. **Shared Memory Bank Conflicts** — should be 0 with padding
3. **Achieved Occupancy** — limited by shared memory usage?

---

## Summary

### Key Takeaways

| Concept | Lesson |
|---------|--------|
| **Coalesced Access** | Adjacent threads should access adjacent memory addresses |
| **Transpose Dilemma** | Can't coalesce both read AND write without shared memory |
| **Shared Memory Tiling** | Use shared memory as a buffer to decouple access patterns |
| **Bank Conflicts** | Column access with stride 32 causes 32-way serialization |
| **Padding** | Add +1 column to make stride 33 → diagonal bank access |

### The Optimization Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Matrix Transpose Optimization Path                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   NAIVE                   SHARED MEMORY           SHARED + PADDING     │
│   ─────                   ─────────────           ────────────────     │
│                                                                         │
│   Global Read:            Global Read:            Global Read:         │
│   ✅ Coalesced            ✅ Coalesced            ✅ Coalesced         │
│                                                                         │
│   Global Write:           Global Write:           Global Write:        │
│   ❌ Strided              ✅ Coalesced            ✅ Coalesced         │
│                                                                         │
│   Shared Read:            Shared Read:            Shared Read:         │
│   N/A                     ❌ Bank Conflicts       ✅ No Conflicts      │
│                                                                         │
│   Bandwidth:              Bandwidth:              Bandwidth:           │
│   60 GB/s                 220 GB/s                310 GB/s             │
│   (17% peak)              (63% peak)              (88% peak)           │
│                                                                         │
│         ───────────────────────────────────────────────────►           │
│                     Optimization Progress                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What's Next?

In the next post, we'll explore **Histogram and Atomics** — how to count and accumulate when millions of threads compete for the same memory locations. This introduces a new challenge: **atomic operations** and their performance implications.

---

## References

1. [NVIDIA CUDA C++ Programming Guide — Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
2. [NVIDIA CUDA C++ Best Practices Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
3. Harris, M. — *Optimizing Parallel Reduction in CUDA* (NVIDIA Developer)
4. [GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
