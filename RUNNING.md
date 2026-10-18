# Running the code

Every post ships its companions in its own `snippets/` folder: a CPU model (`.py`) that reproduces the kernel's logic and prints the numbers the post quotes, and, for posts 01 to 18, the real CUDA kernel (`.cu`). This page is the one run recipe for all of them: what runs where, the two commands, the host-compiler setting Windows needs, the CMake build in `examples/`, and the build failures that still occur with CUDA 12.x.

## What runs where

- Every `.py` model runs on a CPU with Python and NumPy. No GPU is needed to follow any post, because the model prints every number the post derives.
- Every `.cu` kernel runs on one NVIDIA GPU with the CUDA Toolkit 12.x and a host C++ compiler (MSVC on Windows, g++ or clang on Linux). The kernels default to `-arch=sm_75`, the GeForce GTX 1650 (compute capability 7.5) the early posts were measured on.
- Feature floors: Tensor Cores need compute capability 7.0 or above (post 17), where independent thread scheduling also starts and makes `__syncwarp()` necessary in warp-level code (post 09); `cp.async`, BF16 arithmetic and `flash_attention.cu` need `sm_80`, an Ampere card or newer (post 15, post 18); Triton needs Linux (post 19).

## What each part needs

| Part | Posts | CPU models | Kernels |
|---|---|---|---|
| Part I - First Kernels | 01 to 03 | Python and NumPy | CUDA Toolkit 12.x and any NVIDIA card it supports, built with that card's `-arch` |
| Part II - Parallel Patterns | 04 to 09 | Python and NumPy | as Part I |
| Part III - Understanding Performance | 10 to 11 | Python and NumPy | as Part I; post 11 also uses Nsight Systems, Nsight Compute and `compute-sanitizer`, which ship with the toolkit |
| Part IV - Systems & Execution | 12 to 15 | Python and NumPy | as Part I; post 15's `async_pipeline.cu` needs `sm_80` |
| Part V - Tensor & Attention | 16 to 19 | Python and NumPy; post 19's `triton_matmul.py` also needs PyTorch and Triton on Linux | as Part I; post 17 needs compute capability 7.0 or above; post 18's `flash_attention.cu` needs `sm_80` |

## The CPU models

The series root carries a `pyproject.toml` and a `uv.lock` that pin NumPy and PyYAML. With [uv](https://docs.astral.sh/uv/) installed, run any model from the series root:

```bash
uv sync
uv run python posts/04-reduction/snippets/reduction_model.py
```

Any Python 3.11 or newer with NumPy runs the same files with `python` in place of `uv run python`. PyTorch and Triton are not part of the root environment: post 19's `triton_matmul.py` needs a Linux machine with both installed, and post 19's `tools_model.py` runs with NumPy alone.

## The kernels

Build and run a kernel with one `nvcc` line, from the series root:

```bash
nvcc -O3 -arch=sm_XX posts/04-reduction/snippets/reduction.cu -o reduction
./reduction
```

On Windows the executable is `reduction.exe`, run as `.\reduction.exe`. `nvcc` writes a `reduction.lib` and a `reduction.exp` beside a Windows executable; build into a scratch folder to keep the post directories clean. The header comment of every `.cu` states the build line and the `-arch` it was measured with.

### Pick the -arch flag

`-arch=sm_XX` is the card's compute capability with the dot dropped. `nvidia-smi` names the card, and NVIDIA's [compute capability table](https://developer.nvidia.com/cuda-gpus) gives its number. The `CUDA Version` that `nvidia-smi` prints is the newest toolkit the driver supports, not the toolkit installed; `nvcc --version` reports the latter.

| Architecture | Example cards | Compute capability | `-arch` |
|---|---|--:|---|
| Pascal | GTX 10xx | 6.1 | `sm_61` |
| Volta | V100 | 7.0 | `sm_70` |
| Turing | GTX 16xx, RTX 20xx | 7.5 | `sm_75` |
| Ampere | A100 | 8.0 | `sm_80` |
| Ampere | RTX 30xx | 8.6 | `sm_86` |
| Ada | RTX 40xx | 8.9 | `sm_89` |
| Hopper | H100 | 9.0 | `sm_90` |

The three cards the series measured on are the GTX 1650 (7.5), the RTX 3080 (8.6) and the A100 (8.0). A binary built for one architecture does not run on an older one, so a kernel built with the default `sm_75` needs rebuilding with the card's own flag on a Volta or Pascal card.

### The host compiler on Windows

`nvcc` drives the MSVC compiler `cl.exe`, which is on `PATH` only inside a Visual Studio developer environment. The rule: point `-ccbin` at the `Hostx64\x64` folder of the installed MSVC toolset, or run `nvcc` from a Developer Command Prompt (or after `vcvars64.bat`), where `cl.exe` is already on `PATH`. On the author's machine (Visual Studio 2019 Build Tools, CUDA 12.3) the full line is:

```powershell
nvcc -O3 -arch=sm_75 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" posts\01-introduction-to-cuda\snippets\device_query.cu -o device_query.exe
```

The edition (2019 or 2022, `BuildTools` or `Community`) and the MSVC version number (`14.29.30133` here) differ per machine; list the installed versions with `Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC"` (Visual Studio 2022 installs under `C:\Program Files\` instead).

### Check the toolchain

Post 01's `device_query.cu` is the smallest end-to-end test. Built with the line above, it prints the card's name, compute capability, streaming-multiprocessor count, shared memory per block, global memory and theoretical peak bandwidth: the numbers the rest of the series reasons about. If it builds and prints them, every other kernel builds the same way.

## Building every kernel with CMake

`examples/` carries a CMake build that compiles every runnable post kernel straight from its post's `snippets/` folder (no copies), one executable per kernel, together with the standalone example kernels in `examples/src/`. From `examples/`, `cmake -S . -B build` and then `cmake --build build --config Release` put the executables in `examples/build/`. The build file names the MSVC host compiler by a fixed path (`MSVC_COMPILER_PATH` in `examples/CMakeLists.txt`); on another machine, edit that one line to the local `cl.exe`. CMake is optional: every kernel also builds with the single `nvcc` line above.

## Posts whose companions differ

- Post 10 ([How fast can a kernel be](posts/10-roofline/index.md)) ships its model, `roofline_model.py`, which derives the roofline and occupancy numbers by counting, and `occupancy_check.cu`, which compares the model's occupancy rule with the runtime's `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and launches no kernel.
- Post 11 ([Profiling and debugging](posts/11-profiling-debugging/index.md)) ships `occupancy_demo.cu`, the kernel it measures, and `missing_guard.cu` and `shared_race.cu`, whose bugs run behind command-line flags for `compute-sanitizer`. On Windows, Nsight Systems and Nsight Compute may need administrator rights: Nsight Compute stops with `ERR_NVGPUCTRPERM` until the NVIDIA Control Panel allows all users the GPU performance counters.
- Post 18 ([FlashAttention](posts/18-flash-attention/index.md)) ships `flash_attention.cu` as an illustrative excerpt with no `main()`: it compiles with `-c` for `sm_80` and does not link. Its CPU model, `flash_attention.py`, is the runnable and verified companion.
- Post 19 ([CUTLASS and Triton](posts/19-cutlass-triton/index.md)) ships Python only: `tools_model.py` (NumPy) and `triton_matmul.py` (PyTorch and Triton, Linux).

## When a build fails

If the `.py` model prints its numbers but the `.cu` does not build or run, the problem is in the CUDA toolchain, not in the post's code. The failures that still occur with CUDA 12.x:

**`nvcc` is not recognized, or `nvcc: command not found`.**
Cause: the CUDA Toolkit is not installed or its `bin` folder is not on `PATH`; a working `nvidia-smi` shows only that the driver is installed.
Fix: install the toolkit from [NVIDIA's downloads](https://developer.nvidia.com/cuda-downloads), then add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` (Windows) or `/usr/local/cuda/bin` (Linux) to `PATH` and open a new terminal.

**`Cannot find compiler 'cl.exe' in PATH`.**
Cause: the MSVC host compiler is not in the environment (Windows only).
Fix: pass `-ccbin` with the toolset's `Hostx64\x64` folder, or build from a Developer Command Prompt (see "The host compiler on Windows").

**`unsupported Microsoft Visual Studio version`.**
Cause: the installed MSVC is newer than the CUDA release supports.
Fix: install a CUDA release that supports that Visual Studio, or point `-ccbin` at an older supported toolset.

**`cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)`.**
Cause: the Windows front end crashes, usually when `nvcc` picks up a broken compiler environment or a scanner locks its temporary files.
Fix: pass `-ccbin` explicitly; if it persists, exclude the CUDA install folder and the build folder from antivirus scanning; if a one-line `int main(){}` file also fails, repair the toolkit installation.

**`no kernel image is available for execution on the device`.**
Cause: the binary was built for an architecture the card does not have, for example `sm_80` code on a Turing card.
Fix: rebuild with the card's own `-arch`; a kernel whose header says `sm_80` (posts 15 and 18) needs an Ampere card or newer.

**`CUDA driver version is insufficient for CUDA runtime version`.**
Cause: the toolkit is newer than the driver.
Fix: update the driver, or install a toolkit no newer than the `CUDA Version` that `nvidia-smi` reports.

**`no CUDA-capable device is detected`.**
Cause: the driver is missing or the card is not visible to the operating system; under WSL 2, a Linux display driver installed inside WSL or an outdated Windows driver.
Fix: install or update the driver and reboot; under WSL 2, install only the CUDA Toolkit inside WSL and keep the driver on the Windows side.

**The kernel runs but prints zeros or garbage.**
Cause: a launch failed silently, or the result was read before the kernel finished.
Fix: check `cudaGetLastError()` after the launch and `cudaDeviceSynchronize()` before reading results; every kernel in the series wraps its calls in a `CUDA_CHECK` macro for this reason.

## How long it takes

Every CPU model finishes in under two seconds on a laptop CPU, and every kernel of posts 01 to 17 builds in four to six seconds and runs in one to five seconds on a GTX 1650, start-up of the CUDA context included (measured October 2026, CUDA 12.3). A table of GPU timings in a post names the card it was measured on; on another card the digits differ and the shape of the table is the lesson.
