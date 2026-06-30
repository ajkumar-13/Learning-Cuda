# Setting up a machine to run this series

Every post ships two runnable companions: a **CPU model** (`*_model.py` or a small `.py`) that needs only Python and NumPy, and the **real CUDA kernel** (`*.cu`) that needs an NVIDIA GPU and the CUDA Toolkit. This guide gets both working on Windows, Linux, and WSL2. If anything below fails, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md); for the per-post run commands see [RUNNING.md](RUNNING.md).

> **No GPU? You can still follow every post.** The kernels are written to be read without running them, and each post's CPU model reproduces the numbers it quotes. You only need the CUDA Toolkit for the `.cu` files; the `.py` models run anywhere Python does.

---

## What you need

| To run | You need |
|---|---|
| The CPU models (`*.py`) | Python 3.9+ and NumPy (`pip install numpy`) |
| The real kernels (`*.cu`) | An NVIDIA GPU + the **CUDA Toolkit 12.x** (provides `nvcc`) + a host C++ compiler |
| Posts 13–14 (Tensor Cores, FlashAttention) | A **Volta-or-newer** GPU (compute capability ≥ 7.0); `flash_attention.cu` targets `sm_80` (Ampere) |

The host C++ compiler is **MSVC** (`cl.exe`, from Visual Studio) on Windows and **g++/clang** on Linux. `nvcc` drives it for you; you just need it installed.

---

## Step 1 — Find your GPU and its compute capability

Run `nvidia-smi` (it ships with the driver, on every platform):

```bash
nvidia-smi
```

Two things to read off it:

- **The GPU name** (e.g. `NVIDIA GeForce GTX 1650`). Look it up at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) to get its **compute capability** — a number like `7.5` that decides your `-arch` flag (Step 3).
- **The `CUDA Version` field** in the top-right. This is the **maximum** CUDA version the *driver* supports, **not** the Toolkit you have installed. You can install that version or any lower one. `nvidia-smi` showing a CUDA version does **not** mean `nvcc` is installed — the Toolkit is a separate download (Step 2).

If `nvidia-smi` reports no device, the driver is missing or the GPU is not visible to the OS; fix that before going further (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

---

## Step 2 — Install the CUDA Toolkit

The rule of thumb: **the driver must be at least as new as the Toolkit.** Update the driver first, then install a Toolkit version at or below the one `nvidia-smi` reports.

### Linux (Ubuntu/Debian)

```bash
# the toolkit (nvcc + libraries); the driver comes from your distro or NVIDIA's repo
sudo apt update
sudo apt install -y nvidia-cuda-toolkit build-essential
nvcc --version          # confirm nvcc is on PATH
```

For the newest Toolkit, use NVIDIA's official `.run` file or the CUDA apt repo from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) instead of the distro package (which can lag a release or two).

### WSL2 (Ubuntu on Windows)

WSL2 gives you a real Linux `nvcc` while sharing the Windows GPU. The one rule that trips everyone up:

> **Install the CUDA *Toolkit* inside WSL, but do NOT install a Linux display driver inside WSL.** The GPU driver lives on the **Windows host**; WSL sees the GPU through it. Installing a Linux driver inside WSL breaks the passthrough.

```bash
# inside WSL2 — toolkit only, no driver
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6 build-essential
nvidia-smi              # should show your GPU, served by the Windows driver
```

Make sure your Windows NVIDIA driver is current (it carries the WSL GPU support).

### Windows (native)

Three pieces: the **driver**, **Visual Studio Build Tools** (the `cl.exe` host compiler), and the **CUDA Toolkit**. The fast path uses `winget`:

```powershell
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e   # then add "Desktop development with C++"
winget install --id NVIDIA.CUDA -e
```

If `winget` is unavailable, download the installers manually: [driver](https://www.nvidia.com/Download/index.aspx) · [VS Build Tools](https://visualstudio.microsoft.com/downloads/) (select **Desktop development with C++**) · [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

> **Windows reality check.** In a plain PowerShell window `nvcc` often can't find `cl.exe`. The reliable fix is to pass the compiler explicitly with `-ccbin`, or to compile from a **Developer Command Prompt for VS**. Both are covered in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#cannot-find-compiler-clexe).

---

## Step 3 — Pick your `-arch` flag

`nvcc` compiles for a specific GPU architecture. Match `-arch=sm_XX` to your card's compute capability from Step 1:

| GPU family | Example cards | Compute capability | `-arch` |
|---|---|--:|---|
| Pascal | GTX 10xx | 6.1 | `sm_61` |
| Volta | V100 | 7.0 | `sm_70` |
| Turing | GTX 16xx, RTX 20xx | 7.5 | `sm_75` |
| Ampere | A100 | 8.0 | `sm_80` |
| Ampere | RTX 30xx | 8.6 | `sm_86` |
| Ada | RTX 40xx | 8.9 | `sm_89` |
| Hopper | H100 | 9.0 | `sm_90` |

The snippets default to `-arch=sm_75` (the GTX 1650 the early posts were measured on). Change it to your card's value. **Tensor Cores need ≥ 7.0**; `bf16` and the `flash_attention.cu` kernel need **≥ 8.0**. If you omit `-arch`, `nvcc` builds for an old default and may warn or refuse on newer drivers.

---

## Step 4 — Verify your install

Post 01 ships a device-query program — the natural "is everything working?" test. From the repository root:

```bash
nvcc -O2 blog/posts/01-introduction-to-cuda/snippets/device_query.cu -o device_query
./device_query          # Windows: .\device_query.exe
```

It prints your GPU's name, compute capability, SM count, and memory — the same numbers the series reasons about. If it builds and runs, you are ready.

On Windows, if the build fails with `Cannot find compiler 'cl.exe'`, add the host compiler explicitly (adjust the version to yours):

```powershell
nvcc -O2 blog\posts\01-introduction-to-cuda\snippets\device_query.cu -o device_query.exe `
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64\cl.exe"
```

---

## Step 5 — Run a post's companions

Every post is self-contained. From the repository root, the **CPU model** runs anywhere:

```bash
python blog/posts/04-reduction/snippets/reduction_model.py
```

and the **real kernel** builds with one `nvcc` line (use that post's `-arch`; each `.cu` header comment states it):

```bash
nvcc -O3 -arch=sm_75 blog/posts/04-reduction/snippets/reduction.cu -o reduction && ./reduction
```

That two-command pattern — `python` the model, `nvcc` the kernel — is the whole workflow. [RUNNING.md](RUNNING.md) lists it per post; [TROUBLESHOOTING.md](TROUBLESHOOTING.md) covers what to do when a build fails.
