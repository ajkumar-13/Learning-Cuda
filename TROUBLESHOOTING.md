# Troubleshooting

CUDA's toolchain is famously fragile to set up. This page collects the errors you are most likely to hit running this series, on Windows, Linux, and WSL2. For install steps see [SETUP.md](SETUP.md); for the arch table referenced below see [SETUP.md § Step 3](SETUP.md#step-3--pick-your--arch-flag).

> **First, locate the failure.** Every post ships a CPU model (`.py`) *and* a real kernel (`.cu`). If `python …_model.py` prints numbers but the `.cu` won't build or run, the bug is in your CUDA toolchain, not the code — jump to [The `.py` runs but the `.cu` fails](#the-py-runs-but-the-cu-fails).

---

## `nvcc: command not found` / `nvcc is not recognized`

The CUDA Toolkit is not installed or not on `PATH`. Note that `nvidia-smi` working does **not** mean `nvcc` is installed — the driver and the Toolkit are separate (see [SETUP.md § Step 1](SETUP.md#step-1--find-your-gpu-and-its-compute-capability)).

- **Linux/WSL2:** `which nvcc`. If empty, install the toolkit (`sudo apt install nvidia-cuda-toolkit`, or the CUDA apt repo for the newest release), then reopen the shell. If installed but not found, add it: `export PATH=/usr/local/cuda/bin:$PATH`.
- **Windows:** confirm it exists, then add the bin folder to `PATH`:
  ```powershell
  Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
  $env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
  ```
  Open a **new** terminal after a permanent `PATH` change.

---

## `Cannot find compiler 'cl.exe'` <a id="cannot-find-compiler-clexe"></a>

*(Windows only.)* `nvcc` needs the MSVC host compiler, which is only on `PATH` when the Visual Studio environment is initialized. The reliable fix is to point `nvcc` at `cl.exe` directly:

```powershell
nvcc -O2 input.cu -o input.exe `
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64\cl.exe"
```

Find `<version>` with
`Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"`.
Alternatives: compile from a **Developer Command Prompt for VS**, or run `vcvars64.bat` once in your shell to load MSVC into `PATH`.

---

## `cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)`

*(Windows only.)* A long-standing Windows `nvcc` bug, usually environment- or antivirus-related. In order:

1. Always pass `-ccbin` with the explicit `cl.exe` path (above) — this bypasses the faulty environment detection and fixes most cases.
2. Add Defender exclusions (run PowerShell as Administrator):
   ```powershell
   Add-MpPreference -ExclusionPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
   ```
3. If a one-line `int main(){}` `.cu` also fails, the install is broken — repair or reinstall the Toolkit, or drop to CUDA 11.8 (more forgiving with older MSVC).

---

## `no kernel image is available for execution on the device` / `sm_XX not supported`

The binary was built for the wrong architecture. Either you compiled for an `-arch` newer than your toolkit supports, or you ran a binary built for a different GPU than yours. Set `-arch` to **your** card's compute capability from [SETUP.md § Step 3](SETUP.md#step-3--pick-your--arch-flag):

```bash
nvcc -O3 -arch=sm_75 kernel.cu -o kernel    # sm_75 = Turing (GTX 16xx / RTX 20xx)
```

This also bites posts 13–14: `flash_attention.cu` targets `sm_80`, so it will not run on a pre-Ampere card even though it compiles.

---

## `CUDA driver version is insufficient for CUDA runtime version`

Your Toolkit is newer than your driver. Either update the GPU driver, or install a Toolkit version at or below the `CUDA Version` shown by `nvidia-smi`. The driver must be **at least as new** as the toolkit.

---

## `No CUDA-capable device is detected`

- The driver is missing or the GPU is hidden from the OS — check Device Manager (Windows) or `lspci | grep -i nvidia` (Linux), then update the driver and reboot.
- **WSL2:** this almost always means a Linux display driver was installed inside WSL, or the Windows driver is too old. Remove any in-WSL driver, update the **Windows** driver, and reinstall only the `cuda-toolkit` package (see [SETUP.md § WSL2](SETUP.md#wsl2-ubuntu-on-windows)).

---

## CMake can't find CUDA (`No CUDA toolset found` / `No CMAKE_CXX_COMPILER`)

You do **not** need CMake to run this series — every snippet builds with a single `nvcc` line. If you are using CMake anyway and it fails to detect CUDA or the host compiler on Windows, initialize MSVC first:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja
```

When in doubt, skip CMake and compile the `.cu` directly with `nvcc`.

---

## A kernel runs but prints garbage or zeros

The launch silently failed, or you read results before the GPU finished. Add error
checking — every series snippet does this with a `CUDA_CHECK` macro:

```cpp
cudaGetLastError();          // catches a bad launch configuration immediately
cudaDeviceSynchronize();     // surfaces an async runtime fault (and waits for the result)
```

Check the return of **both** after a launch; checking only one lets failures slip past. A launch is asynchronous, so without the sync you may copy results back before they are written.

---

## The `.py` runs but the `.cu` fails

This is the triage unique to this series' dual-companion design. If `python …_model.py` prints its numbers, your Python is fine and the problem is entirely in the CUDA toolchain:

| Symptom | Likely cause | Go to |
|---|---|---|
| `nvcc` not found | Toolkit not installed / not on `PATH` | [above](#nvcc-command-not-found--nvcc-is-not-recognized) |
| `Cannot find compiler 'cl.exe'` | MSVC not in environment (Windows) | [above](#cannot-find-compiler-clexe) |
| `ACCESS_VIOLATION` at compile | Windows `nvcc`/AV bug | [above](#cudafe-died-with-status-0xc0000005-access_violation) |
| Builds, but `no kernel image` at run | wrong `-arch` for your GPU | [above](#no-kernel-image-is-available-for-execution-on-the-device--sm_xx-not-supported) |
| `driver version is insufficient` | toolkit newer than driver | [above](#cuda-driver-version-is-insufficient-for-cuda-runtime-version) |
| Runs but wrong/zero output | unchecked launch error or missing sync | [above](#a-kernel-runs-but-prints-garbage-or-zeros) |

The CPU model is the ground truth: when both run, diff their output against the numbers
the post quotes.

---

## Still stuck?

- Re-verify the basics: `nvidia-smi` and `nvcc --version` both work.
- Rebuild the smallest thing — `device_query.cu` from [SETUP.md § Step 4](SETUP.md#step-4--verify-your-install).
- Search the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/).
