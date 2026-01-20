# Troubleshooting CUDA Compilation on Windows

Common errors and quick fixes when compiling CUDA code on Windows.

---

## "nvcc is not recognized"

**Problem:** You get `nvcc: command not found` in PowerShell.

**Fix:**
1. Verify CUDA is installed: `Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"`
2. If False, install CUDA Toolkit (see [SETUP.md](SETUP.md))
3. If True, add to PATH:
   ```powershell
   $env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
   ```
4. Open a new PowerShell window for changes to take effect.

---

## "Cannot find compiler 'cl.exe' in PATH"

**Problem:**
```
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

**Cause:** Visual Studio C++ compiler is not in the environment.

**Fix (Recommended):** Use `-ccbin` with full path:
```powershell
nvcc -o output.exe input.cu `
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" `
  -O2
```

**Alternatives:**
- Use Developer Command Prompt for VS (search Windows menu)
- Run vcvars64.bat before compiling:
  ```powershell
  & "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  nvcc -o output.exe input.cu
  ```

---

## "cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)"

**Problem:**
```
nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
```

**Cause:** Compiler toolchain issue, often due to environment initialization failure or antivirus interference.

**Fix (Recommended):** Use `-ccbin` with explicit path (same as above). This bypasses environment issues.

**If still fails:**
1. Add CUDA folder to Windows Defender exclusions (run as Administrator):
   ```powershell
   Add-MpPreference -ExclusionPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
   ```
2. Try a simpler test file:
   ```cuda
   #include <stdio.h>
   int main() {
       printf("Hello World\n");
       return 0;
   }
   ```
3. If test file fails too, restart your computer and try again.

---

## "No CUDA-capable device detected"

**Problem:** `nvidia-smi` reports no GPU, or compilation succeeds but runtime fails.

**Fix:**
1. Check GPU in Device Manager (Win + X → Device Manager → Display adapters)
2. Update NVIDIA driver: https://www.nvidia.com/Download/index.aspx
3. Restart your computer
4. Run `nvidia-smi` again

---

## CMake cannot find CUDA toolset

**Problem:** CMake configures but finds no CUDA language/toolset for Visual Studio.

**Fix:** The project uses a custom nvcc target that avoids this. Verify:
```powershell
cmake -S . -B build_vs -G "Visual Studio 16 2019" -A x64
cmake --build build_vs --config Release
```

**If CMake still fails:** Compile directly with nvcc and the `-ccbin` method above.

---

## CMake with Ninja fails: "No CMAKE_CXX_COMPILER"

**Problem:**
```
No CMAKE_CXX_COMPILER could be found.
```

**Fix:** Initialize MSVC environment before CMake:
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build_ninja -G Ninja
cmake --build build_ninja
```

---

## Unsupported GPU architecture warning

**Problem:**
```
nvcc warning: GPU architecture sm_52 is deprecated
```

**Fix:** Specify your GPU's compute capability with `-arch`:
```powershell
# GTX 1650 (Turing)
nvcc -arch=sm_75 -o output.exe input.cu

# RTX 30xx (Ampere)
nvcc -arch=sm_86 -o output.exe input.cu

# RTX 40xx (Ada)
nvcc -arch=sm_89 -o output.exe input.cu
```

Find your GPU: https://developer.nvidia.com/cuda-gpus

---

## Out of memory during compilation

**Problem:** Compiler or linker runs out of memory.

**Fix:**
1. Close other applications
2. Reduce optimization: change `-O2` to `-O0` or remove the flag
3. For large projects, split into smaller compilation units

---

## Wrong architecture or "sm_XX not supported"

**Problem:**
```
error: architecture 'sm_50' not supported
```

**Fix:** Check your GPU's compute capability and use the correct flag:
```powershell
# List supported architectures for your CUDA version
nvcc --version

# Common flags:
# GTX 10xx: -arch=sm_61
# GTX 16xx: -arch=sm_75
# RTX 20xx: -arch=sm_75
# RTX 30xx: -arch=sm_86
# RTX 40xx: -arch=sm_89
```

---

## Build fails with linker errors

**Problem:**
```
error: unresolved external symbol cudaMemcpy
```

**Cause:** CUDA runtime library not linked.

**Fix:** If using custom compilation, ensure you're linking CUDA libraries. For simple kernels with main(), nvcc handles this automatically. For complex projects, see [CMakeLists.txt](CMakeLists.txt).

---

## Still stuck?

- Check [SETUP.md](SETUP.md) for full setup instructions
- Verify `nvidia-smi` and `nvcc --version` work
- Verify `cl.exe` location and VS installation
- Search the NVIDIA forums: https://forums.developer.nvidia.com/

