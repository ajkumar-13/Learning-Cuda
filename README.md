# Learning Cuda

Small Windows project for learning CUDA with CMake and nvcc.

## What’s in this repo

- `.gitignore` – ignores build folders, binaries, and IDE files.
- `CMakeLists.txt` – CMake build script.
	- If `nvcc` is found: builds CUDA examples.
	- If `nvcc` is not found: builds a CPU-only fallback.
- `src/vec_add.cu` – CUDA vector add example (`A + B -> C`).
- `src/multiply.cu` – CUDA element-wise multiply by a constant.
- `src/vec_add_cpu.cpp` – CPU fallback version of the vector add.


## Prerequisites (on Windows)

- CMake installed and on `PATH`.
- Visual Studio C++ build tools (MSVC, Windows SDK).
- NVIDIA CUDA Toolkit with `nvcc` working.

You can verify quickly in PowerShell:

```powershell
cmake --version
nvcc --version
```

## Build and run (PowerShell)

From the project root:

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"

# Configure CMake (writes build files into .\build)
cmake -S . -B build

# Build all default targets (CUDA + CPU fallback)
cmake --build build --config Release

# Run CUDA examples (if nvcc was found)
.\build\vecadd_nvcc.exe
.\build\multiply_nvcc.exe

# If nvcc is NOT found, only the CPU fallback exists
.\build\Release\vecadd.exe
```

## Adding new CUDA examples

- Put new `.cu` files in the `src` folder (e.g. `src/my_kernel.cu`).
- For quick experiments, you can compile them directly with `nvcc`:

	```powershell
	cd "C:\Users\admin\Desktop\Learning Cuda"
	mkdir build -Force
	nvcc -o .\build\my_kernel.exe .\src\my_kernel.cu -ccbin cl.exe -O2
	.\build\my_kernel.exe
	```

- To integrate them into CMake (so they build with `cmake --build`),
	follow the same `add_custom_command` / `add_custom_target` pattern used
	for `vec_add.cu` and `multiply.cu` in `CMakeLists.txt`.