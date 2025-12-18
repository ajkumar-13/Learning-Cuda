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
- `src/Vector Addition/` – vector add GPU challenge (`question.md`, `solution.cu`, `test_solution.cu`).
- `src/Relu/` – ReLU activation challenge (`question.md`, `solution.cu`, `test_solution.cu`).
- `src/Matrix Addition/` – matrix add challenge (`question.md`, `solution.cu`, `test_solution.cu`).
- `src/Matrix Transpose/` – matrix transpose challenge (`question.md`, `solution.cu`, `test_solution.cu`).


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

## Running the small challenge tests

Each challenge folder under `src/` (e.g. `Vector Addition`, `Relu`,
`Matrix Addition`, `Matrix Transpose`) has:

- `question.md` – problem statement.
- `solution.cu` – your GPU implementation.
- `test_solution.cu` – a small host test harness.

To compile and run a test for a given challenge directly with `nvcc`
(example shown for ReLU):

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"
mkdir build -Force

nvcc -o .\build\test_relu.exe `
	".\src\Relu\solution.cu" `
	".\src\Relu\test_solution.cu" `
	-ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" `
	-O2

.\build\test_relu.exe
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