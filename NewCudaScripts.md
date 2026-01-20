# Compiling and Running New CUDA (`.cu`) Files

This project is already set up to compile CUDA files with **nvcc** using CMake.
Here is how to add new `.cu` files and run them.

## 1. Current CUDA examples

There are currently two CUDA programs:

- `src/vec_add.cu`  → built as `vecadd_nvcc(.exe)`
- `src/multiply.cu` → built as `multiply_nvcc(.exe)`

Both are wired into `CMakeLists.txt`, so they build automatically when you run CMake.

On Windows, the executables are created in the `build` folder as:

- `build/vecadd_nvcc.exe`
- `build/multiply_nvcc.exe`

## 2. Configure and build with CMake (recommended)

From the project root (`C:\Users\admin\Desktop\Learning Cuda`):

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"

# 1) Configure CMake (creates the build files in .\build)
cmake -S . -B build

# 2) Build all default targets (vecadd_nvcc and multiply_nvcc)
cmake --build build --config Release

# 3) Run the CUDA programs
./build/vecadd_nvcc.exe
./build/multiply_nvcc.exe
```























































































while keeping the build process consistent and simple.This setup lets you grow a collection of small CUDA examples4. **Run** the generated `.exe` from the `build` folder.3. **Build** using either `nvcc` directly or `cmake --build ...`.     in `CMakeLists.txt` (Option B).   - **Integrated build** → add a new `add_custom_command` + `add_custom_target`   - **Fast experiment** → use direct `nvcc` (Option A).2. Decide how to build:1. **Write** your kernel(s) and a `main()` function in `src/your_file.cu`.Whenever you create a new `.cu` file:## 4. General workflow for new CUDA scripts   ```   .\build\my_kernel_nvcc.exe   ```powershell4. Run your new executable (on Windows):   ```   cmake --build build --config Release   cmake -S . -B build   cd "C:\Users\admin\Desktop\Learning Cuda"   ```powershell3. Re‑configure and build:   ```   add_custom_target(my_kernel ALL DEPENDS ${MYKERNEL_OUT})   # Custom target so you can build it by name     VERBATIM)     COMMENT "Compiling my_kernel.cu with nvcc"     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}     COMMAND "${NVCC_EXECUTABLE}" -o "${MYKERNEL_OUT}" "${CMAKE_SOURCE_DIR}/src/my_kernel.cu" -ccbin cl.exe -O2   add_custom_command(OUTPUT ${MYKERNEL_OUT}   # Custom command to compile your new .cu file   endif()     set(MYKERNEL_OUT ${CMAKE_BINARY_DIR}/my_kernel_nvcc)   else()     set(MYKERNEL_OUT ${CMAKE_BINARY_DIR}/my_kernel_nvcc.exe)   if (WIN32)   # Define an output path (Windows vs others)   ```cmake   follow the same pattern used for `vec_add.cu` and `multiply.cu`:2. **Edit** `CMakeLists.txt` and, inside the `if (NVCC_EXECUTABLE)` block,1. **Create** your file, for example: `src/my_kernel.cu`.To have CMake manage the build (like `vec_add.cu` and `multiply.cu`):### Option B – Integrate the new file into CMakeThis is the fastest way to experiment with a single new `.cu` file.```.\build\my_kernel.exe# Run itnvcc -o .\build\my_kernel.exe .\src\my_kernel.cu -ccbin cl.exe -O2# Compile your new CUDA filemkdir build -Force# Make sure the build folder existscd "C:\Users\admin\Desktop\Learning Cuda"```powershellYou can bypass CMake and call `nvcc` directly from PowerShell:### Option A – Quick direct compile with `nvcc`There are two ways to compile and run it.Suppose you create a new CUDA file `src/my_kernel.cu` with its own `main()`.## 3. Adding a brand‑new `.cu` file and compiling it  `src/vec_add_cpu.cpp`) is built.- If `nvcc` is **not** found, only the CPU fallback `vecadd` (from  for `vec_add.cu` and `multiply.cu`.- If `nvcc` is found, `CMakeLists.txt` uses custom commands to call nvcc```.\build
ectddectdd_nvcc.exe