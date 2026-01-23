# Understanding `CMakeLists.txt`

This file tells CMake how to build your project.

## How to configure and build (PowerShell)

From your project root (`C:\Users\admin\Desktop\Learning Cuda`):

### Using Visual Studio Generator (Recommended)

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"

# 1) Configure CMake (creates build files in .\build_vs)
cmake -S . -B build_vs -G "Visual Studio 16 2019" -A x64

# 2) Build the project (Release configuration)
cmake --build build_vs --config Release

# 3) Run the executables (directly in build_vs folder)
.\build_vs\vecadd_nvcc.exe
.\build_vs\multiply_nvcc.exe
```

### Using Ninja Generator (Faster builds)

```powershell
# Run entire build chain inside cmd.exe
cmd.exe /c """C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake -S . -B build_ninja -G Ninja && cmake --build build_ninja"

# Run the executables
.\build_ninja\vecadd_nvcc.exe
.\build_ninja\multiply_nvcc.exe
```

**Note:** Executables are placed **directly** in the build folder (e.g., `build_vs/vecadd_nvcc.exe`), not in a `Release/` subfolder, because the CMakeLists.txt uses custom commands that output to `${CMAKE_BINARY_DIR}`.

Below is a line‑by‑line explanation of your current `CMakeLists.txt`.

```cmake
cmake_minimum_required(VERSION 3.18)
```
Tells CMake: "You must be at least version **3.18** to run this project."
This makes sure the features we use are available.

```cmake
project(LearningCuda LANGUAGES CXX)
```
Defines a project named **`LearningCuda`** and says we use the **C++ language** (`CXX`).

```cmake
set(CMAKE_CXX_STANDARD 14)
```
Asks CMake to compile C++ code using the **C++14** standard.

```cmake
# Try to find nvcc first. If found, create a custom nvcc-based build step which avoids
# requiring Visual Studio integration for CUDA toolsets (sometimes CMake+VS can't detect
# the CUDA toolset even when nvcc is installed). If nvcc is not found, fall back to CPU.
```
This is just a **comment** for humans. It explains the strategy:
- First, try to find the CUDA compiler `nvcc`.
- If `nvcc` is found, use a custom build command that calls `nvcc` directly.
- If `nvcc` is **not** found, build a CPU‑only version instead.

```cmake
find_program(NVCC_EXECUTABLE nvcc HINTS "$ENV{CUDA_PATH}/bin" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin" PATHS ENV PATH)
```
Tries to find the `nvcc` program and store its full path in `NVCC_EXECUTABLE`.
It looks in:
- `$CUDA_PATH/bin` if that environment variable exists.
- Typical CUDA install folders on Windows.
- The system `PATH`.

If `nvcc` is found, `NVCC_EXECUTABLE` will contain something like:
`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe`.

```cmake
if (NVCC_EXECUTABLE)
```
Checks **if we successfully found** `nvcc`.
- If yes ⇒ run the code inside this `if` block (CUDA build).
- If no ⇒ skip to the `else()` part (CPU fallback).

```cmake
  message(STATUS "NVCC found: ${NVCC_EXECUTABLE}")
```
Prints a status message during CMake configuration, for example:
`-- NVCC found: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe`.

```cmake
  # Output executable paths
  if (WIN32)
    set(VECADD_OUT    ${CMAKE_BINARY_DIR}/vecadd_nvcc.exe)
    set(MULTIPLY_OUT  ${CMAKE_BINARY_DIR}/multiply_nvcc.exe)
  else()
    set(VECADD_OUT    ${CMAKE_BINARY_DIR}/vecadd_nvcc)
    set(MULTIPLY_OUT  ${CMAKE_BINARY_DIR}/multiply_nvcc)
  endif()
```
This block chooses the name of the **output executables** for both programs:
- On Windows (`WIN32` is true): use `.exe` extension (e.g., `vecadd_nvcc.exe`, `multiply_nvcc.exe`).
- On other systems (Linux/macOS): no `.exe` extension.

`CMAKE_BINARY_DIR` is the build directory (for example `build_vs/` or `build_ninja/`).
So the final paths are like `build_vs/vecadd_nvcc.exe` and `build_vs/multiply_nvcc.exe`.

```cmake
  # Use full path to cl.exe for better Ninja support
  set(MSVC_COMPILER_PATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe")
```
Sets the **full path to the MSVC compiler** (`cl.exe`):
- `nvcc` needs a C++ compiler to compile host (CPU) code.
- Using the full path makes it work with **Ninja generator** (which doesn't auto-detect MSVC).
- Visual Studio generator can auto-detect this, but the explicit path ensures consistency.

**Note:** This path is specific to Visual Studio 2019 BuildTools version 14.29.30133. If your VS version differs, you may need to update this path.

```cmake
  # vec_add.cu -> vecadd_nvcc
  add_custom_command(OUTPUT ${VECADD_OUT}
    COMMAND "${NVCC_EXECUTABLE}" -o "${VECADD_OUT}" "${CMAKE_SOURCE_DIR}/src/vec_add.cu" -ccbin "${MSVC_COMPILER_PATH}" -O2
    DEPENDS "${CMAKE_SOURCE_DIR}/src/vec_add.cu"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling vec_add.cu with nvcc"
    VERBATIM)
```
Defines **how to build** the file `${VECADD_OUT}` using a custom command:
- `OUTPUT ${VECADD_OUT}`: this command will create the `vecadd_nvcc.exe` executable.
- `COMMAND "${NVCC_EXECUTABLE}" ...`: the actual compilation command:
  - Calls `nvcc` using its full path.
  - `-o "${VECADD_OUT}"`: sets the output file name.
  - `"${CMAKE_SOURCE_DIR}/src/vec_add.cu"`: input CUDA source file.
  - `-ccbin "${MSVC_COMPILER_PATH}"`: tells `nvcc` to use the specified MSVC compiler for host code.
  - `-O2`: enables optimization level 2.
- `DEPENDS "${CMAKE_SOURCE_DIR}/src/vec_add.cu"`: CMake tracks this dependency - if `vec_add.cu` changes, the executable will be rebuilt automatically.
- `WORKING_DIRECTORY ${CMAKE_BINARY_DIR}`: run the command inside the build folder.
- `COMMENT "Compiling vec_add.cu with nvcc"`: text shown during build.
- `VERBATIM`: tells CMake not to modify or re‑interpret the arguments.

```cmake
  # multiply.cu -> multiply_nvcc
  add_custom_command(OUTPUT ${MULTIPLY_OUT}
    COMMAND "${NVCC_EXECUTABLE}" -o "${MULTIPLY_OUT}" "${CMAKE_SOURCE_DIR}/src/multiply.cu" -ccbin "${MSVC_COMPILER_PATH}" -O2
    DEPENDS "${CMAKE_SOURCE_DIR}/src/multiply.cu"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling multiply.cu with nvcc"
    VERBATIM)
```
Identical to the `vec_add.cu` command above, but for `multiply.cu`:
- Compiles `src/multiply.cu` → `multiply_nvcc.exe`.
- Same compiler flags and dependency tracking.

```cmake
  add_custom_target(vecadd   ALL DEPENDS ${VECADD_OUT})
  add_custom_target(multiply ALL DEPENDS ${MULTIPLY_OUT})
```
Defines build **targets** named `vecadd` and `multiply`:
- `DEPENDS ${VECADD_OUT}` / `${MULTIPLY_OUT}`: when you build these targets, CMake will invoke the custom commands above to create the executables.
- `ALL`: include these targets in the default build, so `cmake --build build_vs --config Release` will build both automatically.

```cmake
else()
  message(STATUS "NVCC not found; building CPU fallback")
  add_executable(vecadd src/vec_add_cpu.cpp)
endif()
```
This is the **fallback path** when `nvcc` is not found:
- `message(...)`: prints a message that we are building the CPU‑only version.
- `add_executable(vecadd src/vec_add_cpu.cpp)`: builds a normal C++ executable
  called `vecadd` using `src/vec_add_cpu.cpp`.

If `nvcc` is missing, you still get a working program that runs the vector addition
on the CPU instead of on the GPU.

---

## Summary

- `CMakeLists.txt` is the script that tells CMake **what to build and how**.
- This particular file:
  - Looks for the CUDA compiler `nvcc`.
  - If `nvcc` is available, it uses custom commands to compile:
    - `src/vec_add.cu` → `vecadd_nvcc.exe`
    - `src/multiply.cu` → `multiply_nvcc.exe`
  - Uses the full path to MSVC compiler for better compatibility with Ninja generator.
  - Tracks source file dependencies so changes trigger automatic rebuilds.
  - If `nvcc` is not available, it builds a CPU‑only executable (`vecadd`) from `src/vec_add_cpu.cpp`.
- This design lets the same project work on machines **with** and **without** CUDA.
- Executables are placed **directly** in the build folder (`build_vs/` or `build_ninja/`), not in a `Release/` subfolder, because of the custom command approach.

## Why Custom Commands Instead of `enable_language(CUDA)`?

This CMakeLists.txt uses **custom `nvcc` commands** rather than CMake's built-in CUDA support (`enable_language(CUDA)`). Here's why:

**Advantages:**
- **More reliable on Windows** when CMake can't auto-detect the CUDA toolset
- **Works with any generator** (Visual Studio, Ninja, etc.)
- **Explicit control** over nvcc flags and MSVC compiler path
- **Fallback to CPU** when CUDA is unavailable

**Trade-offs:**
- **Hardcoded MSVC path** - needs updating if Visual Studio version changes
- **Manual dependency tracking** - must add `DEPENDS` for each source file
- **Non-standard executable paths** - outputs to `build_vs/` instead of `build_vs/Release/`

For simple projects or when CMake's CUDA detection is problematic, this approach is very effective.
