# Understanding `CMakeLists.txt`

This file tells CMake how to build your project.

## How to configure and build (PowerShell)

From your project root (`C:\Users\admin\Desktop\Learning Cuda`):

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"

# 1) Configure CMake (creates the build files in .\build)
cmake -S . -B build

# 2) Build the project (Release configuration)
cmake --build build --config Release

# 3) Run the executable (path depends on CUDA vs CPU)
# If nvcc is found, you'll typically get vecadd_nvcc.exe
./build/vecadd_nvcc.exe

# If nvcc is NOT found, CMake builds the CPU fallback named vecadd.exe
./build/Release/vecadd.exe
```

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
  if (WIN32)
    set(VECADD_OUT ${CMAKE_BINARY_DIR}/vecadd_nvcc.exe)
  else()
    set(VECADD_OUT ${CMAKE_BINARY_DIR}/vecadd_nvcc)
  endif()
```
This block chooses the name of the **output executable**:
- On Windows (`WIN32` is true): use `vecadd_nvcc.exe`.
- On other systems (Linux/macOS): use `vecadd_nvcc` without `.exe`.

`CMAKE_BINARY_DIR` is the build directory (for example `build/`).
So the final path is something like `build/vecadd_nvcc.exe`.

```cmake
  add_custom_command(OUTPUT ${VECADD_OUT}
    COMMAND "${NVCC_EXECUTABLE}" -o "${VECADD_OUT}" "${CMAKE_SOURCE_DIR}/src/vec_add.cu" -ccbin cl.exe -O2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling CUDA binary with nvcc"
    VERBATIM)
```
Defines **how to build** the file `${VECADD_OUT}` using a custom command:
- `OUTPUT ${VECADD_OUT}`: this command will create the executable file.
- `COMMAND "${NVCC_EXECUTABLE}" ...`: the actual command to run:
  - Calls `nvcc` using its full path.
  - `-o "${VECADD_OUT}"`: sets the output file name.
  - `"${CMAKE_SOURCE_DIR}/src/vec_add.cu"`: input CUDA source file.
  - `-ccbin cl.exe`: tells `nvcc` to use MSVC (`cl.exe`) as the host C++ compiler.
  - `-O2`: enables optimization level 2.
- `WORKING_DIRECTORY ${CMAKE_BINARY_DIR}`: run the command inside the build folder.
- `COMMENT "Compiling CUDA binary with nvcc"`: text shown while building.
- `VERBATIM`: tells CMake not to modify or re‑interpret the arguments.

```cmake
  add_custom_target(vecadd ALL DEPENDS ${VECADD_OUT})
```
Defines a build **target** named `vecadd`:
- `DEPENDS ${VECADD_OUT}`: when you build `vecadd`, CMake will make sure `${VECADD_OUT}`
  is created using the custom command above.
- `ALL`: include this target in the default build, so a simple
  `cmake --build build --config Release` will build it.

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
  - If `nvcc` is available, it uses a custom command to compile `src/vec_add.cu`
    into a CUDA executable (`vecadd_nvcc`).
  - If `nvcc` is not available, it builds a CPU‑only executable (`vecadd`) from
    `src/vec_add_cpu.cpp`.
- This design lets the same project work on machines **with** and **without** CUDA.
