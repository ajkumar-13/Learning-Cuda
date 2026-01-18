cmake_minimum_required(VERSION 3.18)
project(LearningCuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# Try to find nvcc first. If found, create a custom nvcc-based build step which avoids
# requiring Visual Studio integration for CUDA toolsets (sometimes CMake+VS can't detect
# the CUDA toolset even when nvcc is installed). If nvcc is not found, fall back to CPU.
find_program(NVCC_EXECUTABLE nvcc HINTS "$ENV{CUDA_PATH}/bin" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin" PATHS ENV PATH)

if (NVCC_EXECUTABLE)
  message(STATUS "NVCC found: ${NVCC_EXECUTABLE}")
  # Output executable paths
  if (WIN32)
    set(VECADD_OUT    ${CMAKE_BINARY_DIR}/vecadd_nvcc.exe)
    set(MULTIPLY_OUT  ${CMAKE_BINARY_DIR}/multiply_nvcc.exe)
  else()
    set(VECADD_OUT    ${CMAKE_BINARY_DIR}/vecadd_nvcc)
    set(MULTIPLY_OUT  ${CMAKE_BINARY_DIR}/multiply_nvcc)
  endif()

  # Use full path to cl.exe for better Ninja support
  set(MSVC_COMPILER_PATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe")

  # vec_add.cu -> vecadd_nvcc
  add_custom_command(OUTPUT ${VECADD_OUT}
    COMMAND "${NVCC_EXECUTABLE}" -o "${VECADD_OUT}" "${CMAKE_SOURCE_DIR}/src/vec_add.cu" -ccbin "${MSVC_COMPILER_PATH}" -O2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling vec_add.cu with nvcc"
    VERBATIM)

  # multiply.cu -> multiply_nvcc
  add_custom_command(OUTPUT ${MULTIPLY_OUT}
    COMMAND "${NVCC_EXECUTABLE}" -o "${MULTIPLY_OUT}" "${CMAKE_SOURCE_DIR}/src/multiply.cu" -ccbin "${MSVC_COMPILER_PATH}" -O2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling multiply.cu with nvcc"
    VERBATIM)

  add_custom_target(vecadd   ALL DEPENDS ${VECADD_OUT})
  add_custom_target(multiply ALL DEPENDS ${MULTIPLY_OUT})
else()
  message(STATUS "NVCC not found; building CPU fallback")
  add_executable(vecadd src/vec_add_cpu.cpp)
endif()
