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
    DEPENDS "${CMAKE_SOURCE_DIR}/src/vec_add.cu"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling vec_add.cu with nvcc"
    VERBATIM)

  # multiply.cu -> multiply_nvcc
  add_custom_command(OUTPUT ${MULTIPLY_OUT}
    COMMAND "${NVCC_EXECUTABLE}" -o "${MULTIPLY_OUT}" "${CMAKE_SOURCE_DIR}/src/multiply.cu" -ccbin "${MSVC_COMPILER_PATH}" -O2
    DEPENDS "${CMAKE_SOURCE_DIR}/src/multiply.cu"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Compiling multiply.cu with nvcc"
    VERBATIM)

  add_custom_target(vecadd   ALL DEPENDS ${VECADD_OUT})
  add_custom_target(multiply ALL DEPENDS ${MULTIPLY_OUT})

  # --- Series post kernels --------------------------------------------------
  # Every runnable single-file kernel from posts/*/snippets is built here too,
  # so `cmake --build` produces one executable per kernel, exactly like the two
  # examples above. The sources stay in their post folders (one source of truth,
  # the same file each post teaches and cites) — they are not copied.
  get_filename_component(REPO_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
  set(POSTS_DIR "${REPO_ROOT}/posts")

  function(add_series_kernel name relpath arch)
    if (WIN32)
      set(_out "${CMAKE_BINARY_DIR}/${name}.exe")
      set(_ccbin -ccbin "${MSVC_COMPILER_PATH}")
    else()
      set(_out "${CMAKE_BINARY_DIR}/${name}")
      set(_ccbin "")
    endif()
    set(_src "${POSTS_DIR}/${relpath}")
    add_custom_command(OUTPUT "${_out}"
      COMMAND "${NVCC_EXECUTABLE}" -O3 -arch=${arch} -o "${_out}" "${_src}" ${_ccbin}
      DEPENDS "${_src}"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "nvcc ${relpath} (${arch}) -> ${name}"
      VERBATIM)
    add_custom_target(${name} ALL DEPENDS "${_out}")
  endfunction()

  add_series_kernel(device_query   01-introduction-to-cuda/snippets/device_query.cu    sm_75)
  add_series_kernel(vector_add     02-vector-addition/snippets/vector_add.cu           sm_75)
  add_series_kernel(matmul         03-matrix-multiplication/snippets/matmul.cu         sm_75)
  add_series_kernel(reduction      04-reduction/snippets/reduction.cu                  sm_75)
  add_series_kernel(histogram      05-histogram/snippets/histogram.cu                  sm_75)
  add_series_kernel(transpose      06-matrix-transpose/snippets/transpose.cu           sm_75)
  add_series_kernel(convolution    07-convolution/snippets/convolution.cu              sm_75)
  add_series_kernel(scan           08-parallel-scan/snippets/scan.cu                   sm_75)
  add_series_kernel(synchronization 09-synchronization/snippets/sync.cu                sm_75)
  add_series_kernel(occupancy_demo 11-profiling-debugging/snippets/occupancy_demo.cu   sm_75)
  add_series_kernel(streams        12-cuda-streams/snippets/streams.cu                 sm_75)
  add_series_kernel(cuda_graphs    13-cuda-graphs/snippets/graphs.cu                   sm_75)
  add_series_kernel(fusion         14-kernel-fusion/snippets/fusion.cu                 sm_75)
  # async_pipeline needs Ampere (sm_80+) to RUN; tensor_core needs a Tensor-Core GPU
  # (Volta+, i.e. RTX not GTX) to RUN. Both still COMPILE with a CUDA 11+ toolkit.
  add_series_kernel(async_pipeline 15-async-copy-pipelining/snippets/async_pipeline.cu sm_80)
  add_series_kernel(tensor_core    17-tensor-cores/snippets/tensor_core.cu             sm_75)
  add_series_kernel(precision      16-numerical-precision/snippets/precision.cu        sm_75)
else()
  message(STATUS "NVCC not found; building CPU fallback")
  add_executable(vecadd src/vec_add_cpu.cpp)
endif()
