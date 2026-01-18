# 🚀 Learning CUDA

A **comprehensive, hands-on guide to CUDA programming** on Windows. This repository contains step-by-step tutorials, practical challenges, and working examples to help you master GPU computing from the ground up.

**What is CUDA?** CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that lets you harness GPUs for general-purpose computing—powering everything from AI/deep learning to scientific simulations.

---

## 📋 Table of Contents

- [Quick Start (2 minutes)](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Documentation](#-documentation)
- [Learning Path](#-learning-path)
- [Build & Run Examples](#-build--run-examples)
- [Challenges & Exercises](#-challenges--exercises)
- [Troubleshooting](#-troubleshooting)
- [Resources](#-resources)

---

## ⚡ Quick Start

### 1. Check Prerequisites

```powershell
# Verify NVIDIA GPU driver
nvidia-smi

# Check CUDA compiler
nvcc --version

# Check build tools
cmake --version
```

### 2. Build Examples

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"

# Configure with CMake (Visual Studio Generator recommended)
cmake -S . -B build_vs -G "Visual Studio 16 2019" -A x64

# Build all targets
cmake --build build_vs --config Release
```

### 3. Run Examples

```powershell
# Run vector addition
.\build_vs\Release\vecadd_nvcc.exe

# Run element-wise multiply
.\build_vs\Release\multiply_nvcc.exe
```

**Expected Output:**
```
Vector Addition: [Success] Computed values match
Multiply: [Success] All elements computed correctly
```

> **⚠️ First time setup?** See [SETUP.md](SETUP.md) for complete Windows CUDA installation guide.

---

## 📁 Repository Structure

```
Learning Cuda/
├── README.md                          # This file
├── SETUP.md                           # 📖 Complete Windows setup guide
├── TROUBLESHOOTING.md                 # 🔧 Common errors & fixes
├── CMakeLists.txt                     # Build configuration
├── CMakeLists_explained.md            # CMake internals
│
├── blog/                              # 📚 Learning tutorials
│   ├── introduction/
│   │   ├── 00_introduction_to_cuda.md # Start here!
│   │   └── images/                    # Architecture diagrams
│   ├── vector_addition/
│   ├── matrix_multiplication/
│   └── [more advanced topics]
│
├── src/                               # 💻 Code examples & challenges
│   ├── vec_add.cu                     # Simple vector add example
│   ├── multiply.cu                    # Element-wise multiply
│   ├── vec_add_cpu.cpp                # CPU-only fallback
│   │
│   ├── vector_addition/               # Challenge: Vector operations
│   │   ├── question.md                # Problem statement
│   │   ├── solution.cu                # GPU implementation
│   │   └── test_solution.cu           # Test harness
│   │
│   ├── relu/                          # Challenge: ReLU activation
│   ├── matrix_addition/               # Challenge: Matrix ops
│   └── matrix_transpose/              # Challenge: Transpose kernel
│
├── build_vs/                          # Build output (Visual Studio)
├── build_ninja/                       # Build output (Ninja)
└── .gitignore                         # Git excludes
```

---

## 🔧 Setup & Installation

### Option 1: Complete Setup (Recommended)

Follow the **[SETUP.md](SETUP.md)** guide for:
- GPU driver installation
- Visual Studio C++ tools setup
- CUDA Toolkit installation
- Environment variable configuration
- Compiler verification

**Time estimate:** 30-45 minutes (mostly downloads/installation)

### Option 2: Quick Check

If you already have CUDA/VS installed:

```powershell
# Verify everything is ready
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"
Get-Command cl.exe
cmake --version
```

If all return `True` or version info, you're ready to build!

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **[SETUP.md](SETUP.md)** | Complete Windows setup guide for CUDA, VS, drivers |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Solutions to common compilation & runtime errors |
| **[CMakeLists_explained.md](CMakeLists_explained.md)** | Deep dive into build configuration |
| **[blog/introduction/00_introduction_to_cuda.md](blog/introduction/00_introduction_to_cuda.md)** | Foundational concepts: threads, blocks, memory hierarchy |

---

## 🎓 Learning Path

### **Beginner**
1. Read [Introduction to CUDA](blog/introduction/00_introduction_to_cuda.md) ← Start here
2. Understand thread hierarchy, memory model, warps
3. Compile and run `src/vec_add.cu` example

### **Intermediate**
4. Solve [Vector Addition Challenge](src/vector_addition/question.md)
5. Solve [ReLU Challenge](src/relu/question.md)
6. Read matrix operation tutorials (coming soon)

### **Advanced**
7. Solve [Matrix Addition Challenge](src/matrix_addition/question.md)
8. Solve [Matrix Transpose Challenge](src/matrix_transpose/question.md)
9. Study advanced topics: reduction, scan, tensor operations

---

## 🏗️ Build & Run Examples

### Using Visual Studio Generator (Recommended)

```powershell
cd "C:\Users\admin\Desktop\Learning Cuda"

# Configure
cmake -S . -B build_vs -G "Visual Studio 16 2019" -A x64

# Build (Release mode)
cmake --build build_vs --config Release

# Run examples
.\build_vs\Release\vecadd_nvcc.exe
.\build_vs\Release\multiply_nvcc.exe
```

✅ **Advantages:** Fast, automatic MSVC environment setup, IDE integration

### Using Ninja Generator (Faster builds)

```powershell
# Initialize MSVC environment first
& cmd.exe /c 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat' && `
cmake -S . -B build_ninja -G Ninja && `
cmake --build build_ninja
```

### Direct nvcc Compilation

For quick one-off experiments:

```powershell
nvcc -o .\build\test.exe .\src\vec_add.cu `
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" `
  -O2

.\build\test.exe
```

**Note:** Replace MSVC path with your installed version. See [SETUP.md](SETUP.md) for path discovery.

---

## 🎯 Challenges & Exercises

Each challenge has three components:

- **`question.md`** – Problem description and hints
- **`solution.cu`** – Your GPU kernel implementation goes here
- **`test_solution.cu`** – Test harness to verify correctness

### Vector Addition
```powershell
# Read the challenge
cat .\src\vector_addition\question.md

# Implement your solution in: .\src\vector_addition\solution.cu

# Compile and test
nvcc -o .\build\test_vecadd.exe `
  .\src\vector_addition\solution.cu `
  .\src\vector_addition\test_solution.cu `
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" `
  -O2

.\build\test_vecadd.exe
```

### ReLU Activation
```powershell
cat .\src\relu\question.md
# Edit .\src\relu\solution.cu
# Run tests (same compilation pattern as above)
```

### Matrix Operations
- Matrix Addition: `.\src\matrix_addition\question.md`
- Matrix Transpose: `.\src\matrix_transpose\question.md`

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `nvcc is not recognized` | See [SETUP.md § nvcc not recognized](SETUP.md#nvcc-is-not-recognized) |
| `Cannot find compiler 'cl.exe'` | See [SETUP.md § cl.exe not found](SETUP.md#nvcc-fatal-cannot-find-compiler-clexe) |
| `ACCESS_VIOLATION` | See [SETUP.md § ACCESS_VIOLATION](SETUP.md#nvcc-error-cudafe-died-with-status-0xc0000005-access_violation) |
| `No CMAKE_CXX_COMPILER` | Use `vcvars64.bat` before CMake, see [SETUP.md § CMake errors](SETUP.md#cmake-generator-verification-windows) |
| Images not rendering in GitHub | All images use forward slashes (`images/file.png`), should work now |

**→ Full troubleshooting guide:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📚 Resources

### Official Documentation
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Samples (GitHub)](https://github.com/NVIDIA/cuda-samples)

### This Repository
- [GPU vs CPU Architecture](blog/introduction/00_introduction_to_cuda.md#1-the-core-difference-cpu-vs-gpu)
- [Thread Hierarchy Explained](blog/introduction/00_introduction_to_cuda.md#3-organizing-the-chaos-the-thread-hierarchy)
- [Memory Model Deep Dive](blog/introduction/00_introduction_to_cuda.md#5-the-memory-hierarchy)
- [Warp Execution & Divergence](blog/introduction/00_introduction_to_cuda.md#warps-the-simt-execution-model)

### Community & Forums
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/gpu-accelerated-libraries/cuda/)
- [Stack Overflow - CUDA tag](https://stackoverflow.com/questions/tagged/cuda)

---

## 💡 Tips for Success

✅ **Do this:**
- Start with [Introduction blog post](blog/introduction/00_introduction_to_cuda.md)
- Run the provided examples first before modifying them
- Use `-ccbin` with full path for reliable compilation
- Test with small data sizes before scaling up
- Read error messages carefully (check [TROUBLESHOOTING.md](TROUBLESHOOTING.md))

❌ **Avoid this:**
- Assuming threads execute sequentially (they don't!)
- Ignoring memory coalescing patterns
- Creating too many blocks/threads (memory limits)
- Forgetting `cudaDeviceSynchronize()` before CPU-GPU sync

---

## 📝 License

This repository is for educational purposes. Feel free to use and modify for learning.

---

## 🤝 Contributing

Have improvements or new examples? PRs welcome!

To add a new challenge:
1. Create `src/my_challenge/` folder
2. Add `question.md` (problem statement)
3. Add `solution.cu` (template) and `test_solution.cu` (tests)
4. Update this README with a link

---

**Happy learning!** 🎓 Start with the [Introduction blog post](blog/introduction/00_introduction_to_cuda.md) or jump to a [Challenge](#-challenges--exercises).
