cmake_minimum_required(VERSION 3.20)
project(rvml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

#===----------------------------------------------------------------------===//
# LLVM/MLIR Configuration
#===----------------------------------------------------------------------===//

# If MLIR_DIR is not set, build from vendored third_party/llvm-project.
if(NOT MLIR_DIR)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/CMakeLists.txt")
    message(FATAL_ERROR
      "Vendored LLVM not found at third_party/llvm-project. Either:\n"
      "  1. Run: git submodule update --init --recursive\n"
      "  2. Set -DMLIR_DIR=/path/to/lib/cmake/mlir")
  endif()

  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
  set(LLVM_TARGETS_TO_BUILD "host;RISCV" CACHE STRING "")
  set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")

  add_subdirectory(third_party/llvm-project/llvm EXCLUDE_FROM_ALL)

  set(MLIR_MAIN_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir")
  set(MLIR_INCLUDE_DIR "${MLIR_MAIN_SRC_DIR}/include")
  set(MLIR_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include")
  set(MLIR_TABLEGEN_EXE mlir-tblgen)

  set(LLVM_MAIN_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm")
  set(LLVM_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include")
else()
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)

  set(MLIR_TABLEGEN_EXE mlir-tblgen)
  set(MLIR_INCLUDE_DIR "${MLIR_INCLUDE_DIRS}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Use -I (not -isystem) so our build's headers take priority over any
# stale system-installed LLVM/MLIR in /usr/local/include.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

#===----------------------------------------------------------------------===//
# LibTorch Configuration
#===----------------------------------------------------------------------===//

# Find libtorch for reading PyTorch exported programs.
# Set Torch_DIR to <libtorch>/share/cmake/Torch, or ensure python -c
# "import torch; print(torch.utils.cmake_prefix_path)" is on CMAKE_PREFIX_PATH.
option(RVML_ENABLE_TORCH_IMPORT "Enable PyTorch model import support" ON)

if(RVML_ENABLE_TORCH_IMPORT)
  # Allow vendored libtorch in third_party/
  if(NOT Torch_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch/TorchConfig.cmake")
    set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
  endif()

  find_package(Torch QUIET)
  if(Torch_FOUND)
    message(STATUS "Found libtorch: ${TORCH_INSTALL_PREFIX}")
  else()
    message(WARNING
      "libtorch not found. PyTorch import support will be disabled.\n"
      "Either:\n"
      "  1. Place libtorch in third_party/libtorch/\n"
      "  2. Set -DTorch_DIR=/path/to/libtorch/share/cmake/Torch\n"
      "  3. pip install torch and set -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')")
    set(RVML_ENABLE_TORCH_IMPORT OFF)
  endif()
endif()

#===----------------------------------------------------------------------===//
# RVML Configuration
#===----------------------------------------------------------------------===//

set(RVML_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(RVML_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(RVML_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")

include_directories(${RVML_INCLUDE_DIR})
include_directories(${RVML_GENERATED_INCLUDE_DIR})

include(AddRVML)

#===----------------------------------------------------------------------===//
# Subdirectories
#===----------------------------------------------------------------------===//

add_subdirectory(include/rvml)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)
add_subdirectory(tests)
