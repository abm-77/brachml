cmake_minimum_required(VERSION 3.20)
project(brachml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

#===----------------------------------------------------------------------===//
# LLVM/MLIR Configuration
#===----------------------------------------------------------------------===//

# If MLIR_DIR is not set, build from vendored third-party/llvm-project.
if(NOT MLIR_DIR)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/llvm-project/llvm/CMakeLists.txt")
    message(FATAL_ERROR
      "Vendored LLVM not found at third-party/llvm-project. Either:\n"
      "  1. Run: git submodule update --init --recursive\n"
      "  2. Set -DMLIR_DIR=/path/to/lib/cmake/mlir")
  endif()

  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
  set(LLVM_TARGETS_TO_BUILD "host;RISCV" CACHE STRING "")
  set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")

  add_subdirectory(third-party/llvm-project/llvm EXCLUDE_FROM_ALL)

  set(MLIR_MAIN_SRC_DIR "${CMAKE_SOURCE_DIR}/third-party/llvm-project/mlir")
  set(MLIR_INCLUDE_DIR "${MLIR_MAIN_SRC_DIR}/include")
  set(MLIR_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/third-party/llvm-project/llvm/tools/mlir/include")
  set(MLIR_TABLEGEN_EXE mlir-tblgen)

  set(LLVM_MAIN_SRC_DIR "${CMAKE_SOURCE_DIR}/third-party/llvm-project/llvm")
  set(LLVM_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/third-party/llvm-project/llvm/include")
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

# BEFORE so our build's headers beat any stale /usr/local/include/mlir.
# SYSTEM so warnings in LLVM/MLIR headers don't fire under -Werror.
include_directories(BEFORE SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(BEFORE SYSTEM ${MLIR_INCLUDE_DIRS})

add_compile_options(-Werror -Wall -Wextra -Wno-unused-parameter)

#===----------------------------------------------------------------------===//
# BrachML Configuration
#===----------------------------------------------------------------------===//

set(BRACHML_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(BRACHML_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(BRACHML_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")

# Collect tool binaries in build/bin/ for lit tests.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(${BRACHML_INCLUDE_DIR})
include_directories(${BRACHML_GENERATED_INCLUDE_DIR})

include(AddBrachML)

#===----------------------------------------------------------------------===//
# Subdirectories
#===----------------------------------------------------------------------===//

add_subdirectory(include/brachml)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)
add_subdirectory(tests)
add_subdirectory(python)
