project(rvml-runtime LANGUAGES C)

option(RVML_RUNTIME_CROSS_COMPILE "Cross-compile the runtime for a bare-metal target" OFF)

set(RVML_RUNTIME_SOURCES
  # runtime.c
  # memory.c
  # math.c
)

if(RVML_RUNTIME_SOURCES)
  add_library(rvml-runtime STATIC ${RVML_RUNTIME_SOURCES})

  target_include_directories(rvml-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # Freestanding: no standard library
  target_compile_options(rvml-runtime PRIVATE
    -ffreestanding
    -nostdlib
    -fno-builtin
  )

  if(RVML_RUNTIME_CROSS_COMPILE)
    # Cross-compilation settings will be configured via a CMake toolchain file:
    #   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/<target>.cmake
    message(STATUS "RVML runtime: cross-compiling for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()
