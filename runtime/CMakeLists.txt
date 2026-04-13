project(brachml-runtime LANGUAGES C)

option(BRACHML_RUNTIME_CROSS_COMPILE "Cross-compile the runtime for a bare-metal target" OFF)

set(BRACHML_RUNTIME_SOURCES
  # runtime.c
  # memory.c
  # math.c
)

if(BRACHML_RUNTIME_SOURCES)
  add_library(brachml-runtime STATIC ${BRACHML_RUNTIME_SOURCES})

  target_include_directories(brachml-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # Freestanding: no standard library
  target_compile_options(brachml-runtime PRIVATE
    -ffreestanding
    -nostdlib
    -fno-builtin
  )

  if(BRACHML_RUNTIME_CROSS_COMPILE)
    # Cross-compilation settings will be configured via a CMake toolchain file:
    #   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/<target>.cmake
    message(STATUS "BrachML runtime: cross-compiling for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()
