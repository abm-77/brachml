# AddBrachML.cmake - Helper functions for building BrachML targets.

# brachml_add_library(name sources... LINK_LIBS libs...)
# Adds an MLIR-style library with standard include paths.
function(brachml_add_library name)
  cmake_parse_arguments(ARG "" "" "LINK_LIBS" ${ARGN})
  add_mlir_library(${name}
    ${ARG_UNPARSED_ARGUMENTS}

    LINK_LIBS PUBLIC
    ${ARG_LINK_LIBS}
  )
endfunction()

# brachml_add_tool(name sources... LINK_LIBS libs...)
# Adds an executable tool linked against BrachML and MLIR libraries.
function(brachml_add_tool name)
  cmake_parse_arguments(ARG "" "" "LINK_LIBS" ${ARGN})
  add_llvm_executable(${name}
    ${ARG_UNPARSED_ARGUMENTS}
  )
  target_link_libraries(${name} PRIVATE ${ARG_LINK_LIBS})
  llvm_update_compile_flags(${name})
endfunction()
