# Lit-based testing for BrachML.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

set(BRACHML_TEST_DEPENDS
  brachml-opt
  FileCheck
  count
  not
)

add_custom_target(check-brachml
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-lit
    ${CMAKE_CURRENT_BINARY_DIR}
    -v
    --param brachml_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  DEPENDS ${BRACHML_TEST_DEPENDS}
  COMMENT "Running BrachML regression tests"
)
