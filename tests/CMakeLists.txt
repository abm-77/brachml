# Lit-based testing for RVML.

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

set(RVML_TEST_DEPENDS
  # rvml-opt
  # rvml-import
  FileCheck
  count
  not
)

add_custom_target(check-rvml
  COMMAND ${Python3_EXECUTABLE} -m lit
    ${CMAKE_CURRENT_BINARY_DIR}
    -v
    --param rvml_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  DEPENDS ${RVML_TEST_DEPENDS}
  COMMENT "Running RVML regression tests"
)
