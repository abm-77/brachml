set(LLVM_TARGET_DEFINITIONS BrachMLOps.td)
mlir_tablegen(BrachMLOps.h.inc -gen-op-decls)
mlir_tablegen(BrachMLOps.cpp.inc -gen-op-defs)
mlir_tablegen(BrachMLDialect.h.inc -gen-dialect-decls -dialect=brachml)
mlir_tablegen(BrachMLDialect.cpp.inc -gen-dialect-defs -dialect=brachml)
mlir_tablegen(_brachml_ops_gen.py -gen-python-op-bindings -bind-dialect=brachml)
add_public_tablegen_target(BrachMLOpsIncGen)

# Copy generated Python bindings into the source tree so they're importable.
add_custom_command(
  TARGET BrachMLOpsIncGen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/_brachml_ops_gen.py
    ${CMAKE_SOURCE_DIR}/python/brachml_import/dialects/_brachml_ops_gen.py
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/_brachml_ops_gen.py
    ${MLIR_PYTHON_PACKAGES_DIR}/mlir_core/mlir/dialects/_brachml_ops_gen.py
  COMMENT "Copying BrachML Python op bindings"
)
