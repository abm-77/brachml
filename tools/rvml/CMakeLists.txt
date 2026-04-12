rvml_add_tool(rvml
  main.cpp

  LINK_LIBS
  RVMLDialect
  RVMLConversion
  RVMLTransforms
  MLIRPass
  MLIRTransforms
  MLIRLLVMToLLVMIRTranslation
)
