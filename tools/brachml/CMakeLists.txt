brachml_add_tool(brachml
  main.cc

  LINK_LIBS
  BrachMLDialect
  BrachMLConversion
  BrachMLTransforms
  MLIRPass
  MLIRTransforms
  MLIRLLVMToLLVMIRTranslation
)
