brachml_add_tool(brachml-opt
  main.cc

  LINK_LIBS
  BrachMLDialect
  BrachMLConversion
  BrachMLTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMLProgramDialect
  MLIROptLib
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
