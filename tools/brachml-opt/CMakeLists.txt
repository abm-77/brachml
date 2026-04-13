brachml_add_tool(brachml-opt
  main.cc

  LINK_LIBS
  BrachMLDialect
  BrachMLQDialect
  BrachMLConversion
  BrachMLTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMLProgramDialect
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
