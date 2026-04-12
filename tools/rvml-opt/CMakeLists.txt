rvml_add_tool(rvml-opt
  main.cpp

  LINK_LIBS
  RVMLDialect
  RVMLConversion
  RVMLTransforms
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
