rvml_add_library(RVMLConversion
  RVMLToLLVM.cpp

  LINK_LIBS
  RVMLDialect
  MLIRLLVMDialect
  MLIRTransforms
)
