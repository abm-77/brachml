brachml_add_library(BrachMLConversion
  BrachMLToLLVM.cc
  BrachMLToLinalg.cc

  DEPENDS
  BrachMLConversionIncGen

  LINK_LIBS
  BrachMLDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
