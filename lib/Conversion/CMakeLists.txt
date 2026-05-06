brachml_add_library(BrachMLConversion
  AccessPatterns.cc
  BrachMLLowerings.cc
  BrachMLToLLVM.cc
  BrachMLToLinalg.cc
  FusedRegionLowering.cc

  DEPENDS
  BrachMLConversionIncGen

  LINK_LIBS
  BrachMLDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
