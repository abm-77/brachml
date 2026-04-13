brachml_add_library(BrachMLConversion
  BrachMLToLLVM.cc

  DEPENDS
  BrachMLConversionIncGen

  LINK_LIBS
  BrachMLDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
