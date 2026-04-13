brachml_add_library(BrachMLTransforms
  FusionPass.cc
  QuantizationPass.cc

  DEPENDS
  BrachMLTransformsIncGen

  LINK_LIBS
  BrachMLDialect
  BrachMLQDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
