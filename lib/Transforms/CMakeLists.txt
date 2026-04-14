brachml_add_library(BrachMLTransforms
  FusionPass.cc
  QuantizationPass.cc

  DEPENDS
  BrachMLTransformsIncGen

  LINK_LIBS
  BrachMLDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
