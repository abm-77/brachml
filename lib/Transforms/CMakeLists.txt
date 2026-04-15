brachml_add_library(BrachMLTransforms
  FusionPass.cc
  BeamSearchFusionPass.cc

  DEPENDS
  BrachMLTransformsIncGen

  LINK_LIBS
  BrachMLDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
