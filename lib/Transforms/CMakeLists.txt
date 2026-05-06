brachml_add_library(BrachMLTransforms
  BeamSearchFusionPass.cc
  VectorizePass.cc

  DEPENDS
  BrachMLTransformsIncGen

  LINK_LIBS
  BrachMLDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
)
