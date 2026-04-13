brachml_add_library(BrachMLDialect
  BrachMLDialect.cc
  BrachMLOps.cc

  DEPENDS
  BrachMLOpsIncGen

  LINK_LIBS
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
)
