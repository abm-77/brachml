brachml_add_library(BrachMLQDialect
  BrachMLQDialect.cc
  BrachMLQOps.cc

  DEPENDS
  BrachMLQOpsIncGen

  LINK_LIBS
  MLIRIR
  MLIRSupport
)
