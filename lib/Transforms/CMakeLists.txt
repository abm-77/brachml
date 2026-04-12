rvml_add_library(RVMLTransforms
  Passes.cpp

  LINK_LIBS
  RVMLDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
