brachml_add_tool(brachml-opt
  main.cc

  LINK_LIBS
  BrachMLDialect
  BrachMLConversion
  BrachMLTransforms

  # Dialects
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMLProgramDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect

  # Dialect transforms (bufferize, memref expand-strided-metadata, etc.)
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRControlFlowTransforms
  MLIRLinalgTransforms
  MLIRMLProgramTransforms
  MLIRMemRefTransforms
  MLIRSCFTransforms
  MLIRMathTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms

  # Conversions to LLVM dialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRVectorToLLVMPass
  MLIRVectorToSCF

  # Other conversions
  MLIRLinalgToStandard

  # Pass / opt infra
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
  MLIRTransforms
)
