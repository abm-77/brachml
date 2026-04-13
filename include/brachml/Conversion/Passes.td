include "mlir/Pass/PassBase.td"

def ConvertBrachMLToLLVM : Pass<"convert-brachml-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower BrachML dialect to LLVM dialect";
  let constructor = "brachml::createConvertBrachMLToLLVMPass()";
  let dependentDialects = [
    "brachml::BrachMLDialect",
    "mlir::LLVM::LLVMDialect",
  ];
}
