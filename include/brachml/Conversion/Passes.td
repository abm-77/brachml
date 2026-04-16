include "mlir/Pass/PassBase.td"

def ConvertBrachMLToLLVM : Pass<"convert-brachml-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower BrachML dialect to LLVM dialect";
  let constructor = "brachml::createConvertBrachMLToLLVMPass()";
  let dependentDialects = [
    "brachml::BrachMLDialect",
    "mlir::LLVM::LLVMDialect",
  ];
}

def ConvertBrachMLToLinalg : Pass<"convert-brachml-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lower BrachML ops to linalg / tensor / arith";
  let constructor = "brachml::createConvertBrachMLToLinalgPass()";
  let dependentDialects = [
    "brachml::BrachMLDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
}

