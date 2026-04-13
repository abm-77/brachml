
include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"brachml-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse common sub-graphs into coarser ops";
  let constructor = "brachml::createFusionPass()";
  let dependentDialects = ["brachml::BrachMLDialect"];
}

def QuantizationPass : Pass<"brachml-quantize", "mlir::func::FuncOp"> {
  let summary = "Quantize BrachML ops to fixed-point representations";
  let constructor = "brachml::createQuantizationPass()";
  let dependentDialects = [
    "brachml::BrachMLDialect",
    "brachml::q::BrachMLQDialect",
  ];
}
