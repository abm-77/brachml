
include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"brachml-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse common sub-graphs into coarser ops";
  let constructor = "brachml::createFusionPass()";
  let dependentDialects = ["brachml::BrachMLDialect"];
}

def QuantizationPass : Pass<"brachml-quantize", "mlir::func::FuncOp"> {
  let summary = "Attach quantization attributes to BrachML ops from a JSON "
                "calibration file";
  let constructor = "brachml::createQuantizationPass()";
  let dependentDialects = ["brachml::BrachMLDialect"];
}
