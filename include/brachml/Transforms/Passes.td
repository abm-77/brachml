
include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"brachml-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse common sub-graphs into coarser ops";
  let constructor = "brachml::createFusionPass()";
  let dependentDialects = ["brachml::BrachMLDialect"];
}

def BeamSearchFusionPass : Pass<"brachml-beam-fusion", "mlir::func::FuncOp"> {
  let summary = "BEAM search over op chains to find the highest-scoring fusion plan";
  let description = [{
    Runs after brachml-fusion. The initial fusion pass collapses known-good
    patterns (conv+bn+relu, linear) into coarser ops, shrinking the search
    space. This pass then explores all remaining fusable chains up to
    `max-depth` ops long and greedily selects the best `beam-width` fusion
    plans, emitting each winning group as a brachml.fused_region.
  }];
  let constructor = "brachml::createBeamSearchFusionPass()";
  let dependentDialects = ["brachml::BrachMLDialect"];
}

