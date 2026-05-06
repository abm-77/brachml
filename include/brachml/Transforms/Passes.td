
include "mlir/Pass/PassBase.td"

def BeamSearchFusionPass : Pass<"brachml-beam-fusion", "mlir::func::FuncOp"> {
  let summary = "BEAM search over op chains to find the highest-scoring fusion plan";
  let description = [{
    Explores all fusable op chains up to `max-depth` ops long and greedily
    selects the best `beam-width` fusion plans, emitting each winning group
    as a brachml.fused_region.
  }];
  let constructor = "brachml::createBeamSearchFusionPass()";
  let dependentDialects = ["brachml::BrachMLDialect"];
}

def VectorizePass : Pass<"brachml-vectorize", "mlir::func::FuncOp"> {
  let summary = "Vectorize linalg ops with NEON lane width on the innermost parallel dim";
  let description = [{
    Calls mlir::linalg::vectorize on every linalg op in the function with a
    heuristic: innermost parallel iterator gets the element-type lane count
    (4 for f32, 16 for i8, ...), static reduction iterators are unrolled
    (vector size = range), everything else stays a loop. Masked transfers
    handle dynamic tile dims.
  }];
  let constructor = "brachml::createVectorizePass()";
  let dependentDialects = ["mlir::vector::VectorDialect"];
}

