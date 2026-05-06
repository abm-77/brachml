#include <brachml/Transforms/Passes.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace brachml {

namespace {

// ─── Lane-width table ─────────────────────────────────────────────────────────
// Cortex-A53 has 128-bit NEON; pick a lane count per element type.
//
// TODO: return lane count for `elemTy`.
//   - f32 / i32 → 4
//   - f16 / i16 → 8
//   - i8        → 16
//   - fall through / unknown → 4
[[maybe_unused]] static int64_t neonLaneCount(mlir::Type elemTy) {
  (void)elemTy;
  return 4;
}

// ─── Per-op vector-size policy ────────────────────────────────────────────────
// Returns one vector size per iterator of `op`. Empty vector = "don't
// vectorize this op" (pass leaves it alone; later linalg-to-loops handles it).
//
// Policy sketch:
//   - Inner-most parallel iterator          → lane count for the op's
//                                             data element type.
//   - Reduction iterator with static range  → full range (unroll).
//   - Other parallel iterators              → 1 (stays as a loop).
//
// TODO:
//   1. Walk op.getDpsInputOperands() (or getResults()) to find the element
//      type used by the op.
//   2. Call op.getIteratorTypesArray() to get iterator kinds.
//   3. Call op.getStaticLoopRanges() to get static iteration extents.
//   4. Walk iterators from innermost to outermost, set the first parallel
//      iterator's vector size to neonLaneCount(elemTy).
//   5. For each iterator: if reduction + static range, set size to that range.
//   6. Return the resulting sizes (or {} if we don't want to vectorize).
[[maybe_unused]] static llvm::SmallVector<int64_t>
pickVectorSizes(mlir::linalg::LinalgOp op) {
  (void)op;
  // TODO: implement the policy above.
  return {};
}

// ─── Pass ─────────────────────────────────────────────────────────────────────
struct VectorizePass
    : public mlir::PassWrapper<VectorizePass,
                                mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizePass)

  llvm::StringRef getArgument() const override { return "brachml-vectorize"; }
  llvm::StringRef getDescription() const override {
    return "Vectorize linalg ops with NEON lane width on the innermost "
           "parallel dim; unroll static reduction dims.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    // TODO: walk every linalg::LinalgOp in the function and collect a
    // worklist. Can't transform during the walk because linalg::vectorize
    // replaces the op.
    //
    //   llvm::SmallVector<mlir::linalg::LinalgOp> worklist;
    //   getOperation().walk([&](mlir::linalg::LinalgOp op) {
    //     worklist.push_back(op);
    //   });
    //
    // TODO: for each op in the worklist:
    //   1. auto sizes = pickVectorSizes(op); if empty, skip.
    //   2. rewriter.setInsertionPoint(op);
    //   3. (void)mlir::linalg::vectorize(rewriter, op, sizes);
    //      On failure the op stays scalar — later linalg-to-loops handles it.
    //
    // TODO: decide whether to also vectorize tensor.pad (see
    // mlir::linalg::populatePadOpVectorizationPatterns) once we turn on
    // padded convs.
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createVectorizePass() {
  return std::make_unique<VectorizePass>();
}

} // namespace brachml
