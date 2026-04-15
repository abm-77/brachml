#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>
#include <brachml/Transforms/Passes.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace brachml {

// indentifies PERMUTE(Transpose) -> MATMUL -> ADD pattern in graph and fuses
// them into LINEAR
struct LinearFusion : mlir::OpRewritePattern<brachml::AddOp> {

  LinearFusion(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<brachml::AddOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(brachml::AddOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto matmul = op.getLhs().getDefiningOp<brachml::MatMulOp>();
    auto bias = op.getRhs();
    if (!matmul) {
      matmul = op.getRhs().getDefiningOp<brachml::MatMulOp>();
      bias = op.getLhs();
    }
    if (!matmul) return mlir::failure();

    auto permute = matmul.getRhs().getDefiningOp<brachml::PermuteOp>();
    auto input = matmul.getLhs();
    if (!permute) return mlir::failure();

    if (!matmul->hasOneUse() || !permute->hasOneUse()) return mlir::failure();

    // must be transpose
    auto dims = permute.getDims();
    if (dims.size() != 2 || dims[0] != 1 || dims[1] != 0)
      return mlir::failure();

    auto fused = brachml::LinearOp::create(rewriter, op->getLoc(), op.getType(),
                                           input, permute.getInput(), bias);

    rewriter.replaceOp(op, fused);

    return mlir::success();
  }
};

// indentifies CONV -> BN -> RELU pattern in graph and fuses them into
// CONV-BN-RELU
struct ConvBnReluFusion : mlir::OpRewritePattern<brachml::ReLUOp> {
  ConvBnReluFusion(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<brachml::ReLUOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(brachml::ReLUOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto bn = op.getInput().getDefiningOp<brachml::BatchNormOp>();
    if (!bn) return mlir::failure();

    auto conv = bn.getInput().getDefiningOp<brachml::ConvOp>();
    if (!conv) return mlir::failure();

    if (!bn->hasOneUse() || !conv->hasOneUse()) return mlir::failure();

    auto fused = brachml::ConvBnReluOp::create(
        rewriter, op->getLoc(), op.getType(), conv.getInput(), conv.getWeight(),
        conv.getBias(), conv.getStrideAttr(), conv.getPaddingAttr(),
        conv.getDilationAttr(), conv.getTransposedAttr(),
        conv.getOutputPaddingAttr(), conv.getGroupsAttr(), bn.getWeight(),
        bn.getBias(), bn.getRunningMean(), bn.getRunningVar(), bn.getEpsAttr());

    rewriter.replaceOp(op, fused);

    return mlir::success();
  }
};

struct FusionPass
    : public mlir::PassWrapper<FusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)

  llvm::StringRef getArgument() const override { return "brachml-fusion"; }
  llvm::StringRef getDescription() const override {
    return "Fuse common sub-graphs into coarser ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<BrachMLDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvBnReluFusion>(&getContext());
    patterns.add<LinearFusion>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

} // namespace brachml
