#include "AccessPatterns.h"
#include "BrachMLLowerings.h"
#include "FusedRegionLowering.h"

#include <brachml/Conversion/Passes.h>
#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MLProgram/IR/MLProgram.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace brachml {

namespace {

struct AddOpLowering : mlir::OpConversionPattern<brachml::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::AddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    rewriter.replaceOp(
        op, lowerAdd(rewriter, op.getLoc(), op, adaptor.getOperands(), out));
    return mlir::success();
  }
};

struct MatMulOpLowering : mlir::OpConversionPattern<brachml::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::MatMulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    rewriter.replaceOp(
        op, lowerMatMul(rewriter, op.getLoc(), op, adaptor.getOperands(), out));
    return mlir::success();
  }
};

struct ReLUOpLowering : mlir::OpConversionPattern<brachml::ReLUOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ReLUOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    rewriter.replaceOp(
        op, lowerReLU(rewriter, op.getLoc(), op, adaptor.getOperands(), out));
    return mlir::success();
  }
};

struct ConvOpLowering : mlir::OpConversionPattern<brachml::ConvOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ConvOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto inputs = adaptor.getOperands();
    auto padded = padConvInputIfNeeded(rewriter, op.getLoc(), inputs[0], op);
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    mlir::Value bias = inputs.size() > 2 ? inputs[2] : mlir::Value{};
    rewriter.replaceOp(
        op, lowerConv(rewriter, op.getLoc(), op, padded, inputs[1], bias, out));
    return mlir::success();
  }
};

struct MaxPoolLowering : mlir::OpConversionPattern<brachml::MaxPool> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::MaxPool op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto inputs = adaptor.getOperands();
    auto padded = padMaxPoolInputIfNeeded(rewriter, op.getLoc(), inputs[0], op);
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    rewriter.replaceOp(op,
                       lowerMaxPool(rewriter, op.getLoc(), op, padded, out));
    return mlir::success();
  }
};

struct BatchNormOpLowering : mlir::OpConversionPattern<brachml::BatchNormOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::BatchNormOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    rewriter.replaceOp(op, lowerBatchNorm(rewriter, op.getLoc(), op,
                                          adaptor.getOperands(), out));
    return mlir::success();
  }
};

struct ReshapeOpLowering : mlir::OpConversionPattern<brachml::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ReshapeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, lowerReshape(rewriter, op.getLoc(), op, adaptor.getOperands()));
    return mlir::success();
  }
};

struct PermuteOpLowering : mlir::OpConversionPattern<brachml::PermuteOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::PermuteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, lowerPermute(rewriter, op.getLoc(), op, adaptor.getOperands()));
    return mlir::success();
  }
};

struct RequantOpLowering : mlir::OpConversionPattern<brachml::RequantOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::RequantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    auto out = createEmpty(rewriter, op.getLoc(), resTy);
    rewriter.replaceOp(op, lowerRequant(rewriter, op.getLoc(), op,
                                        adaptor.getOperands(), out));
    return mlir::success();
  }
};

struct ConvertBrachMLToLinalgPass
    : public mlir::PassWrapper<ConvertBrachMLToLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertBrachMLToLinalgPass)

  llvm::StringRef getArgument() const override {
    return "convert-brachml-to-linalg";
  }
  llvm::StringRef getDescription() const override {
    return "Lower BrachML ops to linalg / tensor / arith";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns
        .add<AddOpLowering, MatMulOpLowering, ReLUOpLowering, ConvOpLowering,
             MaxPoolLowering, BatchNormOpLowering, ReshapeOpLowering,
             PermuteOpLowering, RequantOpLowering, FusedRegionOpLowering>(ctx);

    mlir::ConversionTarget target(*ctx);
    target.addIllegalDialect<brachml::BrachMLDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertBrachMLToLinalgPass() {
  return std::make_unique<ConvertBrachMLToLinalgPass>();
}

} // namespace brachml
