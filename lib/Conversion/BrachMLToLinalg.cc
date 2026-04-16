#include <brachml/Conversion/Passes.h>
#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MLProgram/IR/MLProgram.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
namespace brachml {

namespace {

static mlir::Value createEmptyLike(mlir::OpBuilder &b, mlir::Location loc,
                                   mlir::RankedTensorType type) {
  return mlir::tensor::EmptyOp::create(b, loc, type.getShape(),
                                       type.getElementType());
}

static mlir::Value lowerAdd(mlir::OpBuilder &b, mlir::Location loc,
                            brachml::AddOp op, mlir::ValueRange inputs) {
  (void)op;
  auto out = createEmptyLike(
      b, loc, mlir::cast<mlir::RankedTensorType>(inputs[0].getType()));
  return mlir::linalg::AddOp::create(b, loc,
                                     mlir::ValueRange{inputs[0], inputs[1]},
                                     mlir::ValueRange{out})
      .getResult(0);
}

static mlir::Value lowerMatMul(mlir::OpBuilder &b, mlir::Location loc,
                               brachml::MatMulOp op, mlir::ValueRange inputs) {
  auto lhsTy = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto rhsTy = mlir::cast<mlir::RankedTensorType>(inputs[1].getType());
  auto elemTy =
      mlir::cast<mlir::RankedTensorType>(op.getType()).getElementType();
  auto outTy = mlir::RankedTensorType::get(
      {lhsTy.getShape()[0], rhsTy.getShape()[1]}, elemTy);
  auto out = createEmptyLike(b, loc, outTy);
  auto zero = mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));
  auto filled = mlir::linalg::FillOp::create(b, loc, zero.getResult(), out);
  return mlir::linalg::MatmulOp::create(b, loc,
                                        mlir::ValueRange{inputs[0], inputs[1]},
                                        mlir::ValueRange{filled.result()})
      .getResult(0);
}

static mlir::Value lowerReLU(mlir::OpBuilder &b, mlir::Location loc,
                             brachml::ReLUOp op, mlir::ValueRange inputs) {
  (void)op;
  auto ctx = b.getContext();
  auto inTy = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  int64_t rank = inTy.getRank();
  auto out = createEmptyLike(b, loc, inTy);
  auto generic = mlir::linalg::GenericOp::create(
      b, loc,
      /*resultTypes=*/mlir::TypeRange{out.getType()},
      /*inputs=*/mlir::ValueRange{inputs[0]},
      /*outputs=*/mlir::ValueRange{out},
      /*indexingMaps=*/
      llvm::SmallVector<mlir::AffineMap>{
          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx),
          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx),
      },
      /*iteratorTypes=*/
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      /*bodyBuilder=*/
      [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        auto x = args[0];
        auto zero =
            mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(x.getType()));
        mlir::Value result;
        if (mlir::isa<mlir::FloatType>(x.getType())) {
          result = mlir::arith::MaximumFOp::create(b, loc, x, zero.getResult())
                       .getResult();
        } else {
          result = mlir::arith::MaxSIOp::create(b, loc, x, zero.getResult())
                       .getResult();
        }
        mlir::linalg::YieldOp::create(b, loc, result);
      });
  return generic.getResult(0);
}

static mlir::Value lowerConv(mlir::OpBuilder &b, mlir::Location loc,
                             brachml::ConvOp op, mlir::ValueRange inputs) {
  assert(op.getGroups() == 1);

  auto inputTy = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto weightTy = mlir::cast<mlir::RankedTensorType>(inputs[1].getType());

  auto convInput = inputs[0];
  auto N = inputTy.getShape()[0];
  auto Cin = inputTy.getShape()[1];
  auto Cout = weightTy.getShape()[0];
  auto H = inputTy.getShape()[2];
  auto W = inputTy.getShape()[3];
  auto KH = weightTy.getShape()[2];
  auto KW = weightTy.getShape()[3];
  auto stride = op.getStrideAttr();
  auto padding = op.getPaddingAttr();
  auto dilation = op.getDilationAttr();
  auto outH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
  auto outW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
  auto elemTy =
      mlir::cast<mlir::RankedTensorType>(op.getType()).getElementType();
  auto outTy = mlir::RankedTensorType::get({N, Cout, outH, outW}, elemTy);
  auto zero = mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));

  bool needsPad = llvm::any_of(padding, [](int64_t p) { return p != 0; });
  if (needsPad) {
    auto paddedTy = mlir::RankedTensorType::get(
        {N, Cin, H + 2 * padding[0], W + 2 * padding[1]},
        inputTy.getElementType());
    convInput = mlir::tensor::PadOp::create(
                    b, loc, /*resultType=*/paddedTy,
                    /*source=*/inputs[0],
                    /*low=*/
                    llvm::SmallVector<mlir::OpFoldResult>{
                        b.getIndexAttr(0),          // N, no pad
                        b.getIndexAttr(0),          // C, no pad
                        b.getIndexAttr(padding[0]), // H
                        b.getIndexAttr(padding[1])  // W
                    },
                    /*high=*/
                    llvm::SmallVector<mlir::OpFoldResult>{
                        b.getIndexAttr(0), b.getIndexAttr(0),
                        b.getIndexAttr(padding[0]), b.getIndexAttr(padding[1])},
                    /*nofold=*/false)
                    .getResult();
  }

  auto out = createEmptyLike(b, loc, outTy);
  auto filled = mlir::linalg::FillOp::create(b, loc, zero.getResult(), out);
  mlir::Value result =
      mlir::linalg::Conv2DNchwFchwOp::create(
          b, loc, /*resultTensorTypes=*/mlir::TypeRange{outTy},
          /*inputs=*/mlir::ValueRange{convInput, /*weights=*/inputs[1]},
          /*outputs=*/mlir::ValueRange{filled.result()}, stride, dilation,
          /*attrs*/ {})
          .getResult(0);

  if (op.getBias()) {
    auto biasOut = createEmptyLike(
        b, loc, mlir::cast<mlir::RankedTensorType>(result.getType()));
    result = mlir::linalg::AddOp::create(
                 b, loc, mlir::ValueRange{result, /*bias=*/inputs[2]},
                 mlir::ValueRange{biasOut})
                 .getResult(0);
  }
  return result;
}

static mlir::Value lowerMaxPool(mlir::OpBuilder &b, mlir::Location loc,
                                brachml::MaxPool op, mlir::ValueRange inputs) {
  auto inputTy = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto elemTy = inputTy.getElementType();

  auto poolInput = inputs[0];
  auto N = inputTy.getShape()[0];
  auto Cin = inputTy.getShape()[1];
  auto H = inputTy.getShape()[2];
  auto W = inputTy.getShape()[3];
  auto kernelSize = op.getKernelSizeAttr();
  auto stride = op.getStrideAttr();
  auto padding = op.getPaddingAttr();
  auto dilation = op.getDilationAttr();
  auto outH =
      (H + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] +
      1;
  auto outW =
      (W + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] +
      1;
  auto outTy = mlir::RankedTensorType::get({N, Cin, outH, outW}, elemTy);

  bool needsPad = llvm::any_of(padding, [](int64_t p) { return p != 0; });
  if (needsPad) {
    auto paddedTy = mlir::RankedTensorType::get(
        {N, Cin, H + 2 * padding[0], W + 2 * padding[1]},
        inputTy.getElementType());
    poolInput = mlir::tensor::PadOp::create(
                    b, loc, /*resultType=*/paddedTy,
                    /*source=*/inputs[0],
                    /*low=*/
                    llvm::SmallVector<mlir::OpFoldResult>{
                        b.getIndexAttr(0),          // N, no pad
                        b.getIndexAttr(0),          // C, no pad
                        b.getIndexAttr(padding[0]), // H
                        b.getIndexAttr(padding[1])  // W
                    },
                    /*high=*/
                    llvm::SmallVector<mlir::OpFoldResult>{
                        b.getIndexAttr(0), b.getIndexAttr(0),
                        b.getIndexAttr(padding[0]), b.getIndexAttr(padding[1])},
                    /*nofold=*/false)
                    .getResult();
  }

  mlir::Value identity;
  if (mlir::isa<mlir::FloatType>(elemTy)) {
    identity =
        mlir::arith::ConstantOp::create(
            b, loc,
            b.getFloatAttr(elemTy, -std::numeric_limits<double>::infinity()))
            .getResult();
  } else {
    identity = mlir::arith::ConstantOp::create(
                   b, loc,
                   b.getIntegerAttr(elemTy, std::numeric_limits<int8_t>::min()))
                   .getResult();
  }

  auto out = createEmptyLike(b, loc, outTy);
  auto filled = mlir::linalg::FillOp::create(b, loc, identity, out);
  auto kernel = mlir::tensor::EmptyOp::create(
      b, loc, {kernelSize[0], kernelSize[1]}, elemTy);
  auto pool = mlir::linalg::PoolingNchwMaxOp::create(
      b, loc, /*resultTensorTypes=*/mlir::TypeRange{outTy},
      mlir::ValueRange{poolInput, kernel}, mlir::ValueRange{filled.result()},
      stride, dilation, {});
  return pool.getResult(0);
}

static mlir::Value lowerBatchNorm(mlir::OpBuilder &b, mlir::Location loc,
                                  brachml::BatchNormOp op,
                                  mlir::ValueRange inputs) {
  // TODO: inference-mode BN: output = (input - mean) / sqrt(var + eps) * w +
  // bias
  //   Decompose into linalg.generic ops that broadcast [C] params to [N,C,H,W].
  //   Precompute: scale = weight / sqrt(var + eps), offset = bias - mean *
  //   scale. Apply: linalg.generic input * scale + offset with affine map
  //   broadcasting dim 1 (channel) from [C] to [N,C,H,W].
  (void)b;
  (void)loc;
  (void)op;
  (void)inputs;
  return nullptr;
}

static mlir::Value lowerReshape(mlir::OpBuilder &b, mlir::Location loc,
                                brachml::ReshapeOp op,
                                mlir::ValueRange inputs) {
  // TODO:
  //   1. Get src shape and dst shape (from op.getSize()).
  //   2. If rank decreases: tensor::CollapseShapeOp with reassociation map.
  //   3. If rank increases: tensor::ExpandShapeOp with reassociation map.
  //   4. Compute reassociation greedily by matching dim products.
  (void)b;
  (void)loc;
  (void)op;
  (void)inputs;
  return nullptr;
}

static mlir::Value lowerPermute(mlir::OpBuilder &b, mlir::Location loc,
                                brachml::PermuteOp op,
                                mlir::ValueRange inputs) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto elemTy = inputType.getElementType();
  auto perm = op.getDims();
  llvm::SmallVector<int64_t> newDims;
  for (auto i = 0u; i < perm.size(); ++i) {
    newDims.push_back(inputType.getShape()[perm[i]]);
  }
  auto out = mlir::tensor::EmptyOp::create(b, loc, newDims, elemTy);
  auto transpose =
      mlir::linalg::TransposeOp::create(b, loc, inputs[0], out, perm);
  return transpose->getResult(0);
}

static mlir::Value lowerRequant(mlir::OpBuilder &b, mlir::Location loc,
                                brachml::RequantOp op,
                                mlir::ValueRange inputs) {
  // TODO: linalg.generic (identity maps, all-parallel) with body:
  //   a. arith.extsi i8 → i32
  //   b. arith.subi srcZP
  //   c. arith.sitofp → f32
  //   d. arith.mulf (srcScale / dstScale)
  //   e. math.roundeven (or arith.fptosi for truncation)
  //   f. arith.fptosi → i32
  //   g. arith.addi dstZP
  //   h. arith.maxsi -128, arith.minsi 127
  //   i. arith.trunci → i8
  (void)b;
  (void)loc;
  (void)op;
  (void)inputs;
  return nullptr;
}

static mlir::Value lowerLinear(mlir::OpBuilder &b, mlir::Location loc,
                               brachml::LinearOp op, mlir::ValueRange inputs) {
  // TODO:
  //   1. Matmul: same as lowerMatMul.
  //   2. If bias present: linalg::BroadcastOp [N] → [batch, N] + linalg::AddOp.
  (void)b;
  (void)loc;
  (void)op;
  (void)inputs;
  return nullptr;
}

static mlir::Value lowerConvBnRelu(mlir::OpBuilder &b, mlir::Location loc,
                                   brachml::ConvBnReluOp op,
                                   mlir::ValueRange inputs) {
  // TODO: fold BN into conv weights at lowering time, then emit conv + relu.
  //   folded_weight = conv_weight * (bn_weight / sqrt(bn_var + eps))
  //   folded_bias   = (conv_bias - bn_mean) * scale + bn_bias
  //   Then: lowerConv(folded) + lowerReLU.
  (void)b;
  (void)loc;
  (void)op;
  (void)inputs;
  return nullptr;
}

[[maybe_unused]] static mlir::Value dispatchLower(mlir::Operation *op,
                                                  mlir::ValueRange inputs,
                                                  mlir::OpBuilder &b,
                                                  mlir::Location loc) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
      .Case<brachml::AddOp>([&](auto o) { return lowerAdd(b, loc, o, inputs); })
      .Case<brachml::MatMulOp>(
          [&](auto o) { return lowerMatMul(b, loc, o, inputs); })
      .Case<brachml::ReLUOp>(
          [&](auto o) { return lowerReLU(b, loc, o, inputs); })
      .Case<brachml::ConvOp>(
          [&](auto o) { return lowerConv(b, loc, o, inputs); })
      .Case<brachml::MaxPool>(
          [&](auto o) { return lowerMaxPool(b, loc, o, inputs); })
      .Case<brachml::BatchNormOp>(
          [&](auto o) { return lowerBatchNorm(b, loc, o, inputs); })
      .Case<brachml::ReshapeOp>(
          [&](auto o) { return lowerReshape(b, loc, o, inputs); })
      .Case<brachml::PermuteOp>(
          [&](auto o) { return lowerPermute(b, loc, o, inputs); })
      .Case<brachml::RequantOp>(
          [&](auto o) { return lowerRequant(b, loc, o, inputs); })
      .Case<brachml::LinearOp>(
          [&](auto o) { return lowerLinear(b, loc, o, inputs); })
      .Case<brachml::ConvBnReluOp>(
          [&](auto o) { return lowerConvBnRelu(b, loc, o, inputs); })
      .Default([](auto o) -> mlir::Value {
        llvm_unreachable("unknown BrachML op in fused_region body");
      });
}

struct AddOpLowering : mlir::OpConversionPattern<brachml::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::AddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, lowerAdd(rewriter, op.getLoc(), op, adaptor.getOperands()));
    return mlir::success();
  }
};

struct MatMulOpLowering : mlir::OpConversionPattern<brachml::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::MatMulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, lowerMatMul(rewriter, op.getLoc(), op, adaptor.getOperands()));
    return mlir::success();
  }
};

struct ReLUOpLowering : mlir::OpConversionPattern<brachml::ReLUOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ReLUOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result = lowerReLU(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ConvOpLowering : mlir::OpConversionPattern<brachml::ConvOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ConvOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result = lowerConv(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct MaxPoolLowering : mlir::OpConversionPattern<brachml::MaxPool> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::MaxPool op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result =
        lowerMaxPool(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct BatchNormOpLowering : mlir::OpConversionPattern<brachml::BatchNormOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::BatchNormOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result =
        lowerBatchNorm(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ReshapeOpLowering : mlir::OpConversionPattern<brachml::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ReshapeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result =
        lowerReshape(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct PermuteOpLowering : mlir::OpConversionPattern<brachml::PermuteOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::PermuteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result =
        lowerPermute(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RequantOpLowering : mlir::OpConversionPattern<brachml::RequantOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::RequantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result =
        lowerRequant(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct LinearOpLowering : mlir::OpConversionPattern<brachml::LinearOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::LinearOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result = lowerLinear(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ConvBnReluOpLowering : mlir::OpConversionPattern<brachml::ConvBnReluOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(brachml::ConvBnReluOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto result =
        lowerConvBnRelu(rewriter, op.getLoc(), op, adaptor.getOperands());
    if (!result) return mlir::failure();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

// ── FusedRegionOpLowering
// ───────────────────────────────────────────────────── Emits an scf.forall
// over the output iteration space of the fused_region, then walks the body ops
// and lowers each one inside the loop body using dispatchLower. Results are
// threaded directly between ops via SSA — no intermediate tensor is written to
// memory between fused ops.
//
// TODO:
//   1. L1-aware tile size selection instead of fixed kTileSize.
//   2. Boundary clamping for dims not divisible by kTileSize.
//   3. Affine access maps for reduction op inputs (conv receptive field,
//      matmul K-dim) so those inputs are also tiled correctly rather than
//      passed as full tensors.
struct FusedRegionOpLowering
    : public mlir::OpConversionPattern<brachml::FusedRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(brachml::FusedRegionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: implement — see file header for design.
    //
    // Sketch:
    //   1. Get output shape from op.getResult(0).getType().
    //   2. Build lbs/ubs/steps for scf.forall over output tiles.
    //   3. Create empty output tensor; emit scf.forall with it as init.
    //   4. Inside forall body:
    //      a. Build tileOffsets/tileSizes/tileStrides from ivs.
    //      b. For each (block arg, external input): if shapes match, extract
    //         slice at tile offset; else pass full tensor.
    //      c. Walk body ops, call dispatchLower with remapped tile inputs,
    //         thread results through IRMapping.
    //      d. Emit tensor::ParallelInsertSliceOp for final tile into output.
    //   5. replaceOp with forall results.
    (void)adaptor;
    return mlir::failure();
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
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<BrachMLDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::ml_program::MLProgramDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FusedRegionOpLowering, AddOpLowering, MatMulOpLowering,
                 ReLUOpLowering, ConvOpLowering, MaxPoolLowering,
                 BatchNormOpLowering, ReshapeOpLowering, PermuteOpLowering,
                 RequantOpLowering, LinearOpLowering, ConvBnReluOpLowering>(
        &getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertBrachMLToLinalgPass() {
  return std::make_unique<ConvertBrachMLToLinalgPass>();
}

} // namespace brachml
