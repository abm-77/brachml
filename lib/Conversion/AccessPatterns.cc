#include "AccessPatterns.h"

#include <brachml/Dialect/Basic/BrachMLOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace brachml {

namespace {

// Unit-stride vector of length `n`.
static llvm::SmallVector<mlir::OpFoldResult> unitStrides(mlir::OpBuilder &b,
                                                          unsigned n) {
  llvm::SmallVector<mlir::OpFoldResult> strides;
  strides.reserve(n);
  for (unsigned i = 0; i < n; ++i) strides.push_back(b.getIndexAttr(1));
  return strides;
}

// Materialize an OpFoldResult as an index SSA value.
static mlir::Value asIndexValue(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::OpFoldResult ofr) {
  if (auto v = llvm::dyn_cast_if_present<mlir::Value>(ofr)) return v;
  auto attr = llvm::cast<mlir::IntegerAttr>(
      llvm::cast<mlir::Attribute>(ofr));
  return mlir::arith::ConstantIndexOp::create(b, loc, attr.getInt()).getResult();
}

// Identity access — the input is sliced with the exact same offsets/sizes as
// the output.
static InputAccess identityAccess(
    llvm::ArrayRef<mlir::OpFoldResult> outOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> outSizes, mlir::OpBuilder &b) {
  InputAccess a;
  a.full = false;
  a.offsets.assign(outOffsets.begin(), outOffsets.end());
  a.sizes.assign(outSizes.begin(), outSizes.end());
  a.strides = unitStrides(b, outSizes.size());
  return a;
}

// Access for a 1D per-channel tensor (bn weight, bias, mean, var; conv bias).
// Uses the channel dim of the output (dim 1 for 4D, dim 0 for 1D).
static InputAccess channelAccess(
    int64_t channelDimInOutput,
    llvm::ArrayRef<mlir::OpFoldResult> outOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> outSizes, mlir::OpBuilder &b) {
  InputAccess a;
  a.full = false;
  a.offsets = {outOffsets[channelDimInOutput]};
  a.sizes = {outSizes[channelDimInOutput]};
  a.strides = unitStrides(b, 1);
  return a;
}

// Broadcast access for a lower-rank input over a higher-rank output.
// Right-aligned: input dim i maps to output dim (outRank - inRank + i).
// Size-1 input dims always access index 0 (broadcast).
static InputAccess broadcastAccess(
    mlir::RankedTensorType inTy,
    llvm::ArrayRef<mlir::OpFoldResult> outOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> outSizes, mlir::OpBuilder &b) {
  InputAccess a;
  a.full = false;
  int64_t outRank = (int64_t)outOffsets.size();
  int64_t inRank = inTy.getRank();
  int64_t offset = outRank - inRank;
  for (int64_t i = 0; i < inRank; ++i) {
    if (inTy.getDimSize(i) == 1) {
      a.offsets.push_back(b.getIndexAttr(0));
      a.sizes.push_back(b.getIndexAttr(1));
    } else {
      a.offsets.push_back(outOffsets[offset + i]);
      a.sizes.push_back(outSizes[offset + i]);
    }
  }
  a.strides = unitStrides(b, inRank);
  return a;
}

// Receptive-field access for a 4D [N, C, H, W] input of a conv/maxpool
// whose 4D output is [N, Cout, OH, OW].
//
// offset[0..1] = output[0..1]  (N and Cin aren't tiled by the spatial-only
//                              tiling strategy, but we inherit outOffsets[0])
// spatial offset_h = outOffset_h * stride_h
// spatial size_h   = (outSize_h - 1) * stride_h + (KH - 1) * dilation_h + 1
static InputAccess receptiveFieldAccess(
    int64_t fullCin,
    llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> dilation,
    int64_t KH, int64_t KW,
    llvm::ArrayRef<mlir::OpFoldResult> outOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> outSizes, mlir::OpBuilder &b,
    mlir::Location loc) {
  InputAccess a;
  a.full = false;
  // N, C: full (not spatial-tiled).
  a.offsets.push_back(outOffsets[0]);
  a.sizes.push_back(outSizes[0]);
  a.offsets.push_back(b.getIndexAttr(0));
  a.sizes.push_back(b.getIndexAttr(fullCin));

  auto computeSpatial = [&](mlir::OpFoldResult outOff, mlir::OpFoldResult outSz,
                             int64_t s, int64_t d, int64_t K) {
    mlir::Value outOffV = asIndexValue(b, loc, outOff);
    mlir::Value outSzV = asIndexValue(b, loc, outSz);
    mlir::Value sV =
        mlir::arith::ConstantIndexOp::create(b, loc, s).getResult();
    mlir::Value offIn =
        mlir::arith::MulIOp::create(b, loc, outOffV, sV).getResult();
    mlir::Value one =
        mlir::arith::ConstantIndexOp::create(b, loc, 1).getResult();
    mlir::Value sizeMinus1 =
        mlir::arith::SubIOp::create(b, loc, outSzV, one).getResult();
    mlir::Value term =
        mlir::arith::MulIOp::create(b, loc, sizeMinus1, sV).getResult();
    mlir::Value rf = mlir::arith::ConstantIndexOp::create(
                         b, loc, (K - 1) * d + 1)
                         .getResult();
    mlir::Value sizeIn =
        mlir::arith::AddIOp::create(b, loc, term, rf).getResult();
    return std::make_pair(mlir::OpFoldResult(offIn), mlir::OpFoldResult(sizeIn));
  };

  auto [offH, sizeH] = computeSpatial(outOffsets[2], outSizes[2], stride[0],
                                       dilation[0], KH);
  auto [offW, sizeW] = computeSpatial(outOffsets[3], outSizes[3], stride[1],
                                       dilation[1], KW);
  a.offsets.push_back(offH);
  a.sizes.push_back(sizeH);
  a.offsets.push_back(offW);
  a.sizes.push_back(sizeW);
  a.strides = unitStrides(b, 4);
  return a;
}

} // namespace

llvm::SmallVector<InputAccess>
getInputAccesses(mlir::Operation *op,
                 llvm::ArrayRef<mlir::OpFoldResult> outOffsets,
                 llvm::ArrayRef<mlir::OpFoldResult> outSizes,
                 mlir::OpBuilder &b, mlir::Location loc) {
  return llvm::TypeSwitch<mlir::Operation *,
                           llvm::SmallVector<InputAccess>>(op)

      .Case<brachml::ReLUOp>([&](auto) {
        return llvm::SmallVector<InputAccess>{
            identityAccess(outOffsets, outSizes, b)};
      })

      .Case<brachml::RequantOp>([&](auto) {
        return llvm::SmallVector<InputAccess>{
            identityAccess(outOffsets, outSizes, b)};
      })

      .Case<brachml::AddOp>([&](brachml::AddOp add) {
        auto lhsTy =
            mlir::cast<mlir::RankedTensorType>(add.getLhs().getType());
        auto rhsTy =
            mlir::cast<mlir::RankedTensorType>(add.getRhs().getType());
        int64_t outRank = (int64_t)outSizes.size();
        auto accessFor = [&](mlir::RankedTensorType ty) {
          if (ty.getRank() == outRank)
            return identityAccess(outOffsets, outSizes, b);
          return broadcastAccess(ty, outOffsets, outSizes, b);
        };
        return llvm::SmallVector<InputAccess>{accessFor(lhsTy),
                                               accessFor(rhsTy)};
      })

      .Case<brachml::BatchNormOp>([&](brachml::BatchNormOp bn) {
        // Operands: input, [weight], [bias], mean, var
        llvm::SmallVector<InputAccess> accesses;
        accesses.push_back(identityAccess(outOffsets, outSizes, b));
        if (bn.getWeight())
          accesses.push_back(channelAccess(/*channelDim=*/1, outOffsets,
                                            outSizes, b));
        if (bn.getBias())
          accesses.push_back(channelAccess(1, outOffsets, outSizes, b));
        accesses.push_back(channelAccess(1, outOffsets, outSizes, b));
        accesses.push_back(channelAccess(1, outOffsets, outSizes, b));
        return accesses;
      })

      .Case<brachml::ConvOp>([&](brachml::ConvOp conv) {
        // Only 2D conv with groups=1, not transposed is supported here.
        if (conv.getGroups() != 1 || conv.getTransposed())
          return llvm::SmallVector<InputAccess>{};
        auto inputTy =
            mlir::cast<mlir::RankedTensorType>(conv.getInput().getType());
        auto weightTy =
            mlir::cast<mlir::RankedTensorType>(conv.getWeight().getType());
        int64_t Cin = inputTy.getShape()[1];
        // Conv's post-padding Cin may still be static even if input is padded.
        int64_t KH = weightTy.getShape()[2];
        int64_t KW = weightTy.getShape()[3];
        auto stride = conv.getStrideAttr().asArrayRef();
        auto dilation = conv.getDilationAttr().asArrayRef();

        llvm::SmallVector<InputAccess> accesses;
        accesses.push_back(receptiveFieldAccess(Cin, stride, dilation, KH, KW,
                                                 outOffsets, outSizes, b, loc));
        // Weight is full (Cout is not tiled under the spatial-tiling strategy).
        InputAccess weightAcc;
        weightAcc.full = true;
        accesses.push_back(weightAcc);
        if (conv.getBias()) {
          InputAccess biasAcc;
          biasAcc.full = true;
          accesses.push_back(biasAcc);
        }
        return accesses;
      })

      .Case<brachml::MaxPool>([&](brachml::MaxPool pool) {
        auto inputTy =
            mlir::cast<mlir::RankedTensorType>(pool.getInput().getType());
        int64_t Cin = inputTy.getShape()[1];
        auto kernel = pool.getKernelSizeAttr().asArrayRef();
        auto stride = pool.getStrideAttr().asArrayRef();
        auto dilation = pool.getDilationAttr().asArrayRef();
        return llvm::SmallVector<InputAccess>{receptiveFieldAccess(
            Cin, stride, dilation, kernel[0], kernel[1], outOffsets, outSizes,
            b, loc)};
      })

      .Case<brachml::MatMulOp>([&](brachml::MatMulOp mm) {
        // Only 2D matmul supported here.
        auto lhsTy =
            mlir::cast<mlir::RankedTensorType>(mm.getLhs().getType());
        auto rhsTy =
            mlir::cast<mlir::RankedTensorType>(mm.getRhs().getType());
        if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
          return llvm::SmallVector<InputAccess>{};
        int64_t K = lhsTy.getShape()[1];

        // Output tile is [tM, tN]. LHS needs [tM, K], RHS needs [K, tN].
        InputAccess lhs;
        lhs.full = false;
        lhs.offsets = {outOffsets[0], b.getIndexAttr(0)};
        lhs.sizes = {outSizes[0], b.getIndexAttr(K)};
        lhs.strides = unitStrides(b, 2);

        InputAccess rhs;
        rhs.full = false;
        rhs.offsets = {b.getIndexAttr(0), outOffsets[1]};
        rhs.sizes = {b.getIndexAttr(K), outSizes[1]};
        rhs.strides = unitStrides(b, 2);

        return llvm::SmallVector<InputAccess>{lhs, rhs};
      })

      .Default([](auto) { return llvm::SmallVector<InputAccess>{}; });
}

} // namespace brachml
