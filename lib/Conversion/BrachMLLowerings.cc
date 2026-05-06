#include "BrachMLLowerings.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace brachml {

mlir::Value createEmpty(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::RankedTensorType type) {
  return mlir::tensor::EmptyOp::create(b, loc, type.getShape(),
                                       type.getElementType());
}

mlir::Value createEmptyLike(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value src) {
  auto type = mlir::cast<mlir::RankedTensorType>(src.getType());
  auto shape = type.getShape();
  llvm::SmallVector<mlir::Value> dynDims;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (mlir::ShapedType::isDynamic(shape[i])) {
      auto idx = mlir::arith::ConstantIndexOp::create(b, loc, i).getResult();
      dynDims.push_back(
          mlir::tensor::DimOp::create(b, loc, src, idx).getResult());
    }
  }
  return mlir::tensor::EmptyOp::create(b, loc, shape, type.getElementType(),
                                       dynDims)
      .getResult();
}

void buildZeroPadRegion(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::tensor::PadOp padOp, mlir::Type elemTy) {
  mlir::OpBuilder::InsertionGuard guard(b);
  mlir::Region &region = padOp.getRegion();
  auto srcRank =
      mlir::cast<mlir::RankedTensorType>(padOp.getSource().getType()).getRank();
  llvm::SmallVector<mlir::Type> argTypes(srcRank, b.getIndexType());
  llvm::SmallVector<mlir::Location> argLocs(srcRank, loc);
  mlir::Block *block = b.createBlock(&region, region.end(), argTypes, argLocs);
  b.setInsertionPointToStart(block);
  auto zero = mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));
  mlir::tensor::YieldOp::create(b, loc, zero.getResult());
}

mlir::Value padConvInputIfNeeded(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::Value input, brachml::ConvOp op) {
  auto padding = op.getPaddingAttr().asArrayRef();
  if (!llvm::any_of(padding, [](int64_t p) { return p != 0; })) return input;
  auto inputTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto shape = inputTy.getShape();
  auto elemTy = inputTy.getElementType();
  auto paddedTy = mlir::RankedTensorType::get({shape[0], shape[1],
                                               shape[2] + 2 * padding[0],
                                               shape[3] + 2 * padding[1]},
                                              elemTy);
  auto padOp = mlir::tensor::PadOp::create(
      b, loc, paddedTy, input,
      llvm::SmallVector<mlir::OpFoldResult>{
          b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(padding[0]),
          b.getIndexAttr(padding[1])},
      llvm::SmallVector<mlir::OpFoldResult>{
          b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(padding[0]),
          b.getIndexAttr(padding[1])},
      /*nofold=*/false);
  buildZeroPadRegion(b, loc, padOp, elemTy);
  return padOp.getResult();
}

mlir::Value padMaxPoolInputIfNeeded(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value input, brachml::MaxPool op) {
  auto padding = op.getPaddingAttr().asArrayRef();
  if (!llvm::any_of(padding, [](int64_t p) { return p != 0; })) return input;
  auto inputTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto shape = inputTy.getShape();
  auto elemTy = inputTy.getElementType();
  auto paddedTy = mlir::RankedTensorType::get({shape[0], shape[1],
                                               shape[2] + 2 * padding[0],
                                               shape[3] + 2 * padding[1]},
                                              elemTy);
  auto padOp = mlir::tensor::PadOp::create(
      b, loc, paddedTy, input,
      llvm::SmallVector<mlir::OpFoldResult>{
          b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(padding[0]),
          b.getIndexAttr(padding[1])},
      llvm::SmallVector<mlir::OpFoldResult>{
          b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(padding[0]),
          b.getIndexAttr(padding[1])},
      /*nofold=*/false);
  buildZeroPadRegion(b, loc, padOp, elemTy);
  return padOp.getResult();
}

mlir::Value lowerAdd(mlir::OpBuilder &b, mlir::Location loc, brachml::AddOp op,
                     mlir::ValueRange inputs, mlir::Value out) {
  auto lhsTy = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto rhsTy = mlir::cast<mlir::RankedTensorType>(inputs[1].getType());
  auto outTy = mlir::cast<mlir::RankedTensorType>(out.getType());

  if (lhsTy == rhsTy) {
    return mlir::linalg::AddOp::create(b, loc,
                                       mlir::ValueRange{inputs[0], inputs[1]},
                                       mlir::ValueRange{out})
        .getResult(0);
  }

  // Broadcasting: right-aligned affine maps, size-1 dims map to constant 0.
  auto ctx = b.getContext();
  int64_t outRank = outTy.getRank();

  auto buildBroadcastMap = [&](mlir::RankedTensorType ty) -> mlir::AffineMap {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    int64_t offset = outRank - ty.getRank();
    for (int64_t i = 0; i < ty.getRank(); ++i) {
      if (ty.getDimSize(i) == 1)
        exprs.push_back(mlir::getAffineConstantExpr(0, ctx));
      else
        exprs.push_back(mlir::getAffineDimExpr(offset + i, ctx));
    }
    return mlir::AffineMap::get(outRank, 0, exprs, ctx);
  };

  auto idMap = mlir::AffineMap::getMultiDimIdentityMap(outRank, ctx);
  return mlir::linalg::GenericOp::create(
             b, loc, mlir::TypeRange{outTy},
             mlir::ValueRange{inputs[0], inputs[1]}, mlir::ValueRange{out},
             llvm::SmallVector<mlir::AffineMap>{
                 buildBroadcastMap(lhsTy), buildBroadcastMap(rhsTy), idMap},
             llvm::SmallVector<mlir::utils::IteratorType>(
                 outRank, mlir::utils::IteratorType::parallel),
             [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
               mlir::Value sum;
               if (mlir::isa<mlir::FloatType>(args[0].getType()))
                 sum = mlir::arith::AddFOp::create(b, loc, args[0], args[1])
                           .getResult();
               else
                 sum = mlir::arith::AddIOp::create(b, loc, args[0], args[1])
                           .getResult();
               mlir::linalg::YieldOp::create(b, loc, sum);
             })
      .getResult(0);
}

mlir::Value lowerMatMul(mlir::OpBuilder &b, mlir::Location loc,
                        brachml::MatMulOp op, mlir::ValueRange inputs,
                        mlir::Value out) {
  auto elemTy =
      mlir::cast<mlir::RankedTensorType>(out.getType()).getElementType();
  auto zero = mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));
  auto filled = mlir::linalg::FillOp::create(b, loc, zero.getResult(), out);
  return mlir::linalg::MatmulOp::create(b, loc,
                                        mlir::ValueRange{inputs[0], inputs[1]},
                                        mlir::ValueRange{filled.result()})
      .getResult(0);
}

mlir::Value lowerReLU(mlir::OpBuilder &b, mlir::Location loc,
                      brachml::ReLUOp op, mlir::ValueRange inputs,
                      mlir::Value out) {
  (void)op;
  auto ctx = b.getContext();
  auto outTy = mlir::cast<mlir::RankedTensorType>(out.getType());
  int64_t rank = outTy.getRank();
  auto generic = mlir::linalg::GenericOp::create(
      b, loc, mlir::TypeRange{outTy}, mlir::ValueRange{inputs[0]},
      mlir::ValueRange{out},
      llvm::SmallVector<mlir::AffineMap>{
          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx),
          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx)},
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        auto x = args[0];
        auto zero =
            mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(x.getType()));
        mlir::Value result;
        if (mlir::isa<mlir::FloatType>(x.getType()))
          result = mlir::arith::MaximumFOp::create(b, loc, x, zero.getResult())
                       .getResult();
        else
          result = mlir::arith::MaxSIOp::create(b, loc, x, zero.getResult())
                       .getResult();
        mlir::linalg::YieldOp::create(b, loc, result);
      });
  return generic.getResult(0);
}

mlir::Value lowerConv(mlir::OpBuilder &b, mlir::Location loc,
                      brachml::ConvOp op, mlir::Value paddedInput,
                      mlir::Value weight, mlir::Value bias, mlir::Value out) {
  assert(op.getGroups() == 1 && "grouped conv not supported");
  assert(!op.getTransposed() && "transposed conv not supported");

  auto outTy = mlir::cast<mlir::RankedTensorType>(out.getType());
  auto elemTy = outTy.getElementType();

  auto zero = mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(elemTy));
  auto filled = mlir::linalg::FillOp::create(b, loc, zero.getResult(), out);
  mlir::Value result =
      mlir::linalg::Conv2DNchwFchwOp::create(
          b, loc, mlir::TypeRange{filled.result().getType()},
          mlir::ValueRange{paddedInput, weight},
          mlir::ValueRange{filled.result()}, op.getStrideAttr(),
          op.getDilationAttr(), mlir::ArrayRef<mlir::NamedAttribute>{})
          .getResult(0);

  if (bias) {
    auto ctx = b.getContext();
    auto biasOut = createEmptyLike(b, loc, result);
    auto idMap = mlir::AffineMap::getMultiDimIdentityMap(4, ctx);
    auto biasMap =
        mlir::AffineMap::get(4, 0, {mlir::getAffineDimExpr(1, ctx)}, ctx);
    result =
        mlir::linalg::GenericOp::create(
            b, loc, mlir::TypeRange{result.getType()},
            mlir::ValueRange{result, bias}, mlir::ValueRange{biasOut},
            llvm::SmallVector<mlir::AffineMap>{idMap, biasMap, idMap},
            llvm::SmallVector<mlir::utils::IteratorType>(
                4, mlir::utils::IteratorType::parallel),
            [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
              mlir::Value sum;
              if (mlir::isa<mlir::FloatType>(args[0].getType()))
                sum = mlir::arith::AddFOp::create(b, loc, args[0], args[1])
                          .getResult();
              else
                sum = mlir::arith::AddIOp::create(b, loc, args[0], args[1])
                          .getResult();
              mlir::linalg::YieldOp::create(b, loc, sum);
            })
            .getResult(0);
  }
  return result;
}

mlir::Value lowerMaxPool(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::MaxPool op, mlir::Value paddedInput,
                         mlir::Value out) {
  auto outTy = mlir::cast<mlir::RankedTensorType>(out.getType());
  auto elemTy = outTy.getElementType();
  auto kernelSize = op.getKernelSizeAttr().asArrayRef();

  mlir::Value identity;
  if (mlir::isa<mlir::FloatType>(elemTy))
    identity =
        mlir::arith::ConstantOp::create(
            b, loc,
            b.getFloatAttr(elemTy, -std::numeric_limits<double>::infinity()))
            .getResult();
  else
    identity = mlir::arith::ConstantOp::create(
                   b, loc,
                   b.getIntegerAttr(elemTy, std::numeric_limits<int8_t>::min()))
                   .getResult();

  auto filled = mlir::linalg::FillOp::create(b, loc, identity, out);
  auto kernel = mlir::tensor::EmptyOp::create(
      b, loc, llvm::SmallVector<int64_t>{kernelSize[0], kernelSize[1]}, elemTy);
  return mlir::linalg::PoolingNchwMaxOp::create(
             b, loc, mlir::TypeRange{filled.result().getType()},
             mlir::ValueRange{paddedInput, kernel},
             mlir::ValueRange{filled.result()}, op.getStrideAttr(),
             op.getDilationAttr(), mlir::ArrayRef<mlir::NamedAttribute>{})
      .getResult(0);
}

mlir::Value lowerBatchNorm(mlir::OpBuilder &b, mlir::Location loc,
                           brachml::BatchNormOp op, mlir::ValueRange inputs,
                           mlir::Value out) {
  auto ctx = b.getContext();
  auto outTy = mlir::cast<mlir::RankedTensorType>(out.getType());
  auto elemTy = outTy.getElementType();

  // Operand order: input, [weight], [bias], mean, var
  int idx = 0;
  mlir::Value inputVal = inputs[idx++];
  mlir::Value weightVal = op.getWeight() ? inputs[idx++] : mlir::Value{};
  mlir::Value biasVal = op.getBias() ? inputs[idx++] : mlir::Value{};
  mlir::Value meanVal = inputs[idx++];
  mlir::Value varVal = inputs[idx++];

  double epsVal = op.getEps().convertToDouble();
  mlir::Value epsConst =
      mlir::arith::ConstantOp::create(b, loc, b.getFloatAttr(elemTy, epsVal))
          .getResult();

  auto identMap = mlir::AffineMap::getMultiDimIdentityMap(4, ctx);
  auto chanMap =
      mlir::AffineMap::get(4, 0, {mlir::getAffineDimExpr(1, ctx)}, ctx);

  llvm::SmallVector<mlir::Value> genericInputs = {inputVal, meanVal, varVal};
  llvm::SmallVector<mlir::AffineMap> maps = {identMap, chanMap, chanMap};
  if (weightVal) {
    genericInputs.push_back(weightVal);
    maps.push_back(chanMap);
  }
  if (biasVal) {
    genericInputs.push_back(biasVal);
    maps.push_back(chanMap);
  }
  maps.push_back(identMap);

  bool hasWeight = (bool)weightVal;
  bool hasBias = (bool)biasVal;

  auto generic = mlir::linalg::GenericOp::create(
      b, loc, mlir::TypeRange{outTy}, mlir::ValueRange{genericInputs},
      mlir::ValueRange{out}, maps,
      llvm::SmallVector<mlir::utils::IteratorType>(
          4, mlir::utils::IteratorType::parallel),
      [epsConst, hasWeight, hasBias](mlir::OpBuilder &b, mlir::Location loc,
                                     mlir::ValueRange args) {
        mlir::Value x = args[0], mean = args[1], var = args[2];
        int argIdx = 3;
        mlir::Value w = hasWeight ? args[argIdx++] : mlir::Value{};
        mlir::Value bi = hasBias ? args[argIdx++] : mlir::Value{};

        mlir::Value varPlusEps =
            mlir::arith::AddFOp::create(b, loc, var, epsConst).getResult();
        mlir::Value sqrtVar =
            mlir::math::SqrtOp::create(b, loc, varPlusEps).getResult();
        mlir::Value scale;
        if (w) {
          scale = mlir::arith::DivFOp::create(b, loc, w, sqrtVar).getResult();
        } else {
          mlir::Value one = mlir::arith::ConstantOp::create(
                                b, loc, b.getFloatAttr(x.getType(), 1.0))
                                .getResult();
          scale = mlir::arith::DivFOp::create(b, loc, one, sqrtVar).getResult();
        }

        mlir::Value xSub =
            mlir::arith::SubFOp::create(b, loc, x, mean).getResult();
        mlir::Value scaled =
            mlir::arith::MulFOp::create(b, loc, xSub, scale).getResult();
        mlir::Value result;
        if (bi)
          result = mlir::arith::AddFOp::create(b, loc, scaled, bi).getResult();
        else
          result = scaled;
        mlir::linalg::YieldOp::create(b, loc, result);
      });
  return generic.getResult(0);
}

mlir::Value lowerReshape(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::ReshapeOp op, mlir::ValueRange inputs) {
  auto srcTy = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto dstShape = op.getSize();
  auto elemTy = srcTy.getElementType();
  int64_t srcRank = srcTy.getRank();
  int64_t dstRank = (int64_t)dstShape.size();
  auto flatTy = mlir::RankedTensorType::get({srcTy.getNumElements()}, elemTy);
  auto dstTy = mlir::RankedTensorType::get(dstShape, elemTy);

  mlir::Value flat = inputs[0];
  if (srcRank != 1) {
    llvm::SmallVector<mlir::ReassociationIndices> collapseMap(1);
    for (int64_t i = 0; i < srcRank; ++i) collapseMap[0].push_back(i);
    flat = mlir::tensor::CollapseShapeOp::create(b, loc, flatTy, inputs[0],
                                                 collapseMap)
               .getResult();
  }
  if (dstRank == 1) return flat;

  llvm::SmallVector<mlir::ReassociationIndices> expandMap(1);
  for (int64_t i = 0; i < dstRank; ++i) expandMap[0].push_back(i);
  return mlir::tensor::ExpandShapeOp::create(b, loc, dstTy, flat, expandMap)
      .getResult();
}

mlir::Value lowerPermute(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::PermuteOp op, mlir::ValueRange inputs) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
  auto elemTy = inputType.getElementType();
  auto perm = op.getDims();
  llvm::SmallVector<int64_t> newDims;
  for (auto i = 0u; i < perm.size(); ++i)
    newDims.push_back(inputType.getShape()[perm[i]]);
  auto out = mlir::tensor::EmptyOp::create(b, loc, newDims, elemTy);
  auto transpose =
      mlir::linalg::TransposeOp::create(b, loc, inputs[0], out, perm);
  return transpose->getResult(0);
}

mlir::Value lowerRequant(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::RequantOp op, mlir::ValueRange inputs,
                         mlir::Value out) {
  auto ctx = b.getContext();
  auto outTy = mlir::cast<mlir::RankedTensorType>(out.getType());
  int64_t rank = outTy.getRank();

  mlir::Value ratio =
      mlir::arith::ConstantOp::create(
          b, loc,
          b.getFloatAttr(b.getF32Type(),
                         op.getSrcScale().convertToDouble() /
                             op.getDstScale().convertToDouble()))
          .getResult();
  mlir::Value srcZPConst = mlir::arith::ConstantOp::create(
                               b, loc, b.getI32IntegerAttr(op.getSrcZP()))
                               .getResult();
  mlir::Value dstZPConst = mlir::arith::ConstantOp::create(
                               b, loc, b.getI32IntegerAttr(op.getDstZP()))
                               .getResult();
  mlir::Value i8min =
      mlir::arith::ConstantOp::create(
          b, loc, b.getI32IntegerAttr(std::numeric_limits<int8_t>::min()))
          .getResult();
  mlir::Value i8max =
      mlir::arith::ConstantOp::create(
          b, loc, b.getI32IntegerAttr(std::numeric_limits<int8_t>::max()))
          .getResult();

  auto generic = mlir::linalg::GenericOp::create(
      b, loc, mlir::TypeRange{outTy}, mlir::ValueRange{inputs[0]},
      mlir::ValueRange{out},
      llvm::SmallVector<mlir::AffineMap>{
          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx),
          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx)},
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [ratio, srcZPConst, dstZPConst, i8min,
       i8max](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value sext =
            mlir::arith::ExtSIOp::create(b, loc, b.getI32Type(), args[0])
                .getResult();
        mlir::Value deZP =
            mlir::arith::SubIOp::create(b, loc, sext, srcZPConst).getResult();
        mlir::Value floatVal =
            mlir::arith::SIToFPOp::create(b, loc, b.getF32Type(), deZP)
                .getResult();
        mlir::Value scaled =
            mlir::arith::MulFOp::create(b, loc, floatVal, ratio).getResult();
        mlir::Value intVal =
            mlir::arith::FPToSIOp::create(b, loc, b.getI32Type(), scaled)
                .getResult();
        mlir::Value rezp =
            mlir::arith::AddIOp::create(b, loc, intVal, dstZPConst).getResult();
        mlir::Value clamp =
            mlir::arith::MaxSIOp::create(b, loc, rezp, i8min).getResult();
        clamp = mlir::arith::MinSIOp::create(b, loc, clamp, i8max).getResult();
        mlir::Value narrow =
            mlir::arith::TruncIOp::create(b, loc, b.getI8Type(), clamp)
                .getResult();
        mlir::linalg::YieldOp::create(b, loc, narrow);
      });
  return generic.getResult(0);
}

mlir::Value dispatchLowerInto(mlir::Operation *op, mlir::ValueRange inputs,
                              mlir::Value out, mlir::OpBuilder &b,
                              mlir::Location loc) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
      .Case<brachml::AddOp>(
          [&](auto o) { return lowerAdd(b, loc, o, inputs, out); })
      .Case<brachml::MatMulOp>(
          [&](auto o) { return lowerMatMul(b, loc, o, inputs, out); })
      .Case<brachml::ReLUOp>(
          [&](auto o) { return lowerReLU(b, loc, o, inputs, out); })
      .Case<brachml::ConvOp>([&](auto o) {
        mlir::Value bias = inputs.size() > 2 ? inputs[2] : mlir::Value{};
        return lowerConv(b, loc, o, inputs[0], inputs[1], bias, out);
      })
      .Case<brachml::MaxPool>(
          [&](auto o) { return lowerMaxPool(b, loc, o, inputs[0], out); })
      .Case<brachml::BatchNormOp>(
          [&](auto o) { return lowerBatchNorm(b, loc, o, inputs, out); })
      .Case<brachml::ReshapeOp>(
          [&](auto o) { return lowerReshape(b, loc, o, inputs); })
      .Case<brachml::PermuteOp>(
          [&](auto o) { return lowerPermute(b, loc, o, inputs); })
      .Case<brachml::RequantOp>(
          [&](auto o) { return lowerRequant(b, loc, o, inputs, out); })
      .Default([](auto o) -> mlir::Value {
        llvm_unreachable("unknown BrachML op in fused_region body");
      });
}

} // namespace brachml
