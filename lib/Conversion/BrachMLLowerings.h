#pragma once

#include <brachml/Dialect/Basic/BrachMLOps.h>

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace brachml {

// Each lowering takes an explicit `out` destination tensor. The caller (either
// a standalone op pattern or the fused_region tiling) is responsible for
// creating the output with the correct shape (static for standalone calls,
// dynamic tile-shape for tiled calls).

mlir::Value lowerAdd(mlir::OpBuilder &b, mlir::Location loc, brachml::AddOp op,
                     mlir::ValueRange inputs, mlir::Value out);

mlir::Value lowerMatMul(mlir::OpBuilder &b, mlir::Location loc,
                        brachml::MatMulOp op, mlir::ValueRange inputs,
                        mlir::Value out);

mlir::Value lowerReLU(mlir::OpBuilder &b, mlir::Location loc,
                      brachml::ReLUOp op, mlir::ValueRange inputs,
                      mlir::Value out);

// `paddedInput` must already include padding. Use `padConvInputIfNeeded` first
// to pad the full tensor once, then slice or pass through to this function.
mlir::Value lowerConv(mlir::OpBuilder &b, mlir::Location loc,
                      brachml::ConvOp op, mlir::Value paddedInput,
                      mlir::Value weight, mlir::Value bias, mlir::Value out);

// Analogous to lowerConv: caller handles padding, this runs the pool.
mlir::Value lowerMaxPool(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::MaxPool op, mlir::Value paddedInput,
                         mlir::Value out);

mlir::Value lowerBatchNorm(mlir::OpBuilder &b, mlir::Location loc,
                           brachml::BatchNormOp op, mlir::ValueRange inputs,
                           mlir::Value out);

mlir::Value lowerReshape(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::ReshapeOp op, mlir::ValueRange inputs);

mlir::Value lowerPermute(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::PermuteOp op, mlir::ValueRange inputs);

mlir::Value lowerRequant(mlir::OpBuilder &b, mlir::Location loc,
                         brachml::RequantOp op, mlir::ValueRange inputs,
                         mlir::Value out);

// Create a tensor.empty of the same type as `type`. Shape must be fully static.
mlir::Value createEmpty(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::RankedTensorType type);

// Create a tensor.empty of the same type as `src` (possibly dynamic), using
// tensor.dim to materialize dynamic dimensions.
mlir::Value createEmptyLike(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value src);

// Build a zero-valued pad region for an already-created tensor.pad op.
void buildZeroPadRegion(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::tensor::PadOp padOp, mlir::Type elemTy);

// If the conv has non-zero padding, pad the input and return the padded tensor.
// Otherwise return `input` unchanged.
mlir::Value padConvInputIfNeeded(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::Value input, brachml::ConvOp op);

// Same for max_pool.
mlir::Value padMaxPoolInputIfNeeded(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value input, brachml::MaxPool op);

// Dispatch to the correct lower*Into* function for a body op inside
// fused_region. The op must already have its inputs sliced / its output
// tensor prepared.
mlir::Value dispatchLowerInto(mlir::Operation *op, mlir::ValueRange inputs,
                              mlir::Value out, mlir::OpBuilder &b,
                              mlir::Location loc);

} // namespace brachml
