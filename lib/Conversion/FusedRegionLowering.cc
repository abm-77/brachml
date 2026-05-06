#include "FusedRegionLowering.h"

#include "AccessPatterns.h"
#include "BrachMLLowerings.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace brachml {

namespace {

constexpr int64_t kTileSize = 32;

struct Tile {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
};

// Clamped tile size: min(kTileSize, outSize - iv).
static mlir::Value clampedSize(mlir::OpBuilder &b, mlir::Location loc,
                                int64_t outSize, mlir::Value iv) {
  auto ub = mlir::arith::ConstantIndexOp::create(b, loc, outSize).getResult();
  auto step = mlir::arith::ConstantIndexOp::create(b, loc, kTileSize).getResult();
  auto remaining = mlir::arith::SubIOp::create(b, loc, ub, iv).getResult();
  return mlir::arith::MinSIOp::create(b, loc, step, remaining).getResult();
}

// The index of the operand that is defined by a previous op in `body`.
// Returns -1 if no such operand exists (this op is the first in the chain).
// If multiple such operands exist, returns the first. Our op set has at most
// one chain input per op in practice.
static int chainInputIdx(mlir::Operation *op, mlir::Block &body) {
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    auto *def = operand.getDefiningOp();
    if (def && def->getBlock() == &body) return (int)i;
  }
  return -1;
}

// The "data input" of `op` — the operand that propagates the chain's data
// through the op. If there's a chain input, that's it. Otherwise operand 0
// (the first op in a chain feeds its data from operand 0 by convention).
static int dataInputIdx(mlir::Operation *op, mlir::Block &body) {
  int ci = chainInputIdx(op, body);
  return ci >= 0 ? ci : 0;
}

// Materialize an InputAccess as an actual value: slice the source tensor if
// the access isn't full.
static mlir::Value materializeAccess(mlir::OpBuilder &b, mlir::Location loc,
                                      mlir::Value src, const InputAccess &a) {
  if (a.full) return src;
  return mlir::tensor::ExtractSliceOp::create(b, loc, src, a.offsets, a.sizes,
                                               a.strides)
      .getResult();
}

// Slice shape helpers: construct a shape vector where dynamic sizes are
// ShapedType::kDynamic and the corresponding Value is in `dynDims`.
static void splitStaticDynamic(llvm::ArrayRef<mlir::OpFoldResult> sizes,
                                llvm::SmallVectorImpl<int64_t> &staticShape,
                                llvm::SmallVectorImpl<mlir::Value> &dynDims) {
  for (auto ofr : sizes) {
    if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(ofr)) {
      staticShape.push_back(llvm::cast<mlir::IntegerAttr>(attr).getInt());
    } else {
      staticShape.push_back(mlir::ShapedType::kDynamic);
      dynDims.push_back(llvm::cast<mlir::Value>(ofr));
    }
  }
}

// Build tensor.empty for a tile with possibly dynamic dims.
static mlir::Value buildTileEmpty(mlir::OpBuilder &b, mlir::Location loc,
                                   llvm::ArrayRef<mlir::OpFoldResult> sizes,
                                   mlir::Type elemTy) {
  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<mlir::Value> dynDims;
  splitStaticDynamic(sizes, staticShape, dynDims);
  return mlir::tensor::EmptyOp::create(b, loc, staticShape, elemTy, dynDims)
      .getResult();
}

// Sequential fallback: lower each op in body order, no tiling.
static mlir::LogicalResult sequentialLower(
    brachml::FusedRegionOp op, mlir::ValueRange adaptorInputs,
    mlir::ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  mlir::Block &body = op.getBody().front();
  mlir::IRMapping mapping;
  for (auto [arg, ext] : llvm::zip(body.getArguments(), adaptorInputs))
    mapping.map(arg, ext);

  mlir::Value result;
  for (auto &bodyOp : body.without_terminator()) {
    llvm::SmallVector<mlir::Value> mapped;
    for (auto operand : bodyOp.getOperands()) {
      auto m = mapping.lookup(operand);
      assert(m && "brachml-to-linalg: unmapped operand");
      mapped.push_back(m);
    }
    // For the sequential path, build the output statically from the op's
    // own result type.
    auto resTy = mlir::cast<mlir::RankedTensorType>(bodyOp.getResult(0).getType());

    // Conv/MaxPool need padded input and the lowering signature skips the
    // padding step; do the padding here.
    if (auto convOp = mlir::dyn_cast<brachml::ConvOp>(&bodyOp)) {
      auto padded = padConvInputIfNeeded(rewriter, loc, mapped[0], convOp);
      auto out = createEmpty(rewriter, loc, resTy);
      mlir::Value bias = mapped.size() > 2 ? mapped[2] : mlir::Value{};
      result = lowerConv(rewriter, loc, convOp, padded, mapped[1], bias, out);
    } else if (auto poolOp = mlir::dyn_cast<brachml::MaxPool>(&bodyOp)) {
      auto padded = padMaxPoolInputIfNeeded(rewriter, loc, mapped[0], poolOp);
      auto out = createEmpty(rewriter, loc, resTy);
      result = lowerMaxPool(rewriter, loc, poolOp, padded, out);
    } else {
      mlir::Value out;
      if (!mlir::isa<brachml::ReshapeOp, brachml::PermuteOp>(&bodyOp))
        out = createEmpty(rewriter, loc, resTy);
      result = dispatchLowerInto(&bodyOp, mapped, out, rewriter, loc);
    }
    mapping.map(bodyOp.getResult(0), result);
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

// Generic tiled lowering. Tiles the last `numTileDims` dims of the region
// output, propagates tile shapes backward through the body, then lowers each
// op forward with sliced inputs and tile-shaped outputs.
static mlir::LogicalResult tiledLower(
    brachml::FusedRegionOp op, mlir::ValueRange adaptorInputs,
    mlir::ConversionPatternRewriter &rewriter,
    mlir::RankedTensorType outTy, int numTileDims) {
  auto loc = op.getLoc();
  mlir::Block &body = op.getBody().front();
  auto outShape = outTy.getShape();
  int64_t rank = outTy.getRank();

  // Collect body ops (skip the terminator).
  llvm::SmallVector<mlir::Operation *> ops;
  for (auto &bodyOp : body.without_terminator()) ops.push_back(&bodyOp);

  // Precompute chain / data input indices and preliminary access-pattern
  // availability — any op without an access pattern aborts the tiled path.
  llvm::SmallVector<int> dataIdx(ops.size());
  llvm::SmallVector<int> chainIdx(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    chainIdx[i] = chainInputIdx(ops[i], body);
    dataIdx[i] = dataInputIdx(ops[i], body);
  }

  // Pre-pad conv / maxpool inputs once, outside the loop. We track which
  // block-arg index maps to which (possibly padded) external SSA value.
  llvm::SmallVector<mlir::Value> externals(adaptorInputs.begin(),
                                            adaptorInputs.end());
  for (auto *bodyOp : ops) {
    if (auto c = mlir::dyn_cast<brachml::ConvOp>(bodyOp)) {
      int argIdx = -1;
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(c.getInput())) {
        if (ba.getOwner() == &body) argIdx = ba.getArgNumber();
      }
      if (argIdx < 0) return sequentialLower(op, adaptorInputs, rewriter);
      externals[argIdx] =
          padConvInputIfNeeded(rewriter, loc, externals[argIdx], c);
    } else if (auto p = mlir::dyn_cast<brachml::MaxPool>(bodyOp)) {
      int argIdx = -1;
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(p.getInput())) {
        if (ba.getOwner() == &body) argIdx = ba.getArgNumber();
      }
      if (argIdx < 0) return sequentialLower(op, adaptorInputs, rewriter);
      externals[argIdx] =
          padMaxPoolInputIfNeeded(rewriter, loc, externals[argIdx], p);
    }
  }

  // Set up the scf.forall over the last `numTileDims` output dims.
  llvm::SmallVector<mlir::OpFoldResult> lbs, ubs, steps;
  for (int d = rank - numTileDims; d < rank; ++d) {
    lbs.push_back(rewriter.getIndexAttr(0));
    ubs.push_back(rewriter.getIndexAttr(outShape[d]));
    steps.push_back(rewriter.getIndexAttr(kTileSize));
  }

  auto initTensor = createEmpty(rewriter, loc, outTy);
  auto forall = mlir::scf::ForallOp::create(
      rewriter, loc, lbs, ubs, steps, mlir::ValueRange{initTensor},
      std::nullopt);

  rewriter.setInsertionPointToStart(forall.getBody());
  auto ivs = forall.getInductionVars();

  // Build the region-output tile (one entry per output dim). Untiled leading
  // dims are 0..dimSize; tiled trailing dims use ivs + clamped sizes.
  Tile regionTile;
  regionTile.offsets.resize(rank);
  regionTile.sizes.resize(rank);
  for (int d = 0; d < rank; ++d) {
    if (d < rank - numTileDims) {
      regionTile.offsets[d] = rewriter.getIndexAttr(0);
      regionTile.sizes[d] = rewriter.getIndexAttr(outShape[d]);
    } else {
      int idx = d - (rank - numTileDims);
      regionTile.offsets[d] = ivs[idx];
      regionTile.sizes[d] =
          clampedSize(rewriter, loc, outShape[d], ivs[idx]);
    }
  }

  // Phase 1: backward walk to compute each op's output and data-input tiles.
  // opOutTile[i] is the tile each op must PRODUCE. The region's last op
  // produces the region tile; op[i-1] produces whatever op[i]'s data input
  // needs as a slice.
  llvm::SmallVector<Tile> opOutTile(ops.size());
  llvm::SmallVector<Tile> opDataInTile(ops.size());
  Tile curOut = regionTile;
  for (int i = (int)ops.size() - 1; i >= 0; --i) {
    opOutTile[i] = curOut;
    // Compute accesses to determine the data input tile shape.
    auto accesses = getInputAccesses(ops[i], curOut.offsets, curOut.sizes,
                                      rewriter, loc);
    if (accesses.empty()) {
      // No access pattern available — can't tile this chain. Roll back.
      // Easiest path: erase the forall and fall back to sequential.
      rewriter.setInsertionPoint(forall);
      rewriter.eraseOp(forall);
      return sequentialLower(op, adaptorInputs, rewriter);
    }
    const InputAccess &da = accesses[dataIdx[i]];
    if (da.full) {
      // Data input isn't actually a tile — unusual. Treat as same as output.
      opDataInTile[i] = curOut;
    } else {
      opDataInTile[i] = Tile{da.offsets, da.sizes};
    }
    curOut = opDataInTile[i];
  }

  // Phase 2: forward walk. For each op, gather input values (chain or sliced
  // block arg), build the output tile tensor, and lower.
  llvm::SmallVector<mlir::Value> opResults(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    auto *curOp = ops[i];
    auto accesses = getInputAccesses(curOp, opOutTile[i].offsets,
                                      opOutTile[i].sizes, rewriter, loc);

    llvm::SmallVector<mlir::Value> inputs;
    for (auto [opIdx, operand] : llvm::enumerate(curOp->getOperands())) {
      if ((int)opIdx == chainIdx[i]) {
        // Result of a previous op in this body.
        auto *def = operand.getDefiningOp();
        auto it = llvm::find(ops, def);
        assert(it != ops.end() && "chain input not in body");
        inputs.push_back(opResults[(size_t)(it - ops.begin())]);
      } else {
        // External block arg — possibly pre-padded in `externals`.
        auto ba = mlir::cast<mlir::BlockArgument>(operand);
        inputs.push_back(
            materializeAccess(rewriter, loc, externals[ba.getArgNumber()],
                               accesses[opIdx]));
      }
    }

    // Reshape/Permute don't take an output tensor.
    mlir::Value outTile;
    auto resTy =
        mlir::cast<mlir::RankedTensorType>(curOp->getResult(0).getType());
    if (!mlir::isa<brachml::ReshapeOp, brachml::PermuteOp>(curOp))
      outTile = buildTileEmpty(rewriter, loc, opOutTile[i].sizes,
                                resTy.getElementType());

    opResults[i] = dispatchLowerInto(curOp, inputs, outTile, rewriter, loc);
  }

  // Insert the final tile into the accumulator.
  auto inParallel =
      mlir::cast<mlir::scf::InParallelOp>(forall.getTerminator());
  rewriter.setInsertionPointToStart(inParallel.getBody());
  mlir::tensor::ParallelInsertSliceOp::create(
      rewriter, loc, opResults.back(), forall.getRegionIterArgs()[0],
      regionTile.offsets, regionTile.sizes,
      llvm::SmallVector<mlir::OpFoldResult>(rank, rewriter.getIndexAttr(1)));

  rewriter.replaceOp(op, forall.getResults());
  return mlir::success();
}

} // anonymous namespace

mlir::LogicalResult FusedRegionOpLowering::matchAndRewrite(
    brachml::FusedRegionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto outTy = mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType());
  int64_t rank = outTy.getRank();

  // Decide how many trailing dims to tile. 4D → spatial (2), 2D → matmul M/N
  // (2), 1D → 1. 0D falls back to sequential.
  int numTileDims = 0;
  if (rank >= 2) numTileDims = 2;
  else if (rank == 1) numTileDims = 1;

  if (numTileDims == 0)
    return sequentialLower(op, adaptor.getInputs(), rewriter);

  // Validate every op has an access pattern. If any doesn't, go sequential.
  mlir::Block &body = op.getBody().front();
  auto outShape = outTy.getShape();
  llvm::SmallVector<mlir::OpFoldResult> dummyOff(rank, rewriter.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult> dummySz;
  for (auto d : outShape) dummySz.push_back(rewriter.getIndexAttr(d));
  for (auto &bodyOp : body.without_terminator()) {
    auto a = getInputAccesses(&bodyOp, dummyOff, dummySz, rewriter, op.getLoc());
    if (a.empty())
      return sequentialLower(op, adaptor.getInputs(), rewriter);
  }

  return tiledLower(op, adaptor.getInputs(), rewriter, outTy, numTileDims);
}

} // namespace brachml
