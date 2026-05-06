#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>

namespace brachml {

// How a single input of an op is accessed given the op's output tile.
struct InputAccess {
  // If true, the input is not sliced — pass the full tensor.
  bool full = false;
  // Otherwise these describe a tensor.extract_slice of the input.
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
};

// Compute per-input access for every operand of `op`, given the op's output
// tile (offsets and sizes, one entry per result dim).
//
// Returns an empty vector if this op has no access pattern (meaning it cannot
// participate in tiled fusion and the caller should fall back to sequential
// lowering). Currently this includes reshape and permute.
llvm::SmallVector<InputAccess>
getInputAccesses(mlir::Operation *op,
                 llvm::ArrayRef<mlir::OpFoldResult> outOffsets,
                 llvm::ArrayRef<mlir::OpFoldResult> outSizes,
                 mlir::OpBuilder &b, mlir::Location loc);

} // namespace brachml
