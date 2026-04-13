#include <brachml/Dialect/Basic/BrachMLOps.h>

#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace brachml;

LogicalResult AddOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  // Element types must match.
  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOpError("lhs element type (")
           << lhsType.getElementType() << ") must match rhs element type ("
           << rhsType.getElementType() << ")";

  if (lhsType.getElementType() != resultType.getElementType())
    return emitOpError("result element type (")
           << resultType.getElementType()
           << ") must match operand element type ("
           << lhsType.getElementType() << ")";

  // Check broadcast compatibility and result shape.
  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();
  int64_t resultRank = resultType.getRank();
  int64_t expectedRank = std::max(lhsRank, rhsRank);

  if (resultRank != expectedRank)
    return emitOpError("result rank (")
           << resultRank << ") must equal " << expectedRank;

  for (int64_t i = 0; i < expectedRank; ++i) {
    int64_t lhsIdx = lhsRank - 1 - i;
    int64_t rhsIdx = rhsRank - 1 - i;

    int64_t lhsDim = lhsIdx >= 0 ? lhsType.getDimSize(lhsIdx) : 1;
    int64_t rhsDim = rhsIdx >= 0 ? rhsType.getDimSize(rhsIdx) : 1;

    if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim))
      continue;

    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return emitOpError("operands are not broadcast compatible at dimension ")
             << (expectedRank - 1 - i) << ": " << lhsDim << " vs " << rhsDim;

    int64_t expectedDim = std::max(lhsDim, rhsDim);
    int64_t resultDim = resultType.getDimSize(expectedRank - 1 - i);
    if (!ShapedType::isDynamic(resultDim) && resultDim != expectedDim)
      return emitOpError("result dimension ")
             << (expectedRank - 1 - i) << " (" << resultDim
             << ") must equal broadcast dimension (" << expectedDim << ")";
  }

  return success();
}

LogicalResult ConvOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto weightType = llvm::cast<RankedTensorType>(getWeight().getType());
  int64_t inputRank = inputType.getRank();
  int64_t weightRank = weightType.getRank();
  int64_t spatialDims = inputRank - 2;

  if (spatialDims < 1)
    return emitOpError("input must be at least rank 3 (batch + channel + "
                       "spatial), got rank ")
           << inputRank;

  if (weightRank != inputRank)
    return emitOpError("weight rank (")
           << weightRank << ") must equal input rank (" << inputRank << ")";

  auto stride = getStride();
  auto padding = getPadding();
  auto dilation = getDilation();
  auto outputPadding = getOutputPadding();

  if (static_cast<int64_t>(stride.size()) != spatialDims)
    return emitOpError("stride length (")
           << stride.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  if (static_cast<int64_t>(padding.size()) != spatialDims)
    return emitOpError("padding length (")
           << padding.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  if (static_cast<int64_t>(dilation.size()) != spatialDims)
    return emitOpError("dilation length (")
           << dilation.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  if (static_cast<int64_t>(outputPadding.size()) != spatialDims)
    return emitOpError("output_padding length (")
           << outputPadding.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  int64_t groups = getGroups();
  bool transposed = getTransposed();

  // input channel dim is always dim 1
  int64_t inputChannels = inputType.getDimSize(1);
  // weight: non-transposed [outC, inC/groups, ...], transposed [inC,
  // outC/groups, ...]
  int64_t weightInC =
      transposed ? weightType.getDimSize(0) : weightType.getDimSize(1);
  int64_t weightOutC =
      transposed ? weightType.getDimSize(1) : weightType.getDimSize(0);

  if (!ShapedType::isDynamic(inputChannels) &&
      !ShapedType::isDynamic(weightInC) && groups > 0) {
    int64_t expectedWeightInC = inputChannels / groups;
    if (transposed) {
      // For transposed conv, input channels must equal weight dim 0
      if (inputChannels != weightInC)
        return emitOpError("input channels (")
               << inputChannels << ") must equal weight dim 0 (" << weightInC
               << ") for transposed convolution";
    } else {
      if (weightInC != expectedWeightInC)
        return emitOpError("weight input channels (")
               << weightInC << ") must equal input channels / groups ("
               << expectedWeightInC << ")";
    }
  }

  if (getBias()) {
    auto biasType = llvm::cast<RankedTensorType>(getBias().getType());
    if (biasType.getRank() != 1)
      return emitOpError("bias must be 1D, got rank ") << biasType.getRank();

    int64_t outChannels =
        transposed ? weightType.getDimSize(1) * groups : weightOutC;
    int64_t biasLen = biasType.getDimSize(0);
    if (!ShapedType::isDynamic(biasLen) &&
        !ShapedType::isDynamic(outChannels) && biasLen != outChannels)
      return emitOpError("bias length (")
             << biasLen << ") must equal output channels (" << outChannels
             << ")";
  }

  if (transposed) {
    for (int64_t i = 0; i < spatialDims; ++i) {
      if (outputPadding[i] >= stride[i])
        return emitOpError("output_padding[")
               << i << "] (" << outputPadding[i]
               << ") must be less than stride[" << i << "] (" << stride[i]
               << ") for transposed convolution";
    }
  }

  return success();
}

LogicalResult BatchNormOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t inputRank = inputType.getRank();

  if (inputRank < 2)
    return emitOpError("input must be at least rank 2, got rank ") << inputRank;

  int64_t channels = inputType.getDimSize(1);

  auto meanType = llvm::cast<RankedTensorType>(getRunningMean().getType());
  auto varType = llvm::cast<RankedTensorType>(getRunningVar().getType());

  if (meanType.getRank() != 1)
    return emitOpError("running_mean must be 1D, got rank ")
           << meanType.getRank();

  if (varType.getRank() != 1)
    return emitOpError("running_var must be 1D, got rank ")
           << varType.getRank();

  int64_t meanLen = meanType.getDimSize(0);
  int64_t varLen = varType.getDimSize(0);

  if (!ShapedType::isDynamic(channels) && !ShapedType::isDynamic(meanLen) &&
      meanLen != channels)
    return emitOpError("running_mean length (")
           << meanLen << ") must equal input channel dimension (" << channels
           << ")";

  if (!ShapedType::isDynamic(channels) && !ShapedType::isDynamic(varLen) &&
      varLen != channels)
    return emitOpError("running_var length (")
           << varLen << ") must equal input channel dimension (" << channels
           << ")";

  if (getWeight()) {
    auto weightType = llvm::cast<RankedTensorType>(getWeight().getType());
    if (weightType.getRank() != 1)
      return emitOpError("weight must be 1D, got rank ")
             << weightType.getRank();
    int64_t weightLen = weightType.getDimSize(0);
    if (!ShapedType::isDynamic(weightLen) && !ShapedType::isDynamic(meanLen) &&
        weightLen != meanLen)
      return emitOpError("weight length (")
             << weightLen << ") must equal running_mean length (" << meanLen
             << ")";
  }

  if (getBias()) {
    auto biasType = llvm::cast<RankedTensorType>(getBias().getType());
    if (biasType.getRank() != 1)
      return emitOpError("bias must be 1D, got rank ") << biasType.getRank();
    int64_t biasLen = biasType.getDimSize(0);
    if (!ShapedType::isDynamic(biasLen) && !ShapedType::isDynamic(meanLen) &&
        biasLen != meanLen)
      return emitOpError("bias length (")
             << biasLen << ") must equal running_mean length (" << meanLen
             << ")";
  }

  double eps = getEps().convertToDouble();
  if (eps <= 0.0)
    return emitOpError("eps must be greater than zero, got ") << eps;

  return success();
}

LogicalResult MaxPool::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t inputRank = inputType.getRank();

  if (inputRank < 3)
    return emitOpError("input must be at least rank 3 (batch + channel + "
                       "spatial), got rank ")
           << inputRank;

  int64_t spatialDims = inputRank - 2;
  auto kernelSize = getKernelSize();
  auto stride = getStride();
  auto padding = getPadding();
  auto dilation = getDilation();

  if (static_cast<int64_t>(kernelSize.size()) != spatialDims)
    return emitOpError("kernel_size length (")
           << kernelSize.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  if (static_cast<int64_t>(stride.size()) != spatialDims)
    return emitOpError("stride length (")
           << stride.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  if (static_cast<int64_t>(padding.size()) != spatialDims)
    return emitOpError("padding length (")
           << padding.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  if (static_cast<int64_t>(dilation.size()) != spatialDims)
    return emitOpError("dilation length (")
           << dilation.size() << ") must equal spatial dimensions ("
           << spatialDims << ")";

  for (int64_t i = 0; i < spatialDims; ++i) {
    if (padding[i] > kernelSize[i] / 2)
      return emitOpError("padding[")
             << i << "] (" << padding[i] << ") must be <= half of kernel_size["
             << i << "] (" << kernelSize[i] / 2 << ")";
  }

  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());
  if (resultType.getRank() != inputRank)
    return emitOpError("result rank (")
           << resultType.getRank() << ") must equal input rank (" << inputRank
           << ")";

  return success();
}

LogicalResult MatMulOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());
  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();

  if (lhsRank < 2)
    return emitOpError("lhs must be at least rank 2, got rank ") << lhsRank;

  if (rhsRank < 2)
    return emitOpError("rhs must be at least rank 2, got rank ") << rhsRank;

  // Contracting dimensions: lhs last dim must match rhs second-to-last dim.
  int64_t lhsK = lhsType.getDimSize(lhsRank - 1);
  int64_t rhsK = rhsType.getDimSize(rhsRank - 2);
  if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
      lhsK != rhsK)
    return emitOpError("lhs last dimension (")
           << lhsK << ") must equal rhs second-to-last dimension (" << rhsK
           << ")";

  // Result rank should match the broadcast of batch dims + matrix dims.
  int64_t expectedRank = std::max(lhsRank, rhsRank);
  if (resultType.getRank() != expectedRank)
    return emitOpError("result rank (")
           << resultType.getRank() << ") must equal " << expectedRank;

  // Check batch dimensions match (broadcast rules).
  int64_t lhsBatchRank = lhsRank - 2;
  int64_t rhsBatchRank = rhsRank - 2;
  int64_t maxBatchRank = std::max(lhsBatchRank, rhsBatchRank);

  for (int64_t i = 0; i < maxBatchRank; ++i) {
    int64_t lhsIdx = lhsBatchRank - 1 - i;
    int64_t rhsIdx = rhsBatchRank - 1 - i;

    int64_t lhsDim = lhsIdx >= 0 ? lhsType.getDimSize(lhsIdx) : 1;
    int64_t rhsDim = rhsIdx >= 0 ? rhsType.getDimSize(rhsIdx) : 1;

    if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim))
      continue;

    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return emitOpError("batch dimension mismatch at position ")
             << i << ": lhs has " << lhsDim << ", rhs has " << rhsDim;
  }

  return success();
}

LogicalResult ReshapeOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());
  auto size = getSize();

  // Check at most one -1 in size.
  int negOneCount = 0;
  for (int64_t s : size) {
    if (s == -1)
      ++negOneCount;
    else if (s <= 0)
      return emitOpError("size elements must be positive or -1, got ") << s;
  }

  if (negOneCount > 1)
    return emitOpError("at most one -1 is allowed in size, got ")
           << negOneCount;

  // Result shape must match size.
  if (resultType.getRank() != static_cast<int64_t>(size.size()))
    return emitOpError("result rank (")
           << resultType.getRank() << ") must equal size length ("
           << size.size() << ")";

  for (unsigned i = 0; i < size.size(); ++i) {
    int64_t resultDim = resultType.getDimSize(i);
    if (ShapedType::isDynamic(resultDim) || size[i] == -1)
      continue;
    if (resultDim != size[i])
      return emitOpError("result dimension ")
             << i << " (" << resultDim << ") must match size[" << i << "] ("
             << size[i] << ")";
  }

  // If input shape is fully static, verify element count.
  if (inputType.hasStaticShape()) {
    int64_t inputElements = inputType.getNumElements();
    int64_t sizeProduct = 1;
    bool hasInferred = false;

    for (int64_t s : size) {
      if (s == -1) {
        hasInferred = true;
      } else {
        sizeProduct *= s;
      }
    }

    if (!hasInferred && sizeProduct != inputElements)
      return emitOpError("total elements in size (")
             << sizeProduct << ") must equal input elements (" << inputElements
             << ")";

    if (hasInferred && sizeProduct > 0 && inputElements % sizeProduct != 0)
      return emitOpError("input elements (")
             << inputElements
             << ") must be divisible by the product of known size dimensions ("
             << sizeProduct << ")";
  }

  return success();
}

LogicalResult PermuteOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());
  auto dims = getDims();
  int64_t rank = inputType.getRank();

  if (static_cast<int64_t>(dims.size()) != rank)
    return emitOpError("dims length (")
           << dims.size() << ") must equal input rank (" << rank << ")";

  if (resultType.getRank() != rank)
    return emitOpError("result rank (")
           << resultType.getRank() << ") must equal input rank (" << rank
           << ")";

  // Check valid permutation: each value in [0, rank), no duplicates.
  llvm::DenseSet<int64_t> seen;
  for (unsigned i = 0; i < dims.size(); ++i) {
    int64_t d = dims[i];
    if (d < 0 || d >= rank)
      return emitOpError("dims[")
             << i << "] (" << d << ") must be in range [0, " << rank << ")";
    if (!seen.insert(d).second)
      return emitOpError("dims contains duplicate value ") << d;
  }

  // Verify result shape matches permuted input shape.
  for (int64_t i = 0; i < rank; ++i) {
    int64_t inputDim = inputType.getDimSize(dims[i]);
    int64_t resultDim = resultType.getDimSize(i);
    if (!ShapedType::isDynamic(inputDim) && !ShapedType::isDynamic(resultDim) &&
        inputDim != resultDim)
      return emitOpError("result dimension ")
             << i << " (" << resultDim << ") must match input dimension "
             << dims[i] << " (" << inputDim << ")";
  }

  return success();
}

LogicalResult LinearOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto weightType = llvm::cast<RankedTensorType>(getWeight().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());
  int64_t inputRank = inputType.getRank();

  if (inputRank < 2)
    return emitOpError("input must be at least rank 2, got rank ") << inputRank;

  if (weightType.getRank() != 2)
    return emitOpError("weight must be rank 2, got rank ")
           << weightType.getRank();

  // input's last dim must equal weight's last dim (weight gets transposed)
  int64_t inputLastDim = inputType.getDimSize(inputRank - 1);
  int64_t weightLastDim = weightType.getDimSize(1);
  if (!ShapedType::isDynamic(inputLastDim) &&
      !ShapedType::isDynamic(weightLastDim) && inputLastDim != weightLastDim)
    return emitOpError("input's last dimension (")
           << inputLastDim << ") must equal weight's last dimension ("
           << weightLastDim << ")";

  int64_t outFeatures = weightType.getDimSize(0);

  if (getBias()) {
    auto biasType = llvm::cast<RankedTensorType>(getBias().getType());
    if (biasType.getRank() != 1)
      return emitOpError("bias must be rank 1, got rank ")
             << biasType.getRank();
    int64_t biasLen = biasType.getDimSize(0);
    if (!ShapedType::isDynamic(biasLen) &&
        !ShapedType::isDynamic(outFeatures) && biasLen != outFeatures)
      return emitOpError("bias length (")
             << biasLen << ") must equal weight's first dimension ("
             << outFeatures << ")";
  }

  if (resultType.getRank() != inputRank)
    return emitOpError("result rank (")
           << resultType.getRank() << ") must equal input rank (" << inputRank
           << ")";

  // Result's last dim must equal weight's first dim (output features).
  int64_t resultLastDim = resultType.getDimSize(resultType.getRank() - 1);
  if (!ShapedType::isDynamic(resultLastDim) &&
      !ShapedType::isDynamic(outFeatures) && resultLastDim != outFeatures)
    return emitOpError("result's last dimension (")
           << resultLastDim << ") must equal weight's first dimension ("
           << outFeatures << ")";

  // Batch dimensions must match.
  for (int64_t i = 0; i < inputRank - 1; ++i) {
    int64_t inputDim = inputType.getDimSize(i);
    int64_t resultDim = resultType.getDimSize(i);
    if (!ShapedType::isDynamic(inputDim) &&
        !ShapedType::isDynamic(resultDim) && inputDim != resultDim)
      return emitOpError("result batch dimension ")
             << i << " (" << resultDim << ") must match input dimension ("
             << inputDim << ")";
  }

  return success();
}

#define GET_OP_CLASSES
#include <brachml/Dialect/Basic/BrachMLOps.cpp.inc>
#undef GET_OP_CLASSES
