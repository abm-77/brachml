#pragma once

#include <brachml/Dialect/Basic/BrachMLOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace brachml {

struct FusedRegionOpLowering
    : public mlir::OpConversionPattern<brachml::FusedRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(brachml::FusedRegionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace brachml
