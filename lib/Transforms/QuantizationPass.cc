#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Quant/BrachMLQDialect.h>
#include <brachml/Transforms/Passes.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

namespace brachml {

struct QuantizationPass
    : public mlir::PassWrapper<QuantizationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizationPass)

  llvm::StringRef getArgument() const override { return "brachml-quantize"; }
  llvm::StringRef getDescription() const override {
    return "Quantize BrachML ops to fixed-point representations";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<BrachMLDialect>();
    registry.insert<q::BrachMLQDialect>();
  }

  void runOnOperation() override {
    // TODO: implement quantization
  }
};

std::unique_ptr<mlir::Pass> createQuantizationPass() {
  return std::make_unique<QuantizationPass>();
}

} // namespace brachml
