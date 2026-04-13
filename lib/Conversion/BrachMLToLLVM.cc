#include <brachml/Conversion/Passes.h>
#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>

namespace brachml {

struct ConvertBrachMLToLLVMPass
    : public mlir::PassWrapper<ConvertBrachMLToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertBrachMLToLLVMPass)

  llvm::StringRef getArgument() const override {
    return "convert-brachml-to-llvm";
  }
  llvm::StringRef getDescription() const override {
    return "Lower BrachML dialect to LLVM dialect";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<BrachMLDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // TODO: implement BrachML -> LLVM lowering
  }
};

std::unique_ptr<mlir::Pass> createConvertBrachMLToLLVMPass() {
  return std::make_unique<ConvertBrachMLToLLVMPass>();
}

} // namespace brachml
