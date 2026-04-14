// QuantizationPass — attaches quantization attributes (scale / zero_point)
// to BrachML ops by joining them against a JSON calibration file produced
// on the PyTorch side (see brachml_quant.extract_calibration).
//
// Join key: each MLIR op carries a `brachml.node_name` StringAttr stamped
// by the importer. The JSON is a top-level object keyed by that exact name.
// Each entry may contain any subset of:
//   input_scale / input_zero_point
//   weight_scale / weight_zero_point
//   bias_scale / bias_zero_point
//   output_scale / output_zero_point
// We attach the present subset as `brachml.<field>` discardable attrs on
// the matching op. Ops with no entry are left untouched.

#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Transforms/Passes.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

#include <optional>

namespace brachml {

namespace {

static constexpr const char *kBrachMLNodeNameAttr = "brachml.node_name";

std::optional<llvm::json::Object> loadCalibration(llvm::StringRef path,
                                                  mlir::Operation *anchor) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    anchor->emitError("brachml-quantize: cannot open ")
        << path << " : " << bufOrErr.getError().message();
    return std::nullopt;
  }
  auto buf = std::move(*bufOrErr);

  auto expected = llvm::json::parse(buf->getBuffer());
  if (!expected) {
    anchor->emitError("brachml-quantize: failed to parse JSON: ")
        << llvm::toString(expected.takeError());
    return std::nullopt;
  }
  const auto *root = expected->getAsObject();
  if (!root) {
    anchor->emitError(
        "brachml-quantize: calibration JSON root must be an object");
    return std::nullopt;
  }

  return *root;
}

struct QuantizationPass
    : public mlir::PassWrapper<QuantizationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizationPass)

  QuantizationPass() = default;
  QuantizationPass(const QuantizationPass &other)
      : mlir::PassWrapper<QuantizationPass,
                          mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  llvm::StringRef getArgument() const override { return "brachml-quantize"; }
  llvm::StringRef getDescription() const override {
    return "Attach quantization attributes to BrachML ops from a JSON "
           "calibration file";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<BrachMLDialect>();
  }

  Option<std::string> calibrationFile{
      *this, "calibration-file",
      ::llvm::cl::desc(
          "Path to calibration JSON from brachml_quant.extract_calibration"),
      ::llvm::cl::init("")};

  void runOnOperation() override {
    auto func = getOperation();

    if (!calibrationFile.hasValue()) {
      func->emitWarning(
          "brachml-quantize: no calibration data passed, skipping...");
      return;
    }

    auto cal = loadCalibration(calibrationFile.getValue(), func);
    if (!cal) {
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(&getContext());
    auto i32Ty = builder.getI32Type();
    auto f32Ty = builder.getF32Type();

    static constexpr struct {
      const char *jsonKey;
      const char *attrName;
      bool isInt;
    } kFields[] = {
        {"input_scale", "brachml.input_scale", false},
        {"input_zero_point", "brachml.input_zero_point", true},
        {"weight_scale", "brachml.weight_scale", false},
        {"weight_zero_point", "brachml.weight_zero_point", true},
        {"bias_scale", "brachml.bias_scale", false},
        {"bias_zero_point", "brachml.bias_zero_point", true},
        {"output_scale", "brachml.output_scale", false},
        {"output_zero_point", "brachml.output_zero_point", true},
    };

    func.walk([&](mlir::Operation *op) {
      auto nameAttr = op->getAttrOfType<mlir::StringAttr>(kBrachMLNodeNameAttr);
      if (!nameAttr) return;

      auto obj = cal->getObject(nameAttr.getValue());
      if (!obj) return;

      for (const auto &f : kFields) {
        if (f.isInt) {
          if (auto v = obj->getInteger(f.jsonKey))
            op->setAttr(f.attrName, mlir::IntegerAttr::get(i32Ty, *v));
        } else {
          if (auto v = obj->getNumber(f.jsonKey))
            op->setAttr(f.attrName, mlir::FloatAttr::get(f32Ty, *v));
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createQuantizationPass() {
  return std::make_unique<QuantizationPass>();
}

} // namespace brachml
