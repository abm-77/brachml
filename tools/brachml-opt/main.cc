
#include <brachml/Conversion/Passes.h>
#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>
#include <brachml/Dialect/Quant/BrachMLQDialect.h>
#include <brachml/Transforms/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MLProgram/IR/MLProgram.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<brachml::BrachMLDialect>();
  registry.insert<brachml::q::BrachMLQDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();

  // MLIR built-in passes
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();

  // BrachML passes
  brachml::transforms::registerPasses();
  brachml::conversion::registerPasses();

  mlir::PassPipelineRegistration<>(
      "brachml-optimize", "Run all BrachML optimizations",
      [](mlir::OpPassManager &pm) {
        pm.addNestedPass<mlir::func::FuncOp>(brachml::createFusionPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSymbolDCEPass());
      });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BrachML optimizer\n", registry));
}
