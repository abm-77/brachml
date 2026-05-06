#include <brachml/Conversion/Passes.h>
#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>
#include <brachml/Transforms/Passes.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/LinalgToStandard/LinalgToStandard.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h>
#include <mlir/Conversion/VectorToSCF/VectorToSCF.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MLProgram/IR/MLProgram.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

namespace {

// Pipeline that runs our frontend optimizations and lowers BrachML → linalg.
void buildOptimizePipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(brachml::createBeamSearchFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(brachml::createConvertBrachMLToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

// Pipeline that takes our linalg/tensor/scf IR all the way to the LLVM dialect.
//
// Order of operations:
//   1. Bufferize (tensor → memref). Required before any further lowering.
//   2. Linalg → loops. Linalg on memrefs becomes scf loops.
//   3. scf.forall → scf.for. Our tiling uses forall; single-core bare metal
//      wants sequential for loops.
//   4. vector.* → scf / llvm. vectorize what we can, lower the rest.
//   5. memref + arith + func + cf + index → LLVM dialect.
//   6. reconcile-unrealized-casts cleans up type materialization casts left
//      by the partial conversions above.
//
// TODO: insert brachml-vectorize before bufferization once implemented.
void buildLowerToLLVMPipeline(mlir::OpPassManager &pm) {
  // Bufferize: tensor semantics → memref semantics. This has to happen before
  // any lowering that expects memref inputs.
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Linalg on memrefs → scf loops.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // scf.forall → scf.for (single-core bare metal).
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createForallToForLoopPass());

  // vector.transfer_* → scf.for + vector loads/stores.
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createSCFToControlFlowPass());

  // Expand strided memref metadata (subview offset/size/stride arithmetic).
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());

  // Final conversions to the LLVM dialect.
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());

  // Cleans up unrealized_conversion_cast ops left by partial conversions.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

// Full pipeline: optimize then lower all the way to LLVM dialect.
void buildFullPipeline(mlir::OpPassManager &pm) {
  buildOptimizePipeline(pm);
  buildLowerToLLVMPipeline(pm);
}

} // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<brachml::BrachMLDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::index::IndexDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Bufferization external-model interface registrations. These let OneShot
  // bufferization handle ops from dialects it doesn't own.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::ml_program::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register every MLIR pass so they're reachable via --pass-pipeline or by
  // name. Our pipelines below pull a specific subset into a named recipe.
  mlir::registerAllPasses();

  // BrachML passes
  brachml::transforms::registerPasses();
  brachml::conversion::registerPasses();

  mlir::PassPipelineRegistration<>(
      "brachml-optimize",
      "BEAM fusion + BrachML → linalg (frontend optimizations only)",
      buildOptimizePipeline);

  mlir::PassPipelineRegistration<>(
      "brachml-to-llvm",
      "Lower BrachML IR all the way to the LLVM dialect",
      buildFullPipeline);

  mlir::PassPipelineRegistration<>(
      "lower-to-llvm",
      "Lower already-converted linalg/tensor/scf IR to the LLVM dialect",
      buildLowerToLLVMPipeline);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BrachML optimizer\n", registry));
}
