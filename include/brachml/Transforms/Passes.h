#pragma once

#include <mlir/Pass/Pass.h>

namespace brachml {

std::unique_ptr<mlir::Pass> createFusionPass();
std::unique_ptr<mlir::Pass> createQuantizationPass();

namespace transforms {
#define GEN_PASS_REGISTRATION
#include <brachml/Transforms/Passes.h.inc>
} // namespace transforms

} // namespace brachml
