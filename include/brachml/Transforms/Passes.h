#pragma once

#include <mlir/Pass/Pass.h>

namespace brachml {

std::unique_ptr<mlir::Pass> createBeamSearchFusionPass();
std::unique_ptr<mlir::Pass> createVectorizePass();

namespace transforms {
#define GEN_PASS_REGISTRATION
#include <brachml/Transforms/Passes.h.inc>
} // namespace transforms

} // namespace brachml
