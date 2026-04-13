#pragma once

#include <mlir/Pass/Pass.h>

namespace brachml {

std::unique_ptr<mlir::Pass> createConvertBrachMLToLLVMPass();

namespace conversion {
#define GEN_PASS_REGISTRATION
#include <brachml/Conversion/Passes.h.inc>
} // namespace conversion

} // namespace brachml
