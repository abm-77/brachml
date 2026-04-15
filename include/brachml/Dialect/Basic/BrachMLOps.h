#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <brachml/Dialect/Basic/BrachMLDialect.h>

namespace brachml {

// Trait for ops that may participate in a fused_region body.
template <typename ConcreteOp>
struct FusableOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteOp, FusableOpTrait> {};

} // namespace brachml

#define GET_OP_CLASSES
#include <brachml/Dialect/Basic/BrachMLOps.h.inc>
#undef GET_OP_CLASSES
