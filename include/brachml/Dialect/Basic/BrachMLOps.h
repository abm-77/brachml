#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <brachml/Dialect/Basic/BrachMLDialect.h>

#define GET_OP_CLASSES
#include <brachml/Dialect/Basic/BrachMLOps.h.inc>
#undef GET_OP_CLASSES
