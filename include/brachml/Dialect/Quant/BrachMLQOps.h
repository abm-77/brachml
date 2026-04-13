#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <brachml/Dialect/Quant/BrachMLQDialect.h>

#define GET_OP_CLASSES
#include <brachml/Dialect/Quant/BrachMLQOps.h.inc>
#undef GET_OP_CLASSES
