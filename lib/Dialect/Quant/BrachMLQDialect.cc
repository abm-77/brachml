#include <brachml/Dialect/Quant/BrachMLQDialect.h>
#include <brachml/Dialect/Quant/BrachMLQOps.h>

#include <brachml/Dialect/Quant/BrachMLQDialect.cpp.inc>

void brachml::q::BrachMLQDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <brachml/Dialect/Quant/BrachMLQOps.cpp.inc>
#undef GET_OP_LIST
      >();
}
