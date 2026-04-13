#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>

#include <brachml/Dialect/Basic/BrachMLDialect.cpp.inc>

void brachml::BrachMLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <brachml/Dialect/Basic/BrachMLOps.cpp.inc>
#undef GET_OP_LIST
      >();
}
