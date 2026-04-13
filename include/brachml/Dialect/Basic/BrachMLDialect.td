

include "mlir/IR/DialectBase.td"

def BrachML_Dialect : Dialect {
  let name = "brachml";
  let summary = "high-level ml ops";
  let cppNamespace = "::brachml";
}



