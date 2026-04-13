include "mlir/IR/DialectBase.td"

def BrachMLQ_Dialect : Dialect {
  let name = "brachml.q";
  let summary = "BrachML quantization dialect";
  let cppNamespace = "::brachml::q";
}
