include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "BrachMLQDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class BrachMLQ_Op<string mnemonic, list<Trait> traits = []>
  : Op<BrachMLQ_Dialect, mnemonic, traits>;
