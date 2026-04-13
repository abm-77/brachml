# RUN: %brachml-import %S/models/reshape.pt2 | FileCheck %s

# CHECK: func.func @model(%arg0: tensor<2x3x4xf32>) -> tensor<2x12xf32>
# CHECK:   %0 = "brachml.reshape"(%arg0)
# CHECK:   return %0
