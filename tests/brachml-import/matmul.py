# RUN: %brachml-import %S/models/matmul.pt2 | FileCheck %s

# CHECK: func.func @model(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32>
# CHECK:   %0 = "brachml.matmul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
# CHECK:   return %0
