# RUN: %brachml-import %S/models/relu.pt2 | FileCheck %s

# CHECK: func.func @model(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32>
# CHECK:   %0 = "brachml.relu"(%arg0) : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32>
# CHECK:   return %0
