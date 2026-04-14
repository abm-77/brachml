# RUN: %brachml_import %S/models/add.pt2 | FileCheck %s

# CHECK: func.func @model(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>
# CHECK:   %0 = "brachml.add"(%arg0, %arg1) {{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
# CHECK:   return %0
