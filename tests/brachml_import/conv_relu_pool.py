# RUN: %brachml_import %S/models/conv_relu_pool.pt2 | FileCheck %s

# CHECK: func.func @model
# CHECK:   "brachml.conv"
# CHECK:   "brachml.relu"
# CHECK:   "brachml.max_pool"
# CHECK:   return
