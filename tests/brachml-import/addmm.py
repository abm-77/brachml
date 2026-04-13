# RUN: %brachml-import %S/models/addmm.pt2 | FileCheck %s

# addmm decomposes to permute + matmul + add in core ATen IR
# CHECK: func.func @model
# CHECK:   "brachml.permute"
# CHECK:   "brachml.matmul"
# CHECK:   "brachml.add"
# CHECK:   return
