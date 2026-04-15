# RUN: %brachml_import %S/models/cifar_quant.pt2 | FileCheck %s

# Quantized cifar10 model — verify quant params are stamped directly on ops
# by the importer (no separate calibration file or pass needed).

# First conv gets input + weight scales from the surrounding dequantize nodes.
# CHECK: "brachml.conv"
# CHECK-SAME: brachml.input_scale
# CHECK-SAME: brachml.input_zero_point
# CHECK-SAME: brachml.weight_scale
# CHECK-SAME: brachml.weight_zero_point

# relu gets an output scale from the quantize node that consumes it.
# CHECK: "brachml.relu"
# CHECK-SAME: brachml.output_scale
# CHECK-SAME: brachml.output_zero_point
