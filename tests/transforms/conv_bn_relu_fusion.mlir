// RUN: %brachml-opt --brachml-fusion %s | FileCheck %s

// CHECK-LABEL: func.func @conv_bn_relu
// CHECK-NOT: brachml.conv
// CHECK-NOT: brachml.batch_norm
// CHECK-NOT: brachml.relu
// CHECK: brachml.conv_bn_relu
// CHECK: return

func.func @conv_bn_relu(
    %input: tensor<1x3x32x32xf32>,
    %conv_weight: tensor<16x3x3x3xf32>,
    %conv_bias: tensor<16xf32>,
    %bn_weight: tensor<16xf32>,
    %bn_bias: tensor<16xf32>,
    %running_mean: tensor<16xf32>,
    %running_var: tensor<16xf32>
) -> tensor<1x16x32x32xf32> {
  %0 = "brachml.conv"(%input, %conv_weight, %conv_bias) {
    stride = array<i64: 1, 1>,
    padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    transposed = false,
    output_padding = array<i64: 0, 0>,
    groups = 1 : i64
  } : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x32x32xf32>

  %1 = "brachml.batch_norm"(%0, %bn_weight, %bn_bias, %running_mean, %running_var) {
    eps = 1.0e-5 : f64,
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>
  } : (tensor<1x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x16x32x32xf32>

  %2 = "brachml.relu"(%1) : (tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>

  return %2 : tensor<1x16x32x32xf32>
}
