// RUN: %brachml-opt --brachml-fusion %s | FileCheck %s

// CHECK-LABEL: func.func @linear_fusion
// CHECK-NOT: brachml.permute
// CHECK-NOT: brachml.matmul
// CHECK-NOT: brachml.add
// CHECK: brachml.linear
// CHECK: return
func.func @linear_fusion(
    %input: tensor<1x8xf32>,
    %weight: tensor<4x8xf32>,
    %bias: tensor<4xf32>
) -> tensor<1x4xf32> {
  %0 = "brachml.permute"(%weight) {dims = array<i64: 1, 0>} : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = "brachml.matmul"(%input, %0) : (tensor<1x8xf32>, tensor<8x4xf32>) -> tensor<1x4xf32>
  %2 = "brachml.add"(%1, %bias) : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %2 : tensor<1x4xf32>
}

// Matmul on the other side of add
// CHECK-LABEL: func.func @linear_fusion_add_commuted
// CHECK-NOT: brachml.permute
// CHECK-NOT: brachml.matmul
// CHECK-NOT: brachml.add
// CHECK: brachml.linear
// CHECK: return
func.func @linear_fusion_add_commuted(
    %input: tensor<1x8xf32>,
    %weight: tensor<4x8xf32>,
    %bias: tensor<4xf32>
) -> tensor<1x4xf32> {
  %0 = "brachml.permute"(%weight) {dims = array<i64: 1, 0>} : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = "brachml.matmul"(%input, %0) : (tensor<1x8xf32>, tensor<8x4xf32>) -> tensor<1x4xf32>
  %2 = "brachml.add"(%bias, %1) : (tensor<4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %2 : tensor<1x4xf32>
}

// Should NOT fuse: permute feeds lhs of matmul (weight.T @ input, not input @ weight.T)
// CHECK-LABEL: func.func @no_fusion_permute_lhs
// CHECK: brachml.permute
// CHECK: brachml.matmul
// CHECK: brachml.add
// CHECK: return
func.func @no_fusion_permute_lhs(
    %input: tensor<4x1xf32>,
    %weight: tensor<4x8xf32>,
    %bias: tensor<1xf32>
) -> tensor<8x1xf32> {
  %0 = "brachml.permute"(%weight) {dims = array<i64: 1, 0>} : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = "brachml.matmul"(%0, %input) : (tensor<8x4xf32>, tensor<4x1xf32>) -> tensor<8x1xf32>
  %2 = "brachml.add"(%1, %bias) : (tensor<8x1xf32>, tensor<1xf32>) -> tensor<8x1xf32>
  return %2 : tensor<8x1xf32>
}

// Should NOT fuse: permute is not a transpose
// CHECK-LABEL: func.func @no_fusion_bad_permute
// CHECK: brachml.permute
// CHECK: brachml.matmul
// CHECK: brachml.add
// CHECK: return
func.func @no_fusion_bad_permute(
    %input: tensor<1x8xf32>,
    %weight: tensor<8x4xf32>,
    %bias: tensor<4xf32>
) -> tensor<1x4xf32> {
  %0 = "brachml.permute"(%weight) {dims = array<i64: 0, 1>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = "brachml.matmul"(%input, %0) : (tensor<1x8xf32>, tensor<8x4xf32>) -> tensor<1x4xf32>
  %2 = "brachml.add"(%1, %bias) : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %2 : tensor<1x4xf32>
}

// Should NOT fuse: no permute at all
// CHECK-LABEL: func.func @no_fusion_no_permute
// CHECK: brachml.matmul
// CHECK: brachml.add
// CHECK: return
func.func @no_fusion_no_permute(
    %input: tensor<1x8xf32>,
    %weight: tensor<8x4xf32>,
    %bias: tensor<4xf32>
) -> tensor<1x4xf32> {
  %0 = "brachml.matmul"(%input, %weight) : (tensor<1x8xf32>, tensor<8x4xf32>) -> tensor<1x4xf32>
  %1 = "brachml.add"(%0, %bias) : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}
