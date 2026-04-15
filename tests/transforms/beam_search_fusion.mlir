// RUN: %brachml-opt --brachml-beam-fusion %s | FileCheck %s

// ── Basic 2-op chain ─────────────────────────────────────────────────────────
// relu -> add: relu result has one use, both ops have FusableOpTrait.
// Expect them wrapped in a single fused_region.

// CHECK-LABEL: func.func @two_op_chain
// CHECK:       brachml.fused_region
// CHECK:         brachml.relu
// CHECK:         brachml.add
// CHECK:         brachml.yield
// CHECK:         return

func.func @two_op_chain(%input: tensor<4xf32>, %other: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "brachml.relu"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "brachml.add"(%0, %other) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// ── 3-op chain (default maxDepth = 3) ────────────────────────────────────────
// relu -> add -> relu: all three should land in one fused_region.

// CHECK-LABEL: func.func @three_op_chain
// CHECK:       brachml.fused_region
// CHECK:         brachml.relu
// CHECK:         brachml.add
// CHECK:         brachml.relu
// CHECK:         brachml.yield

func.func @three_op_chain(%input: tensor<4xf32>, %other: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "brachml.relu"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "brachml.add"(%0, %other) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "brachml.relu"(%1) : (tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// ── Single op — no fusion ─────────────────────────────────────────────────────
// A chain needs at least 2 ops. Single fusable op should pass through unchanged.

// CHECK-LABEL: func.func @single_op
// CHECK-NOT:   brachml.fused_region
// CHECK:       brachml.relu

func.func @single_op(%input: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "brachml.relu"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// ── Multi-use result — no fusion ──────────────────────────────────────────────
// relu result consumed by two ops: hasOneUse() is false, so no chain is formed.

// CHECK-LABEL: func.func @multi_use
// CHECK-NOT:   brachml.fused_region
// CHECK:       brachml.relu
// CHECK:       brachml.add

func.func @multi_use(%input: tensor<4xf32>, %other: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "brachml.relu"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "brachml.add"(%0, %other) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// ── No nested fused_regions ───────────────────────────────────────────────────
// An op chain that includes an already-fused_region should not wrap it again.
// FusedRegionOp does not carry FusableOpTrait so it is never added to a chain.

// CHECK-LABEL: func.func @no_nesting
// CHECK:       brachml.fused_region
// CHECK-NOT:   brachml.fused_region

func.func @no_nesting(%input: tensor<4xf32>, %other: tensor<4xf32>) -> tensor<4xf32> {
  %0 = brachml.fused_region(%input : tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%arg0: tensor<4xf32>):
    %r = "brachml.relu"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    brachml.yield %r : tensor<4xf32>
  }
  %1 = "brachml.add"(%0, %other) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// ── Non-fusable op does not participate ───────────────────────────────────────
// conv_bn_relu does not carry FusableOpTrait — it should not be wrapped.
// The relu after it has no fusable successor so it also stays unfused.

// CHECK-LABEL: func.func @non_fusable
// CHECK-NOT:   brachml.fused_region
// CHECK:       brachml.conv_bn_relu

func.func @non_fusable(
    %input: tensor<1x3x32x32xf32>,
    %w: tensor<16x3x3x3xf32>,
    %mean: tensor<16xf32>,
    %var: tensor<16xf32>
) -> tensor<1x16x32x32xf32> {
  %0 = "brachml.conv_bn_relu"(%input, %w, %mean, %var) {
    stride = array<i64: 1, 1>,
    padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    transposed = false,
    output_padding = array<i64: 0, 0>,
    groups = 1 : i64,
    eps = 1.0e-5 : f64,
    operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1, 1>
  } : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x16x32x32xf32>
  return %0 : tensor<1x16x32x32xf32>
}
