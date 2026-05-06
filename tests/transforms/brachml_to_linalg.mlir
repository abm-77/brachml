// RUN: %brachml-opt --convert-brachml-to-linalg %s | FileCheck %s

// ── add ───────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_add
// CHECK:       linalg.add ins(%{{.*}}, %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:             outs(%{{.*}} : tensor<4x4xf32>)
// CHECK-NOT:   brachml.add
func.func @test_add(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = brachml.add %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ── relu (float) ──────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_relu_f32
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}} : tensor<1x32xf32>) outs(%{{.*}} : tensor<1x32xf32>)
// CHECK:       ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK:         arith.constant 0.000000e+00 : f32
// CHECK:         arith.maximumf
// CHECK:         linalg.yield
// CHECK-NOT:   brachml.relu
func.func @test_relu_f32(%a: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %0 = brachml.relu %a : tensor<1x32xf32>
  return %0 : tensor<1x32xf32>
}

// ── relu (int8) ───────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_relu_i8
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}} : tensor<1x32xi8>) outs(%{{.*}} : tensor<1x32xi8>)
// CHECK:       ^bb0(%{{.*}}: i8, %{{.*}}: i8):
// CHECK:         arith.constant 0 : i8
// CHECK:         arith.maxsi
// CHECK:         linalg.yield
// CHECK-NOT:   brachml.relu
func.func @test_relu_i8(%a: tensor<1x32xi8>) -> tensor<1x32xi8> {
  %0 = brachml.relu %a : tensor<1x32xi8>
  return %0 : tensor<1x32xi8>
}

// ── matmul ────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_matmul
// CHECK:       linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x4xf32>)
// CHECK:       linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK-SAME:               outs(%{{.*}} : tensor<4x4xf32>)
// CHECK-NOT:   brachml.matmul
func.func @test_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %0 = brachml.matmul %a, %b : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ── conv (no bias, no pad) ────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_conv_no_bias
// CHECK:       linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<1x16x6x6xf32>)
// CHECK:       linalg.conv_2d_nchw_fchw
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<1x16x6x6xf32>)
// CHECK-NOT:   brachml.conv
func.func @test_conv_no_bias(
    %input:  tensor<1x3x8x8xf32>,
    %weight: tensor<16x3x3x3xf32>
) -> tensor<1x16x6x6xf32> {
  %0 = brachml.conv %input, %weight
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    transposed = false
    output_padding = [0, 0]
    groups = 1
    : tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x6x6xf32>
  return %0 : tensor<1x16x6x6xf32>
}

// ── conv (with bias) ──────────────────────────────────────────────────────────
// Bias [16] is broadcast over [1,16,6,6] via a linalg.generic with a channel map.

// CHECK-LABEL: func.func @test_conv_bias
// CHECK:       linalg.conv_2d_nchw_fchw
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<1x16x6x6xf32>)
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<1x16x6x6xf32>, tensor<16xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<1x16x6x6xf32>)
// CHECK:         arith.addf
// CHECK-NOT:   brachml.conv
func.func @test_conv_bias(
    %input:  tensor<1x3x8x8xf32>,
    %weight: tensor<16x3x3x3xf32>,
    %bias:   tensor<16xf32>
) -> tensor<1x16x6x6xf32> {
  %0 = brachml.conv %input, %weight, %bias : tensor<16xf32>
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    transposed = false
    output_padding = [0, 0]
    groups = 1
    : tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x6x6xf32>
  return %0 : tensor<1x16x6x6xf32>
}

// ── conv (with padding) ───────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_conv_pad
// CHECK:       tensor.pad %{{.*}} low[0, 0, 1, 1] high[0, 0, 1, 1]
// CHECK:         arith.constant 0.000000e+00 : f32
// CHECK:         tensor.yield
// CHECK:       } : tensor<1x3x8x8xf32> to tensor<1x3x10x10xf32>
// CHECK:       linalg.conv_2d_nchw_fchw
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<1x3x10x10xf32>, tensor<16x3x3x3xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<1x16x8x8xf32>)
// CHECK-NOT:   brachml.conv
func.func @test_conv_pad(
    %input:  tensor<1x3x8x8xf32>,
    %weight: tensor<16x3x3x3xf32>
) -> tensor<1x16x8x8xf32> {
  %0 = brachml.conv %input, %weight
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    transposed = false
    output_padding = [0, 0]
    groups = 1
    : tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x8x8xf32>
  return %0 : tensor<1x16x8x8xf32>
}

// ── max_pool ──────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_maxpool
// CHECK:       linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<1x4x4x4xf32>)
// CHECK:       linalg.pooling_nchw_max
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<1x4x8x8xf32>, tensor<2x2xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<1x4x4x4xf32>)
// CHECK-NOT:   brachml.max_pool
func.func @test_maxpool(%input: tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32> {
  %0 = brachml.max_pool %input
    kernel_size = [2, 2]
    stride = [2, 2]
    padding = [0, 0]
    dilation = [1, 1]
    ceil_mode = false
    : tensor<1x4x8x8xf32> -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// ── permute ───────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_permute
// CHECK:       linalg.transpose ins(%{{.*}} : tensor<2x3x4xf32>)
// CHECK-SAME:                   outs(%{{.*}} : tensor<4x3x2xf32>)
// CHECK-SAME:                   permutation = [2, 1, 0]
// CHECK-NOT:   brachml.permute
func.func @test_permute(%a: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
  %0 = brachml.permute %a dims = [2, 1, 0]
    : tensor<2x3x4xf32> -> tensor<4x3x2xf32>
  return %0 : tensor<4x3x2xf32>
}

// ── reshape ───────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_reshape
// CHECK:       tensor.collapse_shape %{{.*}} {{\[\[}}0, 1, 2{{\]\]}}
// CHECK-SAME:    tensor<2x3x4xf32> into tensor<24xf32>
// CHECK:       tensor.expand_shape %{{.*}} {{\[\[}}0, 1{{\]\]}}
// CHECK-SAME:    tensor<24xf32> into tensor<4x6xf32>
// CHECK-NOT:   brachml.reshape
func.func @test_reshape(%a: tensor<2x3x4xf32>) -> tensor<4x6xf32> {
  %0 = brachml.reshape %a size = [4, 6]
    : tensor<2x3x4xf32> -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

// ── reshape (flatten to 1D) ───────────────────────────────────────────────────

// CHECK-LABEL: func.func @test_reshape_flatten
// CHECK:       tensor.collapse_shape %{{.*}} {{\[\[}}0, 1, 2{{\]\]}}
// CHECK-SAME:    tensor<2x3x4xf32> into tensor<24xf32>
// CHECK-NOT:   tensor.expand_shape
// CHECK-NOT:   brachml.reshape
func.func @test_reshape_flatten(%a: tensor<2x3x4xf32>) -> tensor<24xf32> {
  %0 = brachml.reshape %a size = [24]
    : tensor<2x3x4xf32> -> tensor<24xf32>
  return %0 : tensor<24xf32>
}

// ── requant ───────────────────────────────────────────────────────────────────
// Verifies the full dequant→scale→requant arithmetic sequence.

// CHECK-LABEL: func.func @test_requant
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}} : tensor<1x8xi8>) outs(%{{.*}} : tensor<1x8xi8>)
// CHECK:       ^bb0(%{{.*}}: i8, %{{.*}}: i8):
// CHECK:         arith.extsi %{{.*}} : i8 to i32
// CHECK:         arith.subi
// CHECK:         arith.sitofp %{{.*}} : i32 to f32
// CHECK:         arith.mulf
// CHECK:         arith.fptosi %{{.*}} : f32 to i32
// CHECK:         arith.addi
// CHECK:         arith.maxsi
// CHECK:         arith.minsi
// CHECK:         arith.trunci %{{.*}} : i32 to i8
// CHECK:         linalg.yield %{{.*}} : i8
// CHECK-NOT:   brachml.requant
func.func @test_requant(%a: tensor<1x8xi8>) -> tensor<1x8xi8> {
  %0 = brachml.requant %a
    from (scale = 1.000000e-01, zp = 0)
    to   (scale = 5.000000e-01, zp = 0)
    : tensor<1x8xi8>
  return %0 : tensor<1x8xi8>
}

// ── batch_norm ────────────────────────────────────────────────────────────────
// Verifies that per-channel params [C] are broadcast over [N,C,H,W].

// CHECK-LABEL: func.func @test_batch_norm
// CHECK:       linalg.generic
// CHECK-SAME:    tensor<1x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK:       ^bb0({{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32):
// CHECK:         math.sqrt
// CHECK:         arith.divf
// CHECK:         arith.subf
// CHECK:         arith.mulf
// CHECK-NOT:   brachml.batch_norm
func.func @test_batch_norm(
    %input: tensor<1x4x8x8xf32>,
    %weight: tensor<4xf32>,
    %bias: tensor<4xf32>,
    %mean: tensor<4xf32>,
    %var: tensor<4xf32>
) -> tensor<1x4x8x8xf32> {
  %0 = brachml.batch_norm %input, %weight : tensor<4xf32>, %bias : tensor<4xf32>
    eps = 1.000000e-05, %mean, %var
    : tensor<1x4x8x8xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// ── fused_region (relu + add) ─────────────────────────────────────────────────
// Verifies tile loop structure: forall over [1,4,8,8], extract slices,
// lower relu and add on dynamic tile shapes, insert result back.

// CHECK-LABEL: func.func @test_fused_region
// CHECK:       scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (8, 8) step (32, 32)
// CHECK:         tensor.extract_slice %{{.*}}[0, 0, %{{.*}}, %{{.*}}]
// CHECK-SAME:      tensor<1x4x8x8xf32> to tensor<1x4x?x?xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<1x4x?x?xf32>) outs(%{{.*}} : tensor<1x4x?x?xf32>)
// CHECK:             arith.maximumf
// CHECK:         linalg.add ins(%{{.*}}, %{{.*}} : tensor<1x4x?x?xf32>, tensor<1x4x?x?xf32>)
// CHECK-SAME:              outs(%{{.*}} : tensor<1x4x?x?xf32>)
// CHECK:         tensor.parallel_insert_slice
// CHECK-SAME:      tensor<1x4x?x?xf32> into tensor<1x4x8x8xf32>
// CHECK-NOT:   brachml.fused_region
func.func @test_fused_region(
    %a: tensor<1x4x8x8xf32>,
    %b: tensor<1x4x8x8xf32>
) -> tensor<1x4x8x8xf32> {
  %0 = brachml.fused_region(%a, %b : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>)
      -> tensor<1x4x8x8xf32> {
  ^bb0(%arg0: tensor<1x4x8x8xf32>, %arg1: tensor<1x4x8x8xf32>):
    %relu = brachml.relu %arg0 : tensor<1x4x8x8xf32>
    %add  = brachml.add %relu, %arg1 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32> -> tensor<1x4x8x8xf32>
    brachml.yield %add : tensor<1x4x8x8xf32>
  }
  return %0 : tensor<1x4x8x8xf32>
}
