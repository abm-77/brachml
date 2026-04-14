// RUN: %brachml-opt --brachml-quantize=calibration-file=%S/quantization_calibration.json %s | FileCheck %s

// All 8 fields present in the calibration entry get attached. Scales go in
// as f32 FloatAttrs, zero_points as i32 IntegerAttrs.
// CHECK-LABEL: func.func @all_fields
// CHECK-DAG:   brachml.input_scale = 1.000000e-01 : f32
// CHECK-DAG:   brachml.input_zero_point = -128 : i32
// CHECK-DAG:   brachml.weight_scale = 2.000000e-01 : f32
// CHECK-DAG:   brachml.weight_zero_point = 0 : i32
// CHECK-DAG:   brachml.bias_scale = 3.000000e-01 : f32
// CHECK-DAG:   brachml.bias_zero_point = 64 : i32
// CHECK-DAG:   brachml.output_scale = 4.000000e-01 : f32
// CHECK-DAG:   brachml.output_zero_point = -64 : i32
func.func @all_fields(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "brachml.relu"(%arg0) {brachml.node_name = "all_fields_op"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// Partial entries: only the listed fields are attached, others are omitted.
// CHECK-LABEL: func.func @partial
// CHECK-DAG:   brachml.input_scale = 5.000000e-01 : f32
// CHECK-DAG:   brachml.output_scale = 6.000000e-01 : f32
// CHECK-NOT:   brachml.input_zero_point
// CHECK-NOT:   brachml.weight_scale
// CHECK-NOT:   brachml.weight_zero_point
// CHECK-NOT:   brachml.bias_scale
// CHECK-NOT:   brachml.bias_zero_point
// CHECK-NOT:   brachml.output_zero_point
func.func @partial(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "brachml.relu"(%arg0) {brachml.node_name = "partial_op"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// Ops without the `brachml.node_name` join key are left alone (ml_program
// loads, func/return, constants, etc. all fall into this bucket in real IR).
// CHECK-LABEL: func.func @no_node_name
// CHECK-NOT:   brachml.input_scale
// CHECK-NOT:   brachml.output_scale
func.func @no_node_name(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "brachml.relu"(%arg0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// Ops whose node_name has no matching calibration entry are left alone.
// CHECK-LABEL: func.func @no_match
// CHECK-NOT:   brachml.input_scale
// CHECK-NOT:   brachml.output_scale
func.func @no_match(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "brachml.relu"(%arg0) {brachml.node_name = "not_in_json"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
