// No calibration file → pass emits a warning and no-ops (returns success).
// RUN: %brachml-opt --brachml-quantize %s 2>&1 | FileCheck %s --check-prefix=NOFILE
// NOFILE: warning: brachml-quantize: no calibration data passed

// Missing path → pass emits an error and signals failure.
// RUN: not %brachml-opt --brachml-quantize=calibration-file=/does/not/exist.json %s 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: error: brachml-quantize: cannot open /does/not/exist.json

// Malformed JSON → pass emits a parse error and signals failure.
// RUN: echo "not json" > %t.bad.json
// RUN: not %brachml-opt --brachml-quantize=calibration-file=%t.bad.json %s 2>&1 | FileCheck %s --check-prefix=PARSE
// PARSE: error: brachml-quantize: failed to parse JSON

// Non-object root (array, scalar, etc.) → error and signals failure.
// RUN: echo "[1, 2, 3]" > %t.arr.json
// RUN: not %brachml-opt --brachml-quantize=calibration-file=%t.arr.json %s 2>&1 | FileCheck %s --check-prefix=ROOT
// ROOT: error: brachml-quantize: calibration JSON root must be an object

func.func @f(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "brachml.relu"(%arg0) {brachml.node_name = "x"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
