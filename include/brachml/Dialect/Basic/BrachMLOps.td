include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "BrachMLDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class BrachML_Op<string mnemonic, list<Trait> traits = []>
  : Op<BrachML_Dialect, mnemonic, traits>;

// Marker trait for ops eligible to participate in a fused_region body.
// Checked statically by isFusableOp in BeamSearchFusionPass.
def BrachML_FusableOpTrait : NativeOpTrait<"FusableOpTrait"> {
  let cppNamespace = "::brachml";
}

// Primitive Ops
def BrachML_AddOp : BrachML_Op<"add", [Pure, BrachML_FusableOpTrait]> {
  let summary = "Elementwise addition of two tensors (with broadcasting)";

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_ReLUOp : BrachML_Op<"relu", [Pure, SameOperandsAndResultType, BrachML_FusableOpTrait]> {
  let summary = "Elementwise ReLU on input tensor";

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];
}

def BrachML_ConvOp : BrachML_Op<"conv", [Pure, BrachML_FusableOpTrait]> {
  let summary = "Performs general convolution on input tensor";

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    Optional<AnyTensor>:$bias,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    BoolAttr:$transposed,
    DenseI64ArrayAttr:$output_padding,
    I64Attr:$groups
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
      $input `,` $weight (`,` $bias^ `:` type($bias))?
      `stride` `=` $stride
      `padding` `=` $padding
      `dilation` `=` $dilation
      `transposed` `=` $transposed
      `output_padding` `=` $output_padding
      `groups` `=` $groups
      attr-dict `:` type($input) `,` type($weight) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_BatchNormOp : BrachML_Op<"batch_norm", [Pure, AttrSizedOperandSegments, BrachML_FusableOpTrait]> {
  let summary = "Performs batch normalization on the input tensor";

  let arguments = (ins 
    AnyTensor:$input,
    Optional<AnyTensor>:$weight,
    Optional<AnyTensor>:$bias,
    AnyTensor:$running_mean,
    AnyTensor:$running_var,
    F64Attr:$eps
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input (`,` $weight^ `:` type($weight))? (`,` $bias^ `:` type($bias))?
    `eps` `=` $eps
    `,` $running_mean `,` $running_var
    attr-dict `:` type($input) `,` type($running_mean) `,` type($running_var) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_MaxPool : BrachML_Op<"max_pool", [Pure, BrachML_FusableOpTrait]> {
  let summary = "Performs general max_pool on the input tensor";

  let arguments = (ins 
    AnyTensor:$input,
    DenseI64ArrayAttr:$kernel_size, 
    DenseI64ArrayAttr:$stride,      
    DenseI64ArrayAttr:$padding,     
    DenseI64ArrayAttr:$dilation,    
    BoolAttr:$ceil_mode
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input
    `kernel_size` `=` $kernel_size
    `stride` `=` $stride
    `padding` `=` $padding
    `dilation` `=` $dilation
    `ceil_mode` `=` $ceil_mode
    attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_MatMulOp : BrachML_Op<"matmul", [Pure, BrachML_FusableOpTrait]> {
  let summary = "Performs general matrix multiplication on the input tensors";

  let arguments = (ins 
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_ReshapeOp : BrachML_Op<"reshape", [Pure, BrachML_FusableOpTrait]> {
  let summary = "Reshapes the input tensor";

  let arguments = (ins 
    AnyTensor:$input,
    DenseI64ArrayAttr:$size
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input `size` `=` $size attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_PermuteOp : BrachML_Op<"permute", [Pure, BrachML_FusableOpTrait]> {
  let summary = "Permutes the input tensor";

  let arguments = (ins 
    AnyTensor:$input,
    DenseI64ArrayAttr:$dims
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input `dims` `=` $dims attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_RequantOp : BrachML_Op<"requant", [Pure, SameOperandsAndResultType, BrachML_FusableOpTrait]> {
  let summary = "Requantizes input from `srcScale` to `dstScale`";

  let arguments = (ins
    AnyTensor:$input,
    F32Attr:$srcScale,
    I32Attr:$srcZP,
    F32Attr:$dstScale,
    I32Attr:$dstZP
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input
    `from` `(` `scale` `=` $srcScale `,` `zp` `=` $srcZP `)`
    `to` `(` `scale` `=` $dstScale `,` `zp` `=` $dstZP `)`
    attr-dict `:` type($result)
  }];
}

// Fused Ops

def BrachML_YieldOp : BrachML_Op<"yield", [
  Pure,
  ReturnLike,
  Terminator
]> {
  let summary = "Yield values from a fused_region";

  let arguments = (ins
    Variadic<AnyTensor>:$values
  );

  let results = (outs);

  let assemblyFormat = [{
    ($values^ `:` type($values))? attr-dict
  }];

  let builders = [OpBuilder<(ins)>];

  let hasVerifier = 1;
}

def BrachML_FusedRegionOp : BrachML_Op<"fused_region", [
  Pure,
  SingleBlockImplicitTerminator<"YieldOp">,
  RecursiveMemoryEffects
]> {
  let summary = "A fused subgraph to be lowered as a single kernel";
  let description = [{
    Container op produced by the fusion pipeline (initial pattern-based pass
    followed by BEAM search). The body region holds the fused BrachML ops;
    results are yielded by a `brachml.yield` terminator.

    Example (conv + batch_norm + relu fused into one kernel):
    ```mlir
    %out = brachml.fused_region(
               %input, %weight, %bias, %mean, %var
               : tensor<1x32x30x30xf32>, tensor<64x32x3x3xf32>,
                 tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
           -> tensor<1x64x28x28xf32> {
      %conv = brachml.conv %input, %weight, %bias ... -> tensor<1x64x28x28xf32>
      %bn   = brachml.batch_norm %conv ... -> tensor<1x64x28x28xf32>
      %relu = brachml.relu %bn : tensor<1x64x28x28xf32>
      brachml.yield %relu : tensor<1x64x28x28xf32>
    }
    ```
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs
    Variadic<AnyTensor>:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    `(` $inputs `:` type($inputs) `)` `->` type($results) $body attr-dict
  }];

  let hasVerifier = 1;
}

def BrachML_LinearOp : BrachML_Op<"linear", [Pure]> {
  let summary = "Performs a linear transformation: output = input @ weight.T + bias";

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    Optional<AnyTensor>:$bias
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^ `:` type($bias))?
    attr-dict `:` type($input) `,` type($weight) `->` type($result)
  }];

  let hasVerifier = 1;
}

def BrachML_ConvBnReluOp : BrachML_Op<"conv_bn_relu", [Pure, AttrSizedOperandSegments]> {
  let summary = "Fusion of conv-bn-relu";

  let arguments = (ins
    // conv args
    AnyTensor:$input,
    AnyTensor:$conv_weight,
    Optional<AnyTensor>:$conv_bias,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    BoolAttr:$transposed,
    DenseI64ArrayAttr:$output_padding,
    I64Attr:$groups,

    // bn args
    Optional<AnyTensor>:$bn_weight,
    Optional<AnyTensor>:$bn_bias,
    AnyTensor:$running_mean,
    AnyTensor:$running_var,
    F64Attr:$eps
  );

  let results = (outs 
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $input `,` $conv_weight (`,` $conv_bias^ `:` type($conv_bias))?
    `stride` `=` $stride
    `padding` `=` $padding
    `dilation` `=` $dilation
    `transposed` `=` $transposed
    `output_padding` `=` $output_padding
    `groups` `=` $groups
    (`,` $bn_weight^ `:` type($bn_weight))? (`,` $bn_bias^ `:` type($bn_bias))?
    `,` $running_mean `,` $running_var
    `eps` `=` $eps
    attr-dict `:` type($input) `,` type($conv_weight) `,` type($running_mean) `,` type($running_var) `->` type($result)
  }];
}


