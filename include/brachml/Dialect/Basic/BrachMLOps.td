include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "BrachMLDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class BrachML_Op<string mnemonic, list<Trait> traits = []> 
  : Op<BrachML_Dialect, mnemonic, traits>;

// Primitive Ops
def BrachML_AddOp : BrachML_Op<"add", [Pure]> {
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

def BrachML_ReLUOp : BrachML_Op<"relu", [Pure, SameOperandsAndResultType]> {
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

def BrachML_ConvOp : BrachML_Op<"conv", [Pure]> {
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

def BrachML_BatchNormOp : BrachML_Op<"batch_norm", [Pure, AttrSizedOperandSegments]> {
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

def BrachML_MaxPool : BrachML_Op<"max_pool", [Pure]> {
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

def BrachML_MatMulOp : BrachML_Op<"matmul", [Pure]> {
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

def BrachML_ReshapeOp : BrachML_Op<"reshape", [Pure]> {
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

def BrachML_PermuteOp : BrachML_Op<"permute", [Pure]> {
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

// Fused Ops
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
