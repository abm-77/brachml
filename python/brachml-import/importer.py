"""BrachML Importer: converts torch.export Core ATen IR to BrachML MLIR.

Usage:
    python -m brachml-import.importer model.pt2 -o output.mlir
"""

import argparse
import operator
import sys

from mlir.ir import Context, Module, InsertionPoint, Location
from mlir.ir import RankedTensorType, F32Type, F64Type, IntegerType, DenseElementsAttr
from mlir.dialects import func, arith, ml_program

import torch
from torch.export import ExportedProgram

from mlir.dialects._brachml_ops_gen import (
    AddOp,
    BatchNormOp,
    ConvOp,
    MatMulOp,
    MaxPool,
    PermuteOp,
    ReLUOp,
    ReshapeOp,
)


def _lower_conv(node, value_map):
    """aten.convolution(input, weight, bias, stride, padding, dilation,
    transposed, output_padding, groups)"""
    input = value_map[node.args[0]]
    weight = value_map[node.args[1]]
    bias = value_map[node.args[2]] if node.args[2] is not None else None
    stride = list(node.args[3])
    padding = list(node.args[4])
    dilation = list(node.args[5])
    transposed = bool(node.args[6])
    output_padding = list(node.args[7])
    groups = int(node.args[8])
    result_type = convert_tensor_type(node.meta["val"])

    return ConvOp(
        result_type,
        input=input,
        weight=weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
    ).result


def _lower_batch_norm(node, value_map):
    """aten._native_batch_norm_legit_no_training(input, weight, bias,
    running_mean, running_var, momentum, eps)"""
    input = value_map[node.args[0]]
    weight = value_map[node.args[1]] if node.args[1] is not None else None
    bias = value_map[node.args[2]] if node.args[2] is not None else None
    running_mean = value_map[node.args[3]]
    running_var = value_map[node.args[4]]
    # args[5] is momentum — skip, only used for training
    eps = float(node.args[6])
    # batch_norm returns a tuple; take the first element
    result_meta = node.meta["val"]
    if isinstance(result_meta, (tuple, list)):
        result_meta = result_meta[0]
    result_type = convert_tensor_type(result_meta)

    return BatchNormOp(
        result_type,
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
        weight=weight,
        bias=bias,
    ).result


def _lower_max_pool(node, value_map):
    """aten.max_pool2d_with_indices(input, kernel_size, stride, padding,
    dilation, ceil_mode)"""
    input = value_map[node.args[0]]
    kernel_size = list(node.args[1])
    stride = list(node.args[2]) if len(node.args) > 2 and node.args[2] else kernel_size
    padding = list(node.args[3]) if len(node.args) > 3 else [0] * len(kernel_size)
    dilation = list(node.args[4]) if len(node.args) > 4 else [1] * len(kernel_size)
    ceil_mode = bool(node.args[5]) if len(node.args) > 5 else False
    # max_pool2d_with_indices returns (output, indices); take output
    result_meta = node.meta["val"]
    if isinstance(result_meta, (tuple, list)):
        result_meta = result_meta[0]
    result_type = convert_tensor_type(result_meta)

    return MaxPool(
        result_type,
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    ).result


def _lower_matmul(node, value_map):
    """aten.mm(lhs, rhs) / aten.bmm(lhs, rhs)"""
    lhs = value_map[node.args[0]]
    rhs = value_map[node.args[1]]
    result_type = convert_tensor_type(node.meta["val"])

    return MatMulOp(result_type, lhs=lhs, rhs=rhs).result


def _lower_add(node, value_map):
    """aten.add.Tensor(lhs, rhs)"""
    lhs = value_map[node.args[0]]
    rhs = value_map[node.args[1]]
    result_type = convert_tensor_type(node.meta["val"])

    return AddOp(result_type, lhs=lhs, rhs=rhs).result


def _lower_relu(node, value_map):
    """aten.relu(input)"""
    x = value_map[node.args[0]]
    result_type = convert_tensor_type(node.meta["val"])

    return ReLUOp(input=x, results=[result_type]).result


def _lower_reshape(node, value_map):
    """aten.view(input, size)"""
    input = value_map[node.args[0]]
    size = list(node.args[1])
    result_type = convert_tensor_type(node.meta["val"])

    return ReshapeOp(result_type, input=input, size=size).result


def _lower_permute(node, value_map):
    """aten.permute(input, dims)"""
    input = value_map[node.args[0]]
    dims = list(node.args[1])
    result_type = convert_tensor_type(node.meta["val"])

    return PermuteOp(result_type, input=input, dims=dims).result


def _lower_addmm(node, value_map):
    """addmm(bias, lhs, rhs) -> add(matmul(lhs, rhs), bias)"""
    bias = value_map[node.args[0]]
    lhs = value_map[node.args[1]]
    rhs = value_map[node.args[2]]
    result_type = convert_tensor_type(node.meta["val"])

    mm = MatMulOp(result_type, lhs=lhs, rhs=rhs).result
    return AddOp(result_type, lhs=mm, rhs=bias).result


# Core ATen op -> lowering function
ATEN_TO_BRACHML = {
    "aten.convolution.default": _lower_conv,
    "aten._native_batch_norm_legit_no_training.default": _lower_batch_norm,
    "aten.max_pool2d_with_indices.default": _lower_max_pool,
    "aten.mm.default": _lower_matmul,
    "aten.bmm.default": _lower_matmul,
    "aten.add.Tensor": _lower_add,
    "aten.relu.default": _lower_relu,
    "aten.view.default": _lower_reshape,
    "aten.permute.default": _lower_permute,
    "aten.addmm.default": _lower_addmm,
}


def convert_dtype(dtype: torch.dtype):
    """Convert a torch dtype to an MLIR element type."""
    if dtype == torch.float32:
        return F32Type.get()
    elif dtype == torch.float64:
        return F64Type.get()
    elif dtype == torch.int8:
        return IntegerType.get_signless(8)
    elif dtype == torch.int32:
        return IntegerType.get_signless(32)
    elif dtype == torch.int64:
        return IntegerType.get_signless(64)
    elif dtype == torch.bool:
        return IntegerType.get_signless(1)

    raise NotImplementedError(f"{dtype} not supported")


def convert_tensor_type(tensor_meta):
    """Convert a torch FakeTensor / tensor metadata to an MLIR RankedTensorType."""
    return RankedTensorType.get(tensor_meta.shape, convert_dtype(tensor_meta.dtype))


def _emit_global(name: str, tensor: torch.Tensor, module_ip: InsertionPoint):
    """Emit an ml_program.global op at module level for a parameter/buffer."""
    tensor = tensor.contiguous().detach()
    mlir_type = RankedTensorType.get(list(tensor.shape), convert_dtype(tensor.dtype))
    attr = DenseElementsAttr.get(tensor.numpy(), type=mlir_type)
    with module_ip:
        ml_program.GlobalOp(name, mlir_type, is_mutable=False, value=attr)
    return name, mlir_type


def _load_global(name: str, mlir_type):
    """Emit an ml_program.global_load_const to load a global into an SSA value."""
    return ml_program.GlobalLoadConstOp(mlir_type, name).result


def convert_node(node, value_map):
    """Convert a single torch.fx Node to BrachML MLIR operations."""
    target = node.target

    # getitem extracts an element from a tuple-returning op (e.g.
    # max_pool2d_with_indices, batch_norm). Our lowerings return a
    # single MLIR Value stored under the source node . If a downstream node tries to use an index we
    # didn't lower (e.g. pool indices), it will fail at that point.
    if target is operator.getitem:
        value_map[node] = value_map[node.args[0]]
        return

    target_str = str(target)
    lower = ATEN_TO_BRACHML.get(target_str)
    if lower is None:
        raise NotImplementedError(f"unsupported aten op: {target_str}")
    value_map[node] = lower(node, value_map)


def import_exported_program(exported: ExportedProgram) -> Module:
    """Convert an ExportedProgram to an MLIR Module with BrachML ops.

    Args:
        exported: a torch.export ExportedProgram containing Core ATen IR

    Returns:
        An MLIR Module containing func.func with BrachML dialect ops.
    """
    sig = exported.graph_signature
    param_names = set(sig.inputs_to_parameters.keys())
    buffer_names = set(sig.inputs_to_buffers.keys())

    # Map placeholder name -> state_dict key for params and buffers
    param_to_key = sig.inputs_to_parameters
    buffer_to_key = sig.inputs_to_buffers

    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown() as loc:
        ctx.load_all_available_dialects()
        module = Module.create(loc)

        # Only user inputs become function arguments
        user_input_types = []
        user_input_nodes = []
        output_types = []
        for node in exported.graph.nodes:
            if node.op == "placeholder":
                if node.name not in param_names and node.name not in buffer_names:
                    user_input_types.append(convert_tensor_type(node.meta["val"]))
                    user_input_nodes.append(node)
            elif node.op == "output":
                for output in node.args[0]:
                    output_types.append(convert_tensor_type(output.meta["val"]))

        module_ip = InsertionPoint(module.body)

        # Emit ml_program.global ops for parameters and buffers
        globals_info = {}  # placeholder name -> (global_name, mlir_type)
        for node in exported.graph.nodes:
            if node.op != "placeholder":
                continue
            if node.name in param_names:
                key = param_to_key[node.name]
                tensor = exported.state_dict[key]
                globals_info[node.name] = _emit_global(key, tensor, module_ip)
            elif node.name in buffer_names:
                key = buffer_to_key[node.name]
                tensor = exported.state_dict[key]
                globals_info[node.name] = _emit_global(key, tensor, module_ip)

        ft = func.FunctionType.get(user_input_types, output_types)
        with module_ip:
            fn = func.FuncOp("model", ft, loc=loc)

        entry_block = fn.add_entry_block()
        with InsertionPoint(entry_block):
            value_map = {}
            arg_index = 0

            for node in exported.graph.nodes:
                if node.op == "placeholder":
                    if node.name in globals_info:
                        name, mlir_type = globals_info[node.name]
                        value_map[node] = _load_global(name, mlir_type)
                    else:
                        value_map[node] = entry_block.arguments[arg_index]
                        arg_index += 1
                elif node.op == "call_function":
                    convert_node(node, value_map)
                elif node.op == "output":
                    return_vals = [value_map[n] for n in node.args[0]]
                    func.ReturnOp(return_vals)
        return module


def main():
    parser = argparse.ArgumentParser(
        description="Import a torch.export model into BrachML MLIR"
    )
    parser.add_argument("model", help="Path to .pt2 exported model")
    parser.add_argument(
        "-o", "--output", default="-", help="Output file (default: stdout)"
    )
    args = parser.parse_args()

    # Load the exported program
    exported = torch.export.load(args.model)

    # Convert to MLIR
    module = import_exported_program(exported)

    if module is None:
        print("Error: import failed", file=sys.stderr)
        sys.exit(1)

    # Output
    output = str(module)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)


if __name__ == "__main__":
    main()
