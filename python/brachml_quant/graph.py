"""FX graph analysis helpers for pt2e-quantized graphs.

These operate on torch.fx Nodes directly. The importer calls
`maybe_insert_requants` during lowering to emit brachml.requant ops
where binary-op inputs have mismatched scales.
"""


def dequant_scale(fx_node):
    """If fx_node is a dequantize_per_tensor, return (scale, zp). Else None."""
    if fx_node.op == "call_function" and "dequantize_per_tensor" in str(fx_node.target):
        return float(fx_node.args[1]), int(fx_node.args[2])
    return None


def output_quant_scale(fx_node):
    """Find the quantize_per_tensor that directly consumes fx_node's output."""
    for user in fx_node.users:
        t = str(user.target) if user.op == "call_function" else ""
        if "quantize_per_tensor" in t and "dequantize" not in t:
            return float(user.args[1]), int(user.args[2])
    return None


def maybe_insert_requants(node, value_map, RequantOp):
    """For binary ops whose two FX inputs come from dequantize nodes with
    different scales, insert brachml.requant ops to normalize both sides to
    the op's output scale before lowering. Mutates value_map in-place.

    `RequantOp` is passed in to avoid a circular import between
    brachml_quant and the generated dialect bindings."""
    lhs_node, rhs_node = node.args[0], node.args[1]
    lhs_sq = dequant_scale(lhs_node)
    rhs_sq = dequant_scale(rhs_node)

    if not lhs_sq or not rhs_sq or lhs_sq == rhs_sq:
        return

    target = output_quant_scale(node) or lhs_sq
    target_scale, target_zp = target

    for arg_node, sq in ((lhs_node, lhs_sq), (rhs_node, rhs_sq)):
        src_scale, src_zp = sq
        if src_scale == target_scale and src_zp == target_zp:
            continue
        val = value_map[arg_node]
        rq = RequantOp(val, src_scale, src_zp, target_scale, target_zp)
        value_map[arg_node] = rq.result
