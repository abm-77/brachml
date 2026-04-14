"""Per-op quant param extraction from a pt2e-converted FX GraphModule."""


def _target_str(node):
    return str(node.target) if node.op == "call_function" else ""


def _is_quant(node):
    t = _target_str(node)
    return "quantize_per_tensor" in t and "dequantize" not in t


def _is_dequant(node):
    return "dequantize_per_tensor" in _target_str(node)


def _qparams(qnode):
    # quantize/dequantize_per_tensor args: (x, scale, zero_point, qmin, qmax, dtype)
    return float(qnode.args[1]), int(qnode.args[2])


def _layer_fqn(node):
    """Deepest nn.Module FQN from node.meta['nn_module_stack'], or '' for
    functional ops called directly inside the root module."""
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return ""
    *_, last = stack.values()
    return last[0] if last and last[0] else ""


def extract_calibration(gm):
    """Pull per-op quantization parameters out of a pt2e-converted GraphModule.

    For every compute op (i.e. every call_function that isn't a
    quantize/dequantize node itself) we record:
      - `op`                             the aten target string
      - `layer`                          deepest nn.Module FQN ('' for root-
                                         level functional ops like F.relu)
      - if args[i] is fed by a dequantize_per_tensor:
            {input,weight,bias}_{scale,zero_point}
      - if a direct user is a quantize_per_tensor:
            output_{scale,zero_point}

    Ops in the middle of a fusion pattern (where the fused block's output
    quant lives on the last op, e.g. conv inside a conv+relu fusion) will
    have input scales but no output_scale — that's an honest reflection of
    the graph. Downstream consumers can chase the quantized producer/user
    locally when they need it.

    Keyed by `node.name` in the provided graph. Pair with the MLIR importer
    (which stamps `brachml.node_name` on every op) for a zero-translation
    join between this dict and the imported IR.
    """
    calibration = {}
    arg_names = {0: "input", 1: "weight", 2: "bias"}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_quant(node) or _is_dequant(node):
            continue

        entry = {
            "op": _target_str(node),
            "layer": _layer_fqn(node),
        }

        for i, a in enumerate(node.args):
            if hasattr(a, "op") and _is_dequant(a):
                scale, zp = _qparams(a)
                name = arg_names.get(i, f"arg{i}")
                entry[f"{name}_scale"] = scale
                entry[f"{name}_zero_point"] = zp

        for user in node.users:
            if _is_quant(user):
                scale, zp = _qparams(user)
                entry["output_scale"] = scale
                entry["output_zero_point"] = zp
                break

        calibration[node.name] = entry
    return calibration
