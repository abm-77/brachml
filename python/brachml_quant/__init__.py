"""BrachML quantization utilities.

Walks a pt2e-converted torch FX GraphModule to pull per-op quantization
parameters (scale / zero_point) into a plain dict, ready to serialize as
JSON for the MLIR QuantizationPass to consume.

The keys in the returned dict are `node.name` values from the same graph
the MLIR importer reads, so the join on the MLIR side is trivial: every
imported op carries a `brachml.node_name` attribute equal to its source
FX name.

Typical use from a model script:

    from brachml_quant import extract_calibration
    converted = convert_pt2e(prepared)
    exported = export(converted, example_input).run_decompositions(decomp_table=None)
    calibration = extract_calibration(exported.graph_module)
"""

from .calibration import extract_calibration
from .graph import dequant_scale, maybe_insert_requants, output_quant_scale

__all__ = [
    "extract_calibration",
    "dequant_scale",
    "maybe_insert_requants",
    "output_quant_scale",
]
