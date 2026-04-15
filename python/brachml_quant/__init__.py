"""BrachML quantization utilities.

Walks a pt2e-converted torch FX GraphModule to pull per-op quantization
parameters (scale / zero_point) for use during import. The importer reads
these directly from the FX graph and stamps them as attributes on each op —
no calibration file or separate MLIR pass needed.
"""

from .graph import dequant_scale, maybe_insert_requants, output_quant_scale

__all__ = [
    "dequant_scale",
    "maybe_insert_requants",
    "output_quant_scale",
]
