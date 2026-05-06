import os

import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
from torch.utils.data import DataLoader
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def export_model(
    model: nn.Module,
    example_input: tuple,
    output_path: str,
) -> ExportedProgram:
    """Export a float model to a core ATen .pt2 file without quantization.

    Use this when you want the unquantized graph for debugging, reference
    accuracy measurement, or as input to a later quantization step.
    """
    model = model.eval().cpu()
    exported = export(model, example_input)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.export.save(exported, output_path)
    print(f"Exported float model to {output_path}")
    print(exported)

    return exported


def export_and_quantize(
    model: nn.Module,
    example_input: tuple,
    calib_loader: DataLoader,
    output_path: str,
    test_loader: DataLoader | None = None,
    n_calib: int = 1000,
) -> ExportedProgram:
    """Export, quantize, and save a model as a core ATen .pt2 file.

    Steps:
      1. ``torch.export`` → static core ATen graph
      2. ``prepare_pt2e`` inserts per-tensor symmetric int8 observer nodes
      3. Calibrate on *n_calib* samples from *calib_loader*
      4. ``convert_pt2e`` freezes scales / zero-points
      5. Optionally evaluate quantized accuracy using *test_loader*
      6. Re-export the quantized ``GraphModule`` and save to *output_path*

    The re-export in step 6 ensures that the .pt2 file and the graph that the
    importer sees have identical node names, so calibration data joins cleanly.

    Returns the final ``ExportedProgram``.
    """
    model = model.eval().cpu()
    exported = export(model, example_input)

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))
    prepared = prepare_pt2e(exported.module(), quantizer)

    # Calibration: run *n_calib* single-sample forward passes so observers
    # collect activation statistics across realistic inputs.
    with torch.no_grad():
        count = 0
        for X, _ in calib_loader:
            for j in range(len(X)):
                prepared(X[j].unsqueeze(0))
                count += 1
                if count >= n_calib:
                    break
            if count >= n_calib:
                break

    model_quantized = convert_pt2e(prepared)

    if test_loader is not None:
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                for j in range(len(X)):
                    pred = model_quantized(X[j].unsqueeze(0))
                    correct += (pred.argmax(1) == y[j]).item()
                    total += 1
        print(f"Quantized accuracy: {100 * correct / total:.1f}%")

    # Re-export so the saved .pt2 is fully decomposed and has stable node names.
    exported_quantized = export(model_quantized, example_input).run_decompositions(
        decomp_table=None
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.export.save(exported_quantized, output_path)
    print(f"Exported quantized model to {output_path}")
    print(exported_quantized)

    return exported_quantized
