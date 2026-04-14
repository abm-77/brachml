"""Generate test .pt2 models for brachml_import lit tests.

Run this once to create the test fixtures:
    python tests/brachml_import/generate_models.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save(name, model, example_inputs):
    exported = export(model, example_inputs).run_decompositions(decomp_table=None)
    path = os.path.join(MODELS_DIR, f"{name}.pt2")
    torch.export.save(exported, path)
    print(f"  {path}")


class ReLUModel(nn.Module):
    def forward(self, x):
        return F.relu(x)


class AddModel(nn.Module):
    def forward(self, x, y):
        return x + y


class MatMulModel(nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y)


class ReshapeModel(nn.Module):
    def forward(self, x):
        return x.view(2, 12)


class ConvReLUPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        return F.max_pool2d(F.relu(self.conv(x)), 2)


class AddMMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    print("Generating test models:")
    save("relu", ReLUModel(), (torch.randn(1, 3, 32, 32),))
    save("add", AddModel(), (torch.randn(4, 4), torch.randn(4, 4)))
    save("matmul", MatMulModel(), (torch.randn(4, 8), torch.randn(8, 16)))
    save("reshape", ReshapeModel(), (torch.randn(2, 3, 4),))
    save("conv_relu_pool", ConvReLUPoolModel(), (torch.randn(1, 3, 32, 32),))
    save("addmm", AddMMModel(), (torch.randn(1, 8),))
    print("Done.")
