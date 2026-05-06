"""
Concat net — SqueezeNet fire module style.

Architecture:
  Input [1, 3, 32, 32]
  → Conv2d(3, 8, 1×1)  [squeeze]   Compresses from 3 → 8 channels cheaply.
                                     A 1×1 conv is just a learned channel
                                     projection — no spatial filtering.
  → ReLU
  → Two parallel expand paths:
      path1: Conv2d(8, 8, 1×1)      Learns channel combinations (no spatial).
      path2: Conv2d(8, 8, 3×3, pad=1) Learns spatial features.
  → Concat([path1, path2], dim=1)   Joins along channel dim → [N, 16, H, W].
                                     Each output channel comes from either
                                     the 1×1 or 3×3 path — the model learns
                                     which features to expand spatially.
  → BN → ReLU
  → AdaptiveAvgPool2d(1)
  → Flatten → Linear(16, 10)

New ops introduced:
  - brachml.concat (torch.cat on channel dim)
"""

import torch
import torch.nn as nn

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/concat_net/concat_net_model.pt2"
FLOAT_PATH = "models/cifar10/concat_net/concat_net_float.pt2"
QUANT_PATH = "models/cifar10/concat_net/concat_net_core_aten.pt2"


class FireModule(nn.Module):
    """SqueezeNet fire module: squeeze then expand via two parallel paths."""
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        self.squeeze     = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.expand_1x1  = nn.Conv2d(squeeze_channels, expand_channels, 1)
        self.expand_3x3  = nn.Conv2d(squeeze_channels, expand_channels, 3, padding=1)

    def forward(self, x):
        x  = torch.relu(self.squeeze(x))
        e1 = torch.relu(self.expand_1x1(x))
        e3 = torch.relu(self.expand_3x3(x))
        return torch.cat([e1, e3], dim=1)   # [N, expand*2, H, W]


class ConcatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fire = FireModule(3, 8, 8)   # squeeze=8, expand=8 → 16 output channels
        self.bn   = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fire(x)            # [N, 16, 32, 32]
        x = torch.relu(self.bn(x))
        x = self.pool(x)            # [N, 16, 1, 1]
        x = torch.flatten(x, 1)
        return self.fc(x)


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        ConcatNet(), CHECKPOINT, train_loader, test_loader
    )
    example = (torch.randn(1, 3, 32, 32),)
    export_model(model.eval(), example, FLOAT_PATH)
    export_and_quantize(
        model.eval(), example,
        calib_loader=train_loader,
        output_path=QUANT_PATH,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
