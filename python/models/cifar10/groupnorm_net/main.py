"""
GroupNorm + LeakyReLU + AvgPool model.

Architecture:
  Input [1, 3, 32, 32]
  → Conv2d(3, 16, 3×3, pad=1)
  → GroupNorm(4, 16)     Divides 16 channels into 4 groups of 4. Normalizes
                          mean/variance within each group independently.
                          Unlike BatchNorm: stats computed per-sample, not
                          per-batch — works correctly at batch size 1 which
                          is typical at inference time on-device.
  → LeakyReLU(0.1)       Like ReLU but negative inputs are scaled by 0.1
                          rather than clamped to 0. Prevents neurons from
                          permanently dying (outputting zero for all inputs)
                          which can happen with standard ReLU when the bias
                          pushes the pre-activation negative.
  → AvgPool2d(2, 2)      Spatial downsampling by averaging 2×2 windows.
                          Smoother than MaxPool (no hard winner-take-all),
                          preferred in some architectures for its gradient
                          properties during training.
                          [N, 16, 32, 32] → [N, 16, 16, 16]
  → Conv2d(16, 32, 3×3, pad=1)
  → GroupNorm(4, 32)     4 groups of 8 channels.
  → LeakyReLU(0.1)
  → AdaptiveAvgPool2d(1) Global average pool → [N, 32, 1, 1]
  → Flatten → Linear(32, 10)

New ops introduced:
  - brachml.group_norm (or lowered via batch_norm equivalent)
  - brachml.leaky_relu
  - brachml.avg_pool (AvgPool2d, distinct from AdaptiveAvgPool)
"""

import torch
import torch.nn as nn

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/groupnorm_net/groupnorm_net_model.pt2"
FLOAT_PATH = "models/cifar10/groupnorm_net/groupnorm_net_float.pt2"
QUANT_PATH = "models/cifar10/groupnorm_net/groupnorm_net_core_aten.pt2"


class GroupNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.gn1    = nn.GroupNorm(4, 16)
        self.conv2  = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.gn2    = nn.GroupNorm(4, 32)
        self.pool   = nn.AvgPool2d(2, 2)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(32, 10)
        self.lrelu  = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.lrelu(self.gn1(self.conv1(x)))   # [N, 16, 32, 32]
        x = self.pool(x)                           # [N, 16, 16, 16]
        x = self.lrelu(self.gn2(self.conv2(x)))   # [N, 32, 16, 16]
        x = self.gap(x)                            # [N, 32, 1, 1]
        x = torch.flatten(x, 1)
        return self.fc(x)


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        GroupNormNet(), CHECKPOINT, train_loader, test_loader
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
