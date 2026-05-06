"""
Depthwise Separable Conv model — MobileNet building block.

Architecture:
  Input [1, 3, 32, 32]
  → DepthwiseConv2d(3, 3, 3×3, groups=3)  groups=in=out means one filter per
                                            channel — spatial filtering only,
                                            no channel mixing. Much cheaper
                                            than a standard conv.
  → BN → ReLU
  → PointwiseConv2d(3, 16, 1×1)           1×1 conv mixes channels cheaply.
                                            Together with the depthwise conv
                                            this approximates a full conv at
                                            ~8–9x fewer multiply-adds.
  → BN → ReLU
  → AdaptiveAvgPool2d(1, 1)               Collapses [N, C, H, W] → [N, C, 1, 1]
                                            by averaging each channel's spatial
                                            map. Output size is fixed regardless
                                            of input H×W — no hardcoded spatial
                                            dimension in the classifier.
  → Flatten [N, 16]
  → Linear(16, 10)

New ops introduced:
  - brachml.conv with groups=C_in=C_out (depthwise)
  - brachml.avg_pool (via AdaptiveAvgPool2d)
"""

import torch
import torch.nn as nn

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/depthwise_sep/depthwise_sep_model.pt2"
FLOAT_PATH = "models/cifar10/depthwise_sep/depthwise_sep_float.pt2"
QUANT_PATH = "models/cifar10/depthwise_sep/depthwise_sep_core_aten.pt2"


class DepthwiseSepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Depthwise: groups=3 means each of the 3 input channels gets its own
        # 3×3 filter. Output channels = input channels = 3.
        self.dw_conv = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        self.dw_bn   = nn.BatchNorm2d(3)
        # Pointwise: 1×1 conv projects from 3 → 16 channels.
        self.pw_conv = nn.Conv2d(3, 16, 1, bias=False)
        self.pw_bn   = nn.BatchNorm2d(16)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.dw_bn(self.dw_conv(x)))   # depthwise
        x = torch.relu(self.pw_bn(self.pw_conv(x)))   # pointwise
        x = self.pool(x)                               # [N, 16, 1, 1]
        x = torch.flatten(x, 1)                        # [N, 16]
        return self.fc(x)


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        DepthwiseSepNet(), CHECKPOINT, train_loader, test_loader
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
