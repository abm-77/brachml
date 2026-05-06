import torch
import torch.nn as nn
import torch.nn.functional as F

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/resnet/resnet_model.pt2"
FLOAT_PATH = "models/cifar10/resnet/resnet_float.pt2"
QUANT_PATH = "models/cifar10/resnet/resnet_core_aten.pt2"


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + x)


class SmallResNet(nn.Module):
    """stem(3→32) → ResBlock(32) → downsample(32→64) → ResBlock(64) → pool → fc(10)"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = ResBlock(32)
        self.down   = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = ResBlock(64)
        self.fc     = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        SmallResNet(), CHECKPOINT, train_loader, test_loader
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
