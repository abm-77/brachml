import torch
import torch.nn as nn
import torch.nn.functional as F

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/naive_cnn/naive_cnn_model.pt2"
FLOAT_PATH = "models/cifar10/naive_cnn/naive_cnn_float.pt2"
QUANT_PATH = "models/cifar10/naive_cnn/naive_cnn_core_aten.pt2"


class NaiveCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(3,   32, 3),
            nn.Conv2d(32,  64, 3),
            nn.Conv2d(64, 128, 3),
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
        ])
        self.fc = nn.Linear(2 * 2 * 128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.norms):
            x = F.max_pool2d(F.relu(bn(conv(x))), 2)
        x = torch.flatten(x, 1)
        return self.fc(x)


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        NaiveCNN(), CHECKPOINT, train_loader, test_loader
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
