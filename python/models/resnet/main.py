import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.export import export
from torch.utils.data import DataLoader
from torchvision import datasets
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from brachml_quant import extract_calibration
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/resnet/resnet_model.pt2"

EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WORKERS = 24

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

TRAINING_DATA = datasets.CIFAR10(
    root="models/resnet/data", train=True, download=True, transform=TRAIN_TRANSFORM
)

TEST_DATA = datasets.CIFAR10(
    root="models/resnet/data", train=False, download=True, transform=TEST_TRANSFORM
)

TRAIN_DATALOADER = DataLoader(
    TRAINING_DATA,
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    pin_memory=True,
    shuffle=True,
)
TEST_DATALOADER = DataLoader(
    TEST_DATA, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True
)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + x)


class SmallResNet(nn.Module):
    """ResNet-style CIFAR-10 classifier.
    stem(3->32) -> block(32) -> downsample(32->64) -> block(64) -> pool -> fc(10)
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = ResBlock(32)
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = ResBlock(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {100 * correct:.1f}%, Avg loss: {test_loss:.6f}")
    return test_loss


def main():
    model = SmallResNet()
    print(model)

    if os.path.exists(MODEL_PATH):
        print("Found saved model, loading...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        model = model.to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        print(f"Training for {EPOCHS} epochs")
        for t in range(EPOCHS):
            print(f"Epoch {t + 1}")
            train_loop(TRAIN_DATALOADER, model, loss_fn, optimizer)
            test_loss = test_loop(TEST_DATALOADER, model, loss_fn, DEVICE)
            scheduler.step(test_loss)
        print("Done!")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    model.eval()
    example_input = (torch.randn(1, 3, 32, 32),)
    exported = export(model, example_input)

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))
    prepared = prepare_pt2e(exported.module(), quantizer)

    with torch.no_grad():
        count = 0
        for X, _ in TRAIN_DATALOADER:
            for j in range(len(X)):
                prepared(X[j].unsqueeze(0))
                count += 1
                if count >= 1000:
                    break
            if count >= 1000:
                break

    model_quantized = convert_pt2e(prepared)

    correct, total = 0, 0
    with torch.no_grad():
        for X, y in TEST_DATALOADER:
            for j in range(len(X)):
                pred = model_quantized(X[j].unsqueeze(0))
                correct += (pred.argmax(1) == y[j]).item()
                total += 1
    print(f"Quantized accuracy: {100 * correct / total:.1f}%")

    exported = export(model_quantized, example_input).run_decompositions(
        decomp_table=None
    )

    calibration = extract_calibration(exported.graph_module)
    with open("models/resnet/calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    torch.export.save(exported, "models/resnet/resnet_core_aten.pt2")
    print(exported)


if __name__ == "__main__":
    main()
