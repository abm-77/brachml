
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.export import export, ExportedProgram
from torch.utils.data import DataLoader
from torchvision import datasets

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = "cifar_model.pt2"

EPOCHS = 100
KERNEL_SIZE = 3
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WORKERS = 24

# We can use transforms on the data to artificially 
# expand the data set by adding randomly flipped and cropped versions
# of each image
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

TRAINING_DATA = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=TRAIN_TRANSFORM
)

TEST_DATA = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=TEST_TRANSFORM
)

TRAIN_DATALOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True, shuffle=True)
TEST_DATALOADER = DataLoader(TEST_DATA, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True)

class CifarNet(nn.Module):

    def __init__(self) -> None:
        super(CifarNet, self).__init__()

        # 3 convolutional layers
        #
        # NOTE: The first conv layer's in-channels is determined by the 
        # number of channels in the input. In this case CIFAR10 images
        # have 3 depth channels (RGB). It is typical to have increasing 
        # number of channels as you venture deeper into the model.
        # This is because the earlier layers pick up simple features (like edges), so you 
        # actually don't need that many filters, but deeper layers detect complex combinations 
        # of the ealier features, therefore need more capacity.
        self.convs = nn.ModuleList([
            nn.Conv2d(3, 32, KERNEL_SIZE),
            nn.Conv2d(32, 64, KERNEL_SIZE),
            nn.Conv2d(64, 128, KERNEL_SIZE),
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
        ])
        self.fc = nn.Linear(2 * 2 * 128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [CONV -> BN -> RELU -> POOL] * 3 -> FC
        for conv, bn in zip (self.convs, self.norms):
            x = F.max_pool2d(F.relu(bn(conv(x))), 2)
        x = torch.flatten(x, 1) 
        return self.fc(x)

def train_loop (dataloader, model, loss_fn, optimizer) -> None:
    model.train()
    for _, (X, y) in enumerate(dataloader):
        # move data to device
        X, y = X.to(DEVICE), y.to(DEVICE)

        # clear gradients from last batch
        optimizer.zero_grad()

        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        loss.backward()
        optimizer.step()

def test_loop (dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f} \n")
    return test_loss

def main():
    model = CifarNet().to(DEVICE)
    print(model)

    if os.path.exists(MODEL_PATH):
        print("Found saved model, loading...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # scheduler helps reduce the learning rate when we plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        print(f"training model for {EPOCHS} epochs")
        for t in range (EPOCHS):
            print(f"Epoch {t+1}\n---------------------")
            train_loop(TRAIN_DATALOADER, model, loss_fn, optimizer)
            test_loss = test_loop(TEST_DATALOADER, model, loss_fn)
            scheduler.step(test_loss)
        print("Done!")

        # save trained model
        torch.save(model.state_dict(), "cifar_model.pt2")
        print(f"Model saved to {MODEL_PATH}")

    # export core ATen IR for rvml
    model.eval()
    example_input = torch.randn(1, 3, 32, 32).to(DEVICE)
    exported = export(model, (example_input,))
    with torch.no_grad():
        exported = exported.run_decompositions(decomp_table=None)
    print(exported)

if __name__ == "__main__":
    main()
