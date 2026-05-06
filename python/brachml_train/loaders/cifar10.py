from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Per-channel mean and std computed over the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def make_cifar10_loaders(
    data_root: str = "models/cifar10/data",
    batch_size: int = 512,
    num_workers: int = 24,
    mean: tuple[float, float, float] = CIFAR10_MEAN,
    std: tuple[float, float, float] = CIFAR10_STD,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10.

    *mean* and *std* default to the CIFAR-10 training-set statistics but can
    be overridden — useful when fine-tuning a model pre-trained on a different
    distribution, or when experimenting with per-model normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
