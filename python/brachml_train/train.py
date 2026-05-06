import math
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Type alias for a function that maps (model_output, targets) → scalar loss.
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _default_loss_fn() -> LossFn:
    return nn.CrossEntropyLoss()


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_fn: LossFn | None = None,
) -> None:
    if loss_fn is None:
        loss_fn = _default_loss_fn()
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
    loss_fn: LossFn | None = None,
) -> float:
    """Evaluate *model* on *dataloader* and return average loss.

    For 2-D outputs (classification) also prints accuracy.
    For higher-rank outputs (e.g. language model logits [B, T, V]) prints
    perplexity (exp(loss)) instead — accuracy over a vocabulary is meaningless.
    """
    if loss_fn is None:
        loss_fn = _default_loss_fn()
    model.eval()
    num_batches = len(dataloader)
    total_loss  = 0.0
    correct, size = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            if pred.dim() == 2:   # classification — argmax over class dim
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                size    += y.size(0)

    avg_loss = total_loss / num_batches
    if size > 0:
        print(f"Accuracy: {100 * correct / size:.1f}%, Avg loss: {avg_loss:.6f}")
    else:
        print(f"Avg loss: {avg_loss:.6f}, Perplexity: {math.exp(avg_loss):.2f}")
    return avg_loss


def train_or_load(
    model: nn.Module,
    checkpoint_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str | None = None,
    loss_fn: LossFn | None = None,
) -> nn.Module:
    """Train *model* for *epochs* epochs, or reload weights from *checkpoint_path*
    if it already exists.  Returns the model on CPU in eval mode.

    *loss_fn* maps ``(model_output, targets) → scalar loss``.  Defaults to
    ``CrossEntropyLoss``, which covers standard classification.  Pass a custom
    callable for other tasks — e.g. for a character LM::

        loss_fn = lambda logits, y: F.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1)
        )

    The checkpoint is a plain ``state_dict`` written by ``torch.save``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if loss_fn is None:
        loss_fn = _default_loss_fn()

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model.cpu()

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"Training for {epochs} epochs on {device}")
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        train_loop(train_loader, model, optimizer, device, loss_fn)
        val_loss = test_loop(val_loader, model, device, loss_fn)
        scheduler.step(val_loss)
    print("Done!")

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return model.cpu()
