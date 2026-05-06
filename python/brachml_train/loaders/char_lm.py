import urllib.request
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Karpathy's tiny-Shakespeare — ~1 MB, 1M chars, good char-LM benchmark.
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def _download_text(url: str, dest: str) -> str:
    import os
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)
    with open(dest, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset(Dataset):
    """Sliding-window character dataset.

    Each sample is (input_ids, target_ids) where target is input shifted
    right by one — the standard next-token prediction setup.
    """

    def __init__(self, text: str, block_size: int, stoi: dict[str, int]):
        self.block_size = block_size
        data = [stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


@dataclass
class CharLMLoaders:
    train: DataLoader
    val: DataLoader
    vocab_size: int
    stoi: dict[str, int]   # char → index
    itos: dict[int, str]   # index → char


def make_char_lm_loaders(
    text_path: str = "models/lm/data/tinyshakespeare.txt",
    url: str = TINY_SHAKESPEARE_URL,
    block_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
) -> CharLMLoaders:
    """Return train/val loaders for a character-level language model.

    Downloads *url* to *text_path* on the first call.  Subsequent calls reuse
    the cached file.  The vocabulary is built from the unique characters in the
    full corpus, so train and val see the same stoi/itos mapping.
    """
    text = _download_text(url, text_path)

    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)

    # Split at the character level before building Dataset objects so we don't
    # leak future tokens into the validation set through sliding windows.
    split = int(len(text) * (1 - val_fraction))
    train_text = text[:split]
    val_text   = text[split:]

    train_ds = CharDataset(train_text, block_size, stoi)
    val_ds   = CharDataset(val_text,   block_size, stoi)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Corpus: {len(text):,} chars | vocab: {vocab_size} | "
          f"train: {len(train_ds):,} seqs | val: {len(val_ds):,} seqs")

    return CharLMLoaders(
        train=train_loader,
        val=val_loader,
        vocab_size=vocab_size,
        stoi=stoi,
        itos=itos,
    )
