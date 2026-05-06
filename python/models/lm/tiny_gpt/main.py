"""
Tiny GPT — character-level autoregressive language model.

Architecture:
  Input [B, T]  (integer token indices, T = block_size = 64)
  → Embedding(vocab_size, embed_dim=64)     Token embedding table: each integer
                                             index is looked up to produce a
                                             dense vector.  This is a gather op —
                                             fundamentally different from the
                                             linear projections in conv/ViT.
  + Embedding(block_size, embed_dim=64)     Learned positional embedding: same
                                             structure as token embedding but
                                             indexed by position 0..T-1.
                                             Added (not concatenated) to the
                                             token embeddings.
  → GPTBlock × 2:
      LayerNorm
      CausalSelfAttention(heads=4):
        Q, K, V = Linear(64, 64) × 3
        scores  = Q @ K.T / sqrt(16)
        weights = Softmax(scores, masked)   Upper-triangular mask ensures token i
                                             can only attend to positions ≤ i.
                                             This is the key difference from ViT:
                                             bidirectional → causal.
        out     = weights @ V
        Linear(64, 64)
      Residual add
      LayerNorm
      MLP: Linear(64, 128) → GELU → Linear(128, 64)
      Residual add
  → LayerNorm
  → Linear(64, vocab_size)                  Logit over vocab at each position.

  Loss: cross_entropy(logits[B, T, V] → [B*T, V], targets[B, T] → [B*T])
  Training objective: predict the next character at every position in parallel.

New ops vs TinyViT:
  - nn.Embedding          → aten::embedding (gather)
  - is_causal=True SDPA   → causal mask applied inside scaled_dot_product_attention
  - Output is [B, T, V]   → cross_entropy over vocab, not a single class score

Shared with TinyViT:
  - layer_norm, gelu, linear, scaled_dot_product_attention, reshape/permute
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from brachml_train import make_char_lm_loaders, train_or_load
from brachml_train.export import export_model

CHECKPOINT = "models/lm/tiny_gpt/tiny_gpt_model.pt"
FLOAT_PATH = "models/lm/tiny_gpt/tiny_gpt_float.pt2"

BLOCK_SIZE = 64
EMBED_DIM  = 64
N_HEADS    = 4
N_LAYERS   = 2
FF_DIM     = 128


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, T, C)
        return self.proj(x)


class GPTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = CausalSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = BLOCK_SIZE,
        embed_dim: int  = EMBED_DIM,
        num_heads: int  = N_HEADS,
        n_layers: int   = N_LAYERS,
        ff_dim: int     = FF_DIM,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks  = nn.Sequential(*[
            GPTBlock(embed_dim, num_heads, ff_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        x    = self.tok_emb(x) + self.pos_emb(pos)    # [B, T, C]
        x    = self.blocks(x)
        x    = self.norm(x)
        return self.head(x)                            # [B, T, vocab_size]

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy autoregressive generation from a [1, T] prompt."""
        for _ in range(max_new_tokens):
            ctx    = prompt[:, -self.block_size:]
            logits = self(ctx)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt  = torch.cat([prompt, next_id], dim=1)
        return prompt


def main() -> None:
    loaders = make_char_lm_loaders()

    loss_fn = lambda logits, y: F.cross_entropy(
        logits.view(-1, loaders.vocab_size), y.view(-1)
    )

    model = train_or_load(
        TinyGPT(vocab_size=loaders.vocab_size),
        CHECKPOINT,
        train_loader=loaders.train,
        val_loader=loaders.val,
        epochs=10,
        lr=3e-3,
        loss_fn=loss_fn,
    )

    # Quick generation sample
    model.eval()
    seed = torch.tensor([[loaders.stoi[c] for c in "ROMEO:"]], dtype=torch.long)
    out  = model.generate(seed, max_new_tokens=200)
    print("\n--- sample ---")
    print("".join(loaders.itos[i] for i in out[0].tolist()))
    print("--- end ---\n")

    # Float export only — LM quantization (weight-only or KV-cache quant) is
    # a separate concern from the per-tensor symmetric int8 used for CNN/ViT.
    example = (torch.zeros(1, BLOCK_SIZE, dtype=torch.long),)
    export_model(model, example, FLOAT_PATH)


if __name__ == "__main__":
    main()
