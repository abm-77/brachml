"""
Tiny Vision Transformer (ViT) for CIFAR-10.

Architecture:
  Input [1, 3, 32, 32]
  → Patch embedding: split into 4×4 patches → 64 tokens of dim 48
                     Linear projection: 48 → embed_dim=64
  → Prepend [CLS] token                  learnable vector prepended to
                                          the sequence. After all attention
                                          blocks, only the CLS position is
                                          fed to the classifier — it acts as
                                          an aggregate representation of the
                                          whole sequence.
  → Add positional embedding             learned [65, 64] table added to
                                          every token. Transformers have no
                                          built-in notion of position (unlike
                                          conv), so this is how spatial
                                          structure is injected.
  → TransformerEncoder × 2 blocks:
      LayerNorm                           Normalizes across the embedding dim
                                          (not batch/channel like BN/GN).
                                          Works at any batch size, preferred
                                          for sequence models.
      MultiHeadSelfAttention(heads=4):
        Q, K, V = Linear(64, 64) × 3     Project each token to queries, keys,
                                          values. Split into 4 heads of dim 16.
        scores = Q @ K.T / sqrt(16)      Scaled dot-product: measures how much
                                          each token should attend to each other.
        weights = Softmax(scores)         Convert to probabilities along the
                                          sequence dimension.
        out = weights @ V                 Weighted sum of values.
        Linear(64, 64)                    Project back to embedding dim.
      Residual add
      LayerNorm
      MLP: Linear(64, 128) → GELU → Linear(128, 64)
                                          GELU is smoother than ReLU — it has
                                          a small negative region rather than
                                          hard-clamping. Standard in transformers.
      Residual add
  → LayerNorm
  → CLS token → Linear(64, 10)

New ops introduced:
  - layer_norm
  - softmax
  - scaled_dot_product_attention (or matmul + softmax + matmul)
  - gelu
  - many more linear/matmul ops
  - reshape/permute for head splitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/tiny_vit/tiny_vit_model.pt2"
FLOAT_PATH = "models/cifar10/tiny_vit/tiny_vit_float.pt2"
QUANT_PATH = "models/cifar10/tiny_vit/tiny_vit_core_aten.pt2"


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3)
        self.proj      = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3*C] → [B, N, 3, heads, head_dim] → [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, heads, N, head_dim]

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = embed_dim * mlp_ratio
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    ):
        super().__init__()
        num_patches  = (img_size // patch_size) ** 2   # 64 patches
        patch_dim    = in_channels * patch_size ** 2   # 48

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.patch_size  = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        TinyViT(), CHECKPOINT, train_loader, test_loader
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
