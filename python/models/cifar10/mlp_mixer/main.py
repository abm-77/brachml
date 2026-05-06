"""
Tiny MLP-Mixer for CIFAR-10.

Architecture:
  Input [1, 3, 32, 32]
  → Patch embedding: 4×4 patches → 64 tokens, projected to embed_dim=64
  → MixerBlock × 2:
      LayerNorm
      Token mixing MLP:                   Transpose to [B, embed_dim, num_tokens],
                                          apply a shared MLP across the token
                                          dimension — every channel's activations
                                          at all positions are mixed together.
                                          This replaces attention for spatial
                                          communication between tokens.
          Linear(64, 32) → GELU → Linear(32, 64)
      Residual add
      LayerNorm
      Channel mixing MLP:                 Standard MLP applied per-token across
                                          the channel/feature dimension. Same
                                          role as the FFN in a transformer block.
          Linear(64, 128) → GELU → Linear(128, 64)
      Residual add
  → LayerNorm
  → Mean over token dim                  Average all 64 token representations
                                          rather than using a CLS token.
                                          Equivalent to global average pooling
                                          over the spatial positions.
  → Linear(64, 10)

New ops vs ViT:
  - Same: layer_norm, gelu, linear
  - No softmax / scaled_dot_product_attention (no attention at all)
  - Demonstrates that transformers don't require attention — useful baseline

Shared ops with ViT worth noting:
  - The token mixing transpose ([B, N, C] → [B, C, N] → [B, N, C]) produces
    permute ops which your compiler needs to handle.
  - Mean reduction over sequence dim produces a reduce_mean op.
"""

import torch
import torch.nn as nn

from brachml_train import make_cifar10_loaders, train_or_load, export_model, export_and_quantize

CHECKPOINT = "models/cifar10/mlp_mixer/mlp_mixer_model.pt2"
FLOAT_PATH = "models/cifar10/mlp_mixer/mlp_mixer_float.pt2"
QUANT_PATH = "models/cifar10/mlp_mixer/mlp_mixer_core_aten.pt2"


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, embed_dim, token_dim, channel_dim):
        super().__init__()
        self.norm1       = nn.LayerNorm(embed_dim)
        self.token_mix   = nn.Sequential(
            nn.Linear(num_tokens, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_tokens),
        )
        self.norm2       = nn.LayerNorm(embed_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(embed_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, embed_dim),
        )

    def forward(self, x):
        # x: [B, num_tokens, embed_dim]
        y = self.norm1(x)
        y = y.transpose(1, 2)       # [B, embed_dim, num_tokens]
        y = self.token_mix(y)       # mix across num_tokens
        y = y.transpose(1, 2)       # [B, num_tokens, embed_dim]
        x = x + y

        x = x + self.channel_mix(self.norm2(x))
        return x


class TinyMLPMixer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        token_dim=32,
        channel_dim=128,
    ):
        super().__init__()
        num_patches     = (img_size // patch_size) ** 2   # 64
        patch_dim       = in_channels * patch_size ** 2   # 48
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.blocks      = nn.Sequential(*[
            MixerBlock(num_patches, embed_dim, token_dim, channel_dim)
            for _ in range(depth)
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

        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)           # [B, 64]
        return self.head(x)


def main():
    train_loader, test_loader = make_cifar10_loaders()
    model = train_or_load(
        TinyMLPMixer(), CHECKPOINT, train_loader, test_loader
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
