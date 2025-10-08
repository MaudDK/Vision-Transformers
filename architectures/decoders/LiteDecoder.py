import torch
import torch.nn as nn
from architectures.blocks.MLP import MLPBlock
from architectures.blocks.LayerNorm import LayerNorm2d
from architectures.blocks.Transformer import TransformerBlock

class Decoder(nn.Module):
    """
    Decoder Module for ViT.
    Input shape of (B, N, C) where N is number of patches.
    Output shape of (B, out_channels, H, W)
    """
    def __init__(self, embed_dim=768, out_channels=1):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            LayerNorm2d(256),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.neck(x)
        x = self.upsample(x)
        return x

