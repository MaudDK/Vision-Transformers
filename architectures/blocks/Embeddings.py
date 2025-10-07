import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, use_spatial: bool = False):
        super(PositionalEncoding, self).__init__()
        self.use_spatial = use_spatial

        if use_spatial:
            grid_size = int(num_patches**0.5)
            self.position_embeddings = nn.Parameter(torch.zeros(1, grid_size, grid_size, embed_dim))

        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_spatial:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.reshape(B, H, W, C)
            x = x + self.position_embeddings
            x = x.reshape(B, N, C)
        else:
            x = x + self.position_embeddings

        return x
