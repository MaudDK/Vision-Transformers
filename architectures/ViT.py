import torch
import torch.nn as nn
from architectures.blocks.Embeddings import PatchEmbedding, PositionalEncoding
from architectures.blocks.Transformer import TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0, 
                 depth: int = 12, 
                 dropout: float = 0.0
    ):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Transformer Blocks
        self.image_encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(self.patch_embedding.num_patches, embed_dim, use_spatial=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.pos_encoding(x)

        for block in self.image_encoder:
            x = block(x)

        return x