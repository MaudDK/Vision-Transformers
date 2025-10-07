import torch
import torch.nn as nn
from architectures.blocks.MLP import MLPBlock

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.self_attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )

        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLPBlock(embed_dim, embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln_1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x
