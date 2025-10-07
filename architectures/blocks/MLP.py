import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super(MLPBlock, self).__init__()

        mlp_size = int(input_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_size, out_features=output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)