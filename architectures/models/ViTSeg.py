from architectures.decoders.LiteDecoder import Decoder
from architectures.encoders.ViT import VisionTransformer
import torch
import torch.nn as nn

class ViTSeg(nn.Module):
    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0, 
                 depth: int = 12, 
                 dropout: float = 0.0,
                 out_channels: int = 1,
                 load_pretrained_encoder: bool = False,
                 freeze_encoder: bool = False
    ):
        super(ViTSeg, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Vision Transformer Encoder
        self.image_encoder = VisionTransformer(img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, depth, dropout)

        if load_pretrained_encoder:
            if img_size != 1024 or patch_size != 16:
                raise ValueError("Pretrained encoder requires img_size=1024 and patch_size=16")
            self.image_encoder.load_state_dict(torch.load("./checkpoints/medsam1_image_encoder.pth"), strict=False)
            if freeze_encoder:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False

        # SAM Decoder
        self.decoder = Decoder(embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_encoder(x)
        x = self.decoder(x)
        return x