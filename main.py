from architectures.ViT import VisionTransformer
import torch

if __name__ == "__main__":
    model = VisionTransformer(img_size=1024, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, mlp_ratio=4.0, depth=12, dropout=0.0)
    model.load_state_dict(torch.load("./checkpoints/medsam1_image_encoder.pth"), strict=False)
    print(model)
