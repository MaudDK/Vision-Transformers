import torch
import torch.nn as nn
import pandas as pd
import os
from architectures.models.ViTSeg import ViTSeg
from datasets.CAMUSData import CamusSegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from architectures.encoders.ViT import VisionTransformer
from training.train import Trainer
from losses.DiceLoss import DiceLoss, DiceBCELoss


if __name__ == "__main__":
    CAMUS_DIR = "./datasets/CAMUS"
    camus_df = pd.read_csv(os.path.join(CAMUS_DIR, "CAMUS_dataset.csv"))
    train_df = camus_df[camus_df['split'] == 'train']
    test_df = camus_df[camus_df['split'] == 'test']

    IMG_SIZE = 224

    image_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = CamusSegmentationDataset(train_df, transform=image_transforms)
    test_dataset = CamusSegmentationDataset(test_df, transform=image_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for image, mask in train_dataset:
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        break

    model = ViTSeg(img_size=IMG_SIZE, 
                   patch_size=16, 
                   in_channels=1, 
                   embed_dim=768, 
                   num_heads=12, 
                   mlp_ratio=4.0, 
                   depth=12, 
                   dropout=0.0, 
                   out_channels=1, 
                   load_pretrained_encoder=False,
                   freeze_encoder=False)
    
    checkpoint_path = "./checkpoints/20251007/epoch_10_metric_0.8330_ViTSeg"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path), strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = DiceLoss()
    trainer = Trainer(model, train_dataloader, test_dataloader, epochs=10000, criterion=criterion, optimizer=optimizer, model_name="ViTSeg")
    trainer.train()
    trainer.plot_losses()
