import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np

class CamusSegmentationDataset(Dataset):
    def __init__(self, dataframe, transform= None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def load_nii(self, filepath, frame=0):
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)

        # Get specific frame if it's a sequence
        if array.ndim == 3:
            frame = array[frame]
        else:
            frame = array

        return frame

    def __getitem__(self, idx):
        # Load image and mask paths
        image_path = self.dataframe.iloc[idx]['nifti_path']
        mask_path = self.dataframe.iloc[idx]['mask_path']

        if 'frame' in image_path:
            frame_num = int(image_path.split('_frame')[1].split('.')[0])
            image_path = image_path.replace(f'_frame{frame_num}', '')
            mask_path = mask_path.replace(f'_frame{frame_num}', '')
        else:
            frame_num = 0

        # Load image and mask
        image = self.load_nii(image_path, frame=frame_num)
        mask = self.load_nii(mask_path, frame=frame_num)
        # Convert to binary segmentation: 0 for background, 1 for endocardium only
        mask = (mask == 1).astype(np.uint8)

        image = Image.fromarray(image)
        
        # Convert mask to PIL Image (keep as is for segmentation)
        mask = mask.astype(np.uint8)
        mask_pil = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            # Apply resize transform to mask but keep original values
            mask_resized = transforms.Resize((256, 256))(mask_pil)
            mask = torch.tensor(np.array(mask_resized), dtype=torch.long)

        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# if __name__ == "__main__":
#     CAMUS_DIR = "./datasets/CAMUS"
#     camus_df = pd.read_csv(os.path.join(CAMUS_DIR, "CAMUS_dataset.csv"))
#     train_df = camus_df[camus_df['split'] != 'test']
#     test_df = camus_df[camus_df['split'] == 'test']

#     print(f"Training samples: {len(train_df)}")
#     print(f"Testing samples: {len(test_df)}")

#     image_transforms = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])

#     train_dataset = CamusSegmentationDataset(train_df, transform=image_transforms)
#     test_dataset = CamusSegmentationDataset(test_df, transform=image_transforms)

