import os
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Patch

def make_camus_dataframe(camus_path = "./datasets/CAMUS"):
    nifti_dir = os.path.join(camus_path, "database_nifti")
    training = os.path.join(camus_path, "database_split", "subgroup_training.txt")
    validation = os.path.join(camus_path, "database_split", "subgroup_validation.txt")
    testing = os.path.join(camus_path, "database_split", "subgroup_testing.txt")


    data = []
    view = ["2CH", "4CH"]
    instant = ["ED", "ES", "half_sequence"]

    for split, file_path in [("train", training), ("val", validation), ("test", testing)]:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    subject = line.strip()
                    for v in view:
                        for i in instant:
                            nifti_path = os.path.join(nifti_dir, subject, f"{subject}_{v}_{i}.nii.gz")
                            nifti_path_gt = os.path.join(nifti_dir, subject, f"{subject}_{v}_{i}_gt.nii.gz")

                            if i == "half_sequence":
                                cfg_path = os.path.join(nifti_dir, subject, f"Info_{v}.cfg")
                                if os.path.exists(cfg_path):
                                    cfg_info = {}
                                    with open(cfg_path, "r") as cfg_file:
                                        for line in cfg_file:
                                            if ':' in line:
                                                key, value = line.strip().split(':', 1)
                                                cfg_info[key.strip()] = value.strip()

                                    num_frames = int(cfg_info.get("NbFrame", 1))
                                    # for frame in range(0, num_frames, max(1, num_frames // 5)):
                                    for frame in range(2, num_frames - 1, 2):
                                        nifti_path_frame = nifti_path.replace('.nii.gz', f'_frame{frame}.nii.gz')
                                        nifti_path_gt_frame = nifti_path_gt.replace('.nii.gz', f'_frame{frame}.nii.gz')
                                        data.append({"split": split, "subject": subject, "view": v, "instant": i, "nifti_path": nifti_path_frame, "mask_path": nifti_path_gt_frame})
                            
                            else:
                                data.append({"split": split, "subject": subject, "view": v, "instant": i, "nifti_path": nifti_path, "mask_path": nifti_path_gt})

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(camus_path, "CAMUS_dataset.csv"), index=False)

def load_image(filepath):
    image = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(image)
    print(f"Loaded: {filepath}\nShape: {array.shape}, Dtype: {array.dtype}")
    
    # Get middle frame if it's a sequence
    if array.ndim == 3:
        mid_frame = array.shape[0] // 2
        frame = array[mid_frame]
    else:
        frame = array

    return frame

def show_images_in_grid(images, titles=None, cols=3, figsize=(15, 10)):
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of 2D array

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')  # Hide unused subplots
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def load_image_and_mask(image_path):
    # Load image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # Load middle frame if sequence
    if image_array.ndim == 3:
        mid_frame = image_array.shape[0] // 2
        image_array = image_array[mid_frame]

    # Infer corresponding mask path (e.g., _ED.nii.gz → _ED_gt.nii.gz)
    mask_path = image_path.replace('.nii.gz', '_gt.nii.gz')
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {image_path}")
        mask_array = np.zeros_like(image_array)  # empty mask
    else:
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)

        if mask_array.ndim == 3:
            mask_array = mask_array

    return image_array, mask_array


def overlay_image_with_mask(image, mask, alpha=0.4):
    """
    Overlay a multi-class mask on a grayscale image.

    Parameters:
    - image: 2D numpy array (grayscale image)
    - mask: 2D numpy array (segmentation mask with labels)
    - alpha: float between 0 and 1, transparency of the overlay

    Returns:
    - overlay: 3D RGB image with the mask overlaid
    """

    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    if mask.shape != image.shape:
        raise ValueError(f"Image and mask must have the same shape. Got image: {image.shape}, mask: {mask.shape}")

    # Normalize image to [0, 1]
    image = image.astype(np.float32)
    if image.max() > 0:
        image /= image.max()

    # Convert grayscale image to RGB (H, W) -> (H, W, 3)
    image_rgb = np.stack([image] * 3, axis=-1)  # shape: (H, W, 3)

    # Output overlay
    overlay = image_rgb.copy()

    # Label color map (add or change as needed)
    colormap = {
        1: [1, 0, 0],   # Endocardium (Red)
        2: [0, 1, 0],   # Myocardium (Green)
        3: [0, 0, 1],   # Epicardium (Blue)
    }

    for label, color in colormap.items():
        mask_area = (mask == label)
        for c in range(3):  # RGB channels
            overlay[..., c][mask_area] = (
                (1 - alpha) * image_rgb[..., c][mask_area] + alpha * color[c]
            )

    return overlay

def load_image_and_mask(image_path, frame_num=None):
    # Load image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # Determine if it's a sequence
    if image_array.ndim == 3:
        image_frame = image_array[frame_num]
    else:
        image_frame = image_array
        frame_num = 0  # Default index if 2D

    # Infer corresponding mask path
    mask_path = image_path.replace('.nii.gz', '_gt.nii.gz')

    if not os.path.exists(mask_path):
        print(f"⚠️ Warning: Mask not found for {image_path}")
        mask_frame = np.zeros_like(image_frame)
    else:
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)

        # Handle mask dimensions
        if mask_array.ndim == 3:
            mask_frame = mask_array[frame_num]
        else:
            mask_frame = mask_array

    return image_frame, mask_frame

def show_images_in_grid(images, titles=None, cols=3, figsize=(15, 10)):
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    # Mask Legend
    legend_elements = [
        Patch(facecolor='red', label='LVendo'),
        Patch(facecolor='green', label='LVepi'), 
        Patch(facecolor='blue', label='LA')
    ]

    plt.figlegend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            if titles:
                ax.set_title(titles[i], fontsize=9)
        else:
            ax.axis('off')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Build dataframe
    make_camus_dataframe()
    camus_df = pd.read_csv("./datasets/CAMUS/CAMUS_dataset.csv")

    # Get N non-mask images
    N = 9
    start_idx = np.random.choice(len(camus_df) - N + 1)
    camus_df = camus_df.iloc[start_idx:start_idx + N]
    selected_paths = camus_df['nifti_path'].to_list()

    # Load and overlay
    overlays = []
    titles = []

    for path in selected_paths:
        if "frame" in path:
            # Extract frame number from the path
            frame_num = path.split('_frame')[1].split('.')[0]
            original_path = path.replace(f'_frame{frame_num}', '')
            img, mask = load_image_and_mask(original_path, frame_num=int(frame_num))
            print(np.unique(mask))
        else:
            img, mask = load_image_and_mask(path)

        overlay = overlay_image_with_mask(img, mask)
        overlays.append(overlay)
        titles.append(os.path.basename(path))

    # Show grid
    show_images_in_grid(overlays, titles=titles, cols=3)
    print(len(camus_df))