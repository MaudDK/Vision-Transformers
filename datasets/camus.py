import os
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np

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
                            cfg_path = os.path.join(nifti_dir, subject, f"Info_{v}.cfg")
                            nifti_path = os.path.join(nifti_dir, subject, f"{subject}_{v}_{i}.nii.gz")
                            nifti_path_gt = os.path.join(nifti_dir, subject, f"{subject}_{v}_{i}_gt.nii.gz")
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
        1: [1, 0, 0],   # Red
        2: [0, 1, 0],   # Green
        3: [0, 0, 1],   # Blue
    }

    for label, color in colormap.items():
        mask_area = (mask == label)
        for c in range(3):  # RGB channels
            overlay[..., c][mask_area] = (
                (1 - alpha) * image_rgb[..., c][mask_area] + alpha * color[c]
            )

    return overlay

def load_image_and_mask(image_path):
    # Load image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # Determine if it's a sequence
    if image_array.ndim == 3:
        mid_frame = image_array.shape[0] // 2
        image_frame = image_array[mid_frame]
    else:
        image_frame = image_array
        mid_frame = 0  # Default index if 2D

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
            mask_frame = mask_array[mid_frame]
        else:
            mask_frame = mask_array

    return image_frame, mask_frame

def show_images_in_grid(images, titles=None, cols=3, figsize=(15, 10)):
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

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
    N = 6
    selected_paths = camus_df['nifti_path'].to_list()[:N]

    # Load and overlay
    overlays = []
    titles = []
    for path in selected_paths:
        img, msk = load_image_and_mask(path)
        overlay = overlay_image_with_mask(img, msk)
        overlays.append(overlay)
        titles.append(os.path.basename(path))

    # Show grid
    show_images_in_grid(overlays, titles=titles, cols=3)