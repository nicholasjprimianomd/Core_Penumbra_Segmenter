import os
import glob
import nibabel as nib
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm
import cv2
import shutil

# Base directory - updated to use the finetune-SAM dataset directory
base_dir = Path('/mnt/d/Core_Penumbra_Segmenter/external/finetune-SAM/datasets/CTP')
splits = ['train', 'test', 'val']

# Clean up old data
print("Cleaning up old data...")
for split in splits:
    # Clean up old PNG files and 3D files
    for subdir in ['images', 'masks']:
        dir_path = base_dir / split / subdir
        if dir_path.exists():
            for f in dir_path.glob('*.png'):
                os.remove(f)
            
print("Creating directories...")
# Create directories for 2D slices
for split in splits:
    os.makedirs(base_dir / split / 'images', exist_ok=True)
    os.makedirs(base_dir / split / 'masks', exist_ok=True)

def normalize_image(img):
    """Normalize image to 0-255 range"""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        normalized = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img, dtype=np.uint8)
    return normalized

def process_dataset(split):
    """Process images and masks for a given split"""
    csv_rows = []
    # Get all nifti image files in the split
    img_files = list(glob.glob(str(base_dir / split / 'images' / '*.nii.gz')))
    
    for img_path in tqdm(img_files, desc=f"Processing {split} set"):
        img_path = Path(img_path)
        filename = img_path.name
        
        # Determine mask path - labels are in the 'labels' directory
        mask_filename = filename
        # Remove the _0000 suffix if present
        if '_0000.nii.gz' in mask_filename:
            mask_filename = mask_filename.replace('_0000.nii.gz', '.nii.gz')
            
        mask_path = base_dir / split / 'labels' / mask_filename
        
        if not mask_path.exists():
            print(f"Warning: No mask found for {filename}")
            continue
        
        # Load image and mask volumes
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # Get base name without extension
        base_name = filename.split('.')[0]
        
        # Process each slice
        for z in range(img_data.shape[2]):
            # Skip slices with no mask
            mask_slice = mask_data[:, :, z]
            if np.sum(mask_slice) == 0:
                continue
            
            # Get the image slice
            img_slice = img_data[:, :, z]
            
            # Normalize and convert to 8-bit image
            img_slice_norm = normalize_image(img_slice)
            mask_slice_norm = (mask_slice > 0).astype(np.uint8) * 255
            
            # Save slices
            slice_img_filename = f"{base_name}_z{z:03d}.png"
            slice_mask_filename = f"{base_name}_z{z:03d}.png"
            
            slice_img_path = base_dir / split / 'images' / slice_img_filename
            slice_mask_path = base_dir / split / 'masks' / slice_mask_filename
            
            cv2.imwrite(str(slice_img_path), img_slice_norm)
            cv2.imwrite(str(slice_mask_path), mask_slice_norm)
            
            # Add to CSV rows with "CTP" in the path
            rel_img_path = f"CTP/{split}/images/{slice_img_filename}"
            rel_mask_path = f"CTP/{split}/masks/{slice_mask_filename}"
            
            csv_rows.append([rel_img_path, rel_mask_path])
    
    # Write CSV
    csv_path = base_dir / f"{split}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(csv_rows)
    
    print(f"Created {len(csv_rows)} image-mask pairs for {split}")
    return len(csv_rows)

# Process all splits
total_slices = 0
for split in splits:
    slice_count = process_dataset(split)
    total_slices += slice_count

print(f"Dataset preparation complete. Total slices: {total_slices}")
print(f"CSV files are available at: {base_dir}")
print(f"Image and mask slices are available in each split's images/ and masks/ directories") 