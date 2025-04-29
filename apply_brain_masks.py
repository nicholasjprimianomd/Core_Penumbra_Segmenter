import os
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse

def apply_brain_masks(image_dir, mask_dir, output_dir):
    """
    Apply brain segmentation masks to CT images.
    
    Args:
        image_dir: Directory containing CT images
        mask_dir: Directory containing brain segmentation masks
        output_dir: Directory to save masked images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all brain masks
    brain_masks = sorted(glob.glob(os.path.join(mask_dir, "*_brain.nii.gz")))
    
    print(f"Found {len(brain_masks)} brain masks to apply")
    
    # Process each mask
    for i, mask_path in enumerate(brain_masks):
        print(f"Processing mask {i+1} of {len(brain_masks)}: {os.path.basename(mask_path)}")
        
        # Get case ID from filename
        case_id = os.path.basename(mask_path).replace("_brain.nii.gz", "")
        
        # Find corresponding CT image
        image_path = os.path.join(image_dir, f"{case_id}_0000.nii.gz")
        
        if not os.path.exists(image_path):
            print(f"  Error: CT image not found at {image_path}")
            continue
        
        # Output path for masked image
        output_path = os.path.join(output_dir, f"{case_id}_brain_only.nii.gz")
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"  Skipping {case_id} (already processed)")
            continue
        
        try:
            # Load CT image and brain mask
            ct_image = sitk.ReadImage(image_path)
            brain_mask = sitk.ReadImage(mask_path)
            
            # Convert to numpy arrays
            ct_array = sitk.GetArrayFromImage(ct_image)
            mask_array = sitk.GetArrayFromImage(brain_mask)
            
            # Ensure mask is binary (should already be, but just in case)
            mask_array = (mask_array > 0).astype(np.float32)
            
            # Apply mask (set non-brain voxels to a background value, e.g., -1000 HU)
            background_value = -1000
            masked_array = ct_array.copy()
            masked_array[mask_array == 0] = background_value
            
            # Create a new image with the same metadata as the original
            masked_image = sitk.GetImageFromArray(masked_array)
            masked_image.CopyInformation(ct_image)
            
            # Save masked image
            sitk.WriteImage(masked_image, output_path)
            
            print(f"  Saved brain-only image to {output_path}")
            
        except Exception as e:
            print(f"  Error processing {case_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply brain segmentation masks to CT images")
    parser.add_argument("--image_dir", default="./external/nnUNet/nnUNet_raw/Dataset501_CPAISD/imagesTr", 
                        help="Directory containing CT images")
    parser.add_argument("--mask_dir", default="./segmentation_results/brains", 
                        help="Directory containing brain segmentation masks")
    parser.add_argument("--output_dir", default="./segmentation_results/brain_only_images", 
                        help="Directory to save masked images")
    
    args = parser.parse_args()
    
    apply_brain_masks(args.image_dir, args.mask_dir, args.output_dir) 