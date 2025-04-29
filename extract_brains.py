import os
import glob
import subprocess
from pathlib import Path
import argparse

def extract_brains(input_dir, output_dir, num_images=None):
    """
    Extract brain segmentations from CT images using TotalSegmentator.
    
    Args:
        input_dir: Directory containing CT images in .nii.gz format
        output_dir: Directory to save extracted brain segmentations
        num_images: Number of images to process (None for all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CT images
    ct_images = sorted(glob.glob(os.path.join(input_dir, "*_0000.nii.gz")))
    
    if num_images is not None:
        ct_images = ct_images[:num_images]
    
    print(f"Found {len(ct_images)} CT images to process")
    
    # Process each image
    for i, img_path in enumerate(ct_images):
        print(f"Processing image {i+1} of {len(ct_images)}: {os.path.basename(img_path)}")
        
        # Get case ID from filename
        case_id = os.path.basename(img_path).replace("_0000.nii.gz", "")
        
        # Output directory for this case
        case_output_dir = os.path.join(output_dir, case_id)
        
        # Skip if already processed
        if os.path.exists(os.path.join(case_output_dir, "brain.nii.gz")):
            print(f"  Skipping {case_id} (already processed)")
            continue
        
        # Run TotalSegmentator
        cmd = [
            "TotalSegmentator",
            "-i", img_path,
            "-o", case_output_dir,
            "--task", "total"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Copy just the brain segmentation to a separate directory for convenience
            brain_dir = os.path.join(output_dir, "brains")
            os.makedirs(brain_dir, exist_ok=True)
            
            brain_file = os.path.join(case_output_dir, "brain.nii.gz")
            if os.path.exists(brain_file):
                output_brain_file = os.path.join(brain_dir, f"{case_id}_brain.nii.gz")
                cmd_copy = ["cp", brain_file, output_brain_file]
                subprocess.run(cmd_copy, check=True)
                print(f"  Saved brain segmentation to {output_brain_file}")
            else:
                print(f"  Warning: Brain segmentation not found for {case_id}")
                
        except subprocess.CalledProcessError as e:
            print(f"  Error processing {case_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract brain segmentations from CT images using TotalSegmentator")
    parser.add_argument("--input_dir", default="./external/nnUNet/nnUNet_raw/Dataset501_CPAISD/imagesTr", 
                        help="Directory containing CT images in .nii.gz format")
    parser.add_argument("--output_dir", default="./segmentation_results", 
                        help="Directory to save extracted brain segmentations")
    parser.add_argument("--num_images", type=int, default=5, 
                        help="Number of images to process (default: 5, use 0 for all)")
    
    args = parser.parse_args()
    
    # If num_images is 0, process all images
    num_images = None if args.num_images == 0 else args.num_images
    
    extract_brains(args.input_dir, args.output_dir, num_images) 