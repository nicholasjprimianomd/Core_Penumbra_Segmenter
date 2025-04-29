import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from collections import OrderedDict
import glob
import pydicom

def convert_to_nnunet_format(source_dir_str, target_nnunet_raw_dir_str, dataset_id, dataset_name, label_map, channel_names):
    """
    Converts a dataset from CPAISD format to the nnU-Net V2 format.

    Args:
        source_dir_str (str): Path to the source dataset directory.
        target_nnunet_raw_dir_str (str): Path to the base nnU-Net raw data directory.
        dataset_id (int): The target nnU-Net dataset ID.
        dataset_name (str): The target nnU-Net dataset name.
        label_map (dict): Dictionary mapping label names to integer values.
        channel_names (dict): Dictionary mapping channel index (as str) to channel name.
    """
    source_dir = Path(source_dir_str)
    target_nnunet_raw_dir = Path(target_nnunet_raw_dir_str)
    dataset_str = f"Dataset{dataset_id:03d}_{dataset_name}"
    target_dataset_dir = target_nnunet_raw_dir / dataset_str

    # Directory structures
    train_dir = source_dir / "train"
    
    target_imagesTr_dir = target_dataset_dir / "imagesTr"
    target_labelsTr_dir = target_dataset_dir / "labelsTr"

    # --- Create target directories ---
    print(f"Creating target directories...")
    target_imagesTr_dir.mkdir(parents=True, exist_ok=True)
    target_labelsTr_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all case directories in the training set
    print(f"Finding available cases...")
    case_dirs = sorted(list(train_dir.glob('*')))
    
    if not case_dirs:
        print(f"Error: No case directories found in {train_dir}")
        return
    
    # Process each case
    num_training_cases = 0
    file_ending = ".nii.gz"
    
    print(f"Processing {len(case_dirs)} cases...")
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"Processing case {case_id}...")
        
        # Get all slice directories
        slice_dirs = sorted(list(case_dir.glob('[0-9]*')))
        
        if not slice_dirs:
            print(f"Warning: No slice directories found for case {case_id}. Skipping case.")
            continue
        
        # Read images and masks from each slice
        images = []
        masks = []
        z_positions = []
        
        for slice_dir in slice_dirs:
            # Get image data
            image_file = slice_dir / "image.npz"
            mask_file = slice_dir / "mask.npz"
            
            if not image_file.exists() or not mask_file.exists():
                print(f"Warning: Missing image or mask file in {slice_dir}. Skipping slice.")
                continue
            
            # Read DICOM file to get z-position for sorting
            dcm_file = slice_dir / "raw.dcm"
            if dcm_file.exists():
                try:
                    dcm = pydicom.dcmread(str(dcm_file))
                    z_pos = float(dcm.ImagePositionPatient[2])
                    z_positions.append((len(images), z_pos))
                except Exception as e:
                    print(f"Warning: Error reading DICOM file {dcm_file}: {e}")
                    z_positions.append((len(images), len(images)))  # Use index as fallback
            else:
                z_positions.append((len(images), len(images)))  # Use index as fallback
            
            # Load image data
            try:
                img_data = np.load(image_file)
                img = img_data['image']
                images.append(img)
                
                mask_data = np.load(mask_file)
                mask = mask_data['mask']
                masks.append(mask)
            except Exception as e:
                print(f"Warning: Error loading data from {slice_dir}: {e}")
                continue
        
        if not images:
            print(f"Warning: No valid slices found for case {case_id}. Skipping case.")
            continue
        
        # Sort slices by z-position
        z_positions.sort(key=lambda x: x[1])
        sorted_indices = [x[0] for x in z_positions]
        
        images = [images[i] for i in sorted_indices]
        masks = [masks[i] for i in sorted_indices]
        
        # Stack 2D slices to create 3D volumes
        image_3d = np.stack(images, axis=0)  # Shape: (Z, X, Y)
        mask_3d = np.stack(masks, axis=0)  # Shape: (Z, X, Y)
        
        # Transpose to nnUNet expected format: (X, Y, Z)
        image_3d = np.transpose(image_3d, (1, 2, 0))
        mask_3d = np.transpose(mask_3d, (1, 2, 0))
        
        # Create sitk images
        image_sitk = sitk.GetImageFromArray(image_3d)
        mask_sitk = sitk.GetImageFromArray(mask_3d)
        
        # Set spacing to isotropic 1mm if not available from DICOM
        image_sitk.SetSpacing([1.0, 1.0, 1.0])
        mask_sitk.SetSpacing([1.0, 1.0, 1.0])
        
        # Save as nifti files
        target_image_filename = f"{case_id}_0000{file_ending}"  # CASE_ID_0000.nii.gz
        target_label_filename = f"{case_id}{file_ending}"  # CASE_ID.nii.gz
        
        target_image_file = target_imagesTr_dir / target_image_filename
        target_label_file = target_labelsTr_dir / target_label_filename
        
        print(f"Saving {target_image_file.name}...")
        sitk.WriteImage(image_sitk, str(target_image_file))
        
        print(f"Saving {target_label_file.name}...")
        sitk.WriteImage(mask_sitk, str(target_label_file))
        
        num_training_cases += 1

    if num_training_cases == 0:
        print("Error: No training cases were successfully processed.")
        # Clean up potentially empty directories
        shutil.rmtree(target_dataset_dir)
        print(f"Removed empty target directory: {target_dataset_dir}")
        return

    # --- Generate dataset.json ---
    print(f"Generating dataset.json...")
    json_dict = OrderedDict()
    json_dict['channel_names'] = channel_names
    json_dict['labels'] = label_map
    json_dict['numTraining'] = num_training_cases
    json_dict['file_ending'] = file_ending

    json_object = json.dumps(json_dict, indent=4)

    dataset_json_path = target_dataset_dir / "dataset.json"
    with open(dataset_json_path, "w") as outfile:
        outfile.write(json_object)

    print("-" * 30)
    print(f"Successfully converted {num_training_cases} cases.")
    print(f"Dataset saved to: {target_dataset_dir}")
    print(f"Generated {dataset_json_path}")
    print("-" * 30)


if __name__ == "__main__":
    # --- Configuration ---
    SOURCE_DATA_DIR = "/mnt/c/Users/nprim/Downloads/10892316/dataset/dataset"
    NNUNET_RAW_DIR = "./external/nnUNet/nnUNet_raw" # nnU-Net expects this subfolder
    DATASET_ID = 501
    DATASET_NAME = "CPAISD"

    # Define labels (background must be 0, others consecutive integers)
    # Based on inspection, it seems masks are all zeros, but we'll define the 
    # label map according to what would be expected for segmentation
    LABEL_MAP = OrderedDict({
        "background": 0,
        "core": 1,
        "penumbra": 2
    })

    # Define channel names (modality)
    CHANNEL_NAMES = OrderedDict({
       "0": "CT" # Key is string index '0', value is name 'CT'
    })
    # --- End Configuration ---

    convert_to_nnunet_format(
        source_dir_str=SOURCE_DATA_DIR,
        target_nnunet_raw_dir_str=NNUNET_RAW_DIR,
        dataset_id=DATASET_ID,
        dataset_name=DATASET_NAME,
        label_map=LABEL_MAP,
        channel_names=CHANNEL_NAMES
    )