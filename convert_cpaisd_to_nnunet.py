import os
import json
import shutil
from pathlib import Path
from collections import OrderedDict

def convert_to_nnunet_format(source_dir_str, target_nnunet_raw_dir_str, dataset_id, dataset_name, label_map, channel_names):
    """
    Converts a dataset from a FLARE-like format to the nnU-Net V2 format.

    Args:
        source_dir_str (str): Path to the source dataset directory (e.g., '/mnt/d/MedSAM/data/CPAISD_FLARE_format').
        target_nnunet_raw_dir_str (str): Path to the base nnU-Net raw data directory (e.g., '/mnt/d/Core_Penumbra_Segmenter/external/nnUNet/nnUNet_raw').
        dataset_id (int): The target nnU-Net dataset ID (e.g., 501).
        dataset_name (str): The target nnU-Net dataset name (e.g., 'CPAISD').
        label_map (dict): Dictionary mapping label names to integer values (e.g., {"background": 0, "Class1": 1, "Class2": 2}).
        channel_names (dict): Dictionary mapping channel index (as str) to channel name (e.g., {"0": "CT"}).
    """
    source_dir = Path(source_dir_str)
    target_nnunet_raw_dir = Path(target_nnunet_raw_dir_str)
    dataset_str = f"Dataset{dataset_id:03d}_{dataset_name}"
    target_dataset_dir = target_nnunet_raw_dir / dataset_str

    source_images_dir = source_dir / "images"
    source_labels_dir = source_dir / "labels"

    target_imagesTr_dir = target_dataset_dir / "imagesTr"
    target_labelsTr_dir = target_dataset_dir / "labelsTr"

    # --- Create target directories ---
    print(f"Creating target directories...")
    target_imagesTr_dir.mkdir(parents=True, exist_ok=True)
    target_labelsTr_dir.mkdir(parents=True, exist_ok=True)

    num_training_cases = 0
    file_ending = ".nii.gz"  # We know it's .nii.gz from the listing

    # --- First get all available label files since these determine valid cases ---
    print(f"Finding available label files...")
    label_files = sorted(list(source_labels_dir.glob('*.nii.gz')))
    
    if not label_files:
        print(f"Error: No label files found in {source_labels_dir}")
        return
    
    # Process each label file and find its corresponding image
    print(f"Processing {len(label_files)} label files...")
    
    for label_file in label_files:
        # Get case identifier from label file (without extension)
        case_identifier = label_file.name.replace('.nii.gz', '')
        
        # Find the corresponding image file (with _0000 suffix)
        image_filename = f"{case_identifier}_0000.nii.gz"
        source_image_file = source_images_dir / image_filename
        
        if not source_image_file.exists():
            print(f"Warning: Image file not found for label {label_file.name}. Expected: {image_filename}. Skipping case.")
            continue
        
        # Construct target paths (in nnUNet format)
        target_image_filename = f"{case_identifier}_0000{file_ending}"  # CASE_ID_XXXX.ext
        target_label_filename = f"{case_identifier}{file_ending}"  # CASE_ID.ext
        
        target_image_file = target_imagesTr_dir / target_image_filename
        target_label_file = target_labelsTr_dir / target_label_filename
        
        # Copy files
        print(f"Copying {source_image_file.name} -> {target_image_file.name}")
        shutil.copyfile(source_image_file, target_image_file)
        
        print(f"Copying {label_file.name} -> {target_label_file.name}")
        shutil.copyfile(label_file, target_label_file)
        
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
    SOURCE_DATA_DIR = "/mnt/d/MedSAM/data/CPAISD_FLARE_format"
    NNUNET_RAW_DIR = "/mnt/d/Core_Penumbra_Segmenter/external/nnUNet/nnUNet_raw" # nnU-Net expects this subfolder
    DATASET_ID = 501
    DATASET_NAME = "CPAISD"

    # Define labels (background must be 0, others consecutive integers)
    # IMPORTANT: Update "Class1", "Class2" with actual semantic names if known!
    LABEL_MAP = OrderedDict({
        "background": 0,
        "Class1": 1,
        "Class2": 2
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