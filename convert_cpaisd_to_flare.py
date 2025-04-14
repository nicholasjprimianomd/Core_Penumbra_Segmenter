import os
import json
import shutil
import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path

def analyze_datasets():
    """Analyze both datasets and report their structures"""
    print("Analyzing datasets...")
    
    # Source dataset structure (CPAISD)
    source_base = Path("external/MedSAM/data/CPAISD_dataset")
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = source_base / split
        if not split_dir.exists():
            print(f"Split directory {split_dir} not found!")
            continue
            
        patient_dirs = list(split_dir.glob("*"))
        patient_dirs = [d for d in patient_dirs if d.is_dir()]
        
        print(f"Split {split}: {len(patient_dirs)} patients")
        
        if len(patient_dirs) > 0:
            # Examine first patient
            sample_patient = patient_dirs[0]
            slice_dirs = [d for d in sample_patient.glob("*") if d.is_dir()]
            
            print(f"  Sample patient {sample_patient.name}: {len(slice_dirs)} slices")
            
            if len(slice_dirs) > 0:
                # Examine first slice
                sample_slice = slice_dirs[0]
                files = list(sample_slice.glob("*"))
                
                print(f"  Sample slice {sample_slice.name} files: {[f.name for f in files]}")
                
                # Examine image and mask format
                if (sample_slice / "image.npz").exists():
                    try:
                        image_data = np.load(sample_slice / "image.npz")
                        print(f"  Image file keys: {image_data.files}")
                        if 'image' in image_data:
                            img = image_data['image']
                            print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
                    except Exception as e:
                        print(f"  Error loading image.npz: {e}")
                
                if (sample_slice / "mask.npz").exists():
                    try:
                        mask_data = np.load(sample_slice / "mask.npz")
                        print(f"  Mask file keys: {mask_data.files}")
                        if 'mask' in mask_data:
                            mask = mask_data['mask']
                            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
                            print(f"  Unique mask values: {np.unique(mask)}")
                    except Exception as e:
                        print(f"  Error loading mask.npz: {e}")
                
                if (sample_slice / "raw.dcm").exists():
                    try:
                        dcm = pydicom.dcmread(sample_slice / "raw.dcm")
                        print(f"  DICOM image size: {dcm.Rows}x{dcm.Columns}")
                        print(f"  DICOM spacing: {dcm.PixelSpacing if hasattr(dcm, 'PixelSpacing') else 'N/A'}")
                        print(f"  DICOM slice thickness: {dcm.SliceThickness if hasattr(dcm, 'SliceThickness') else 'N/A'}")
                    except Exception as e:
                        print(f"  Error reading DICOM: {e}")
    
    # Target dataset structure (FLARE22)
    flare_base = Path("external/MedSAM/data/FLARE22Train")
    
    if not flare_base.exists():
        print(f"FLARE22 directory {flare_base} not found!")
        return
        
    images_dir = flare_base / "images"
    labels_dir = flare_base / "labels"
    
    if images_dir.exists():
        image_files = list(images_dir.glob("*.nii.gz"))
        if len(image_files) > 0:
            sample_image = image_files[0]
            try:
                img_sitk = sitk.ReadImage(str(sample_image))
                img_array = sitk.GetArrayFromImage(img_sitk)
                print(f"FLARE22 sample image: {sample_image.name}")
                print(f"  Size: {img_sitk.GetSize()}")
                print(f"  Shape: {img_array.shape}")
                print(f"  Spacing: {img_sitk.GetSpacing()}")
                print(f"  Origin: {img_sitk.GetOrigin()}")
                print(f"  Direction: {img_sitk.GetDirection()}")
            except Exception as e:
                print(f"Error reading FLARE22 image: {e}")
    
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.nii.gz"))
        if len(label_files) > 0:
            sample_label = label_files[0]
            try:
                label_sitk = sitk.ReadImage(str(sample_label))
                label_array = sitk.GetArrayFromImage(label_sitk)
                print(f"FLARE22 sample label: {sample_label.name}")
                print(f"  Size: {label_sitk.GetSize()}")
                print(f"  Shape: {label_array.shape}")
                print(f"  Unique values: {np.unique(label_array)}")
            except Exception as e:
                print(f"Error reading FLARE22 label: {e}")

def create_output_directories():
    """Create output directories for the converted dataset"""
    output_base = Path("external/MedSAM/data/CPAISD_FLARE_format")
    images_dir = output_base / "images"
    labels_dir = output_base / "labels"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    return output_base, images_dir, labels_dir

def get_patient_slice_data(patient_dir):
    """Get all slices from a patient directory and sort them by slice location"""
    slices = []
    
    # Get all numbered directories that contain slices
    slice_dirs = [d for d in patient_dir.glob("[0-9][0-9][0-9][0-9][0-9]") if d.is_dir()]
    
    for slice_dir in slice_dirs:
        # Check if this slice directory has the required files
        if not (slice_dir / "image.npz").exists() or not (slice_dir / "mask.npz").exists():
            continue
            
        # Read metadata to get slice location
        metadata_path = slice_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    slice_location = metadata.get("slice_location", 0)
            except:
                # If we can't read metadata or it doesn't have slice_location, use the directory name
                slice_location = float(slice_dir.name)
        else:
            # No metadata file, use directory name
            slice_location = float(slice_dir.name)
            
        slices.append((slice_dir, slice_location))
    
    # Sort slices by location (typically superior to inferior)
    slices.sort(key=lambda x: x[1])
    
    return slices

def convert_patient_to_nifti(patient_dir, output_image_path, output_label_path):
    """Convert a patient's slices to a single 3D NIfTI file for both image and label"""
    # Get all slices and sort them by location
    sorted_slices = get_patient_slice_data(patient_dir)
    
    if len(sorted_slices) == 0:
        print(f"No valid slices found for {patient_dir.name}")
        return False
    
    # Read the first slice to get dimensions
    first_slice_dir = sorted_slices[0][0]
    try:
        image_data = np.load(first_slice_dir / "image.npz")
        mask_data = np.load(first_slice_dir / "mask.npz")
        
        first_img = image_data['image'] if 'image' in image_data else None
        first_mask = mask_data['mask'] if 'mask' in mask_data else None
        
        if first_img is None or first_mask is None:
            print(f"Missing image or mask data in {first_slice_dir}")
            return False
            
        # Try to read DICOM to get spacing information
        spacing = [1.0, 1.0, 1.0]  # Default spacing if not available
        if (first_slice_dir / "raw.dcm").exists():
            try:
                dcm = pydicom.dcmread(first_slice_dir / "raw.dcm")
                if hasattr(dcm, 'PixelSpacing') and dcm.PixelSpacing:
                    spacing[0:2] = dcm.PixelSpacing
                if hasattr(dcm, 'SliceThickness') and dcm.SliceThickness:
                    spacing[2] = dcm.SliceThickness
            except:
                pass
    except Exception as e:
        print(f"Error reading first slice data: {e}")
        return False
    
    # Create empty volumes for images and masks
    num_slices = len(sorted_slices)
    volume_shape = (num_slices,) + first_img.shape
    
    # If the image is already 3D (has a channel dimension)
    if len(first_img.shape) == 3:
        # Assuming the last dimension is channels (typically RGB)
        image_volume = np.zeros((num_slices, first_img.shape[0], first_img.shape[1]), dtype=np.uint8)
        # Convert RGB to grayscale if needed
        if first_img.shape[2] == 3:
            use_rgb = False  # Convert to grayscale
        else:
            use_rgb = False  # Just use first channel
    else:
        image_volume = np.zeros((num_slices, first_img.shape[0], first_img.shape[1]), dtype=np.uint8)
        use_rgb = False
        
    mask_volume = np.zeros((num_slices, first_mask.shape[0], first_mask.shape[1]), dtype=np.uint8)
    
    # Fill the volumes
    for i, (slice_dir, _) in enumerate(sorted_slices):
        try:
            # Load image
            image_data = np.load(slice_dir / "image.npz")
            img = image_data['image'] if 'image' in image_data else np.zeros_like(first_img)
            
            # Convert RGB to grayscale if needed
            if use_rgb and len(img.shape) == 3 and img.shape[2] == 3:
                # Simple averaging for grayscale conversion
                img_gray = (img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114).astype(np.uint8)
                image_volume[i] = img_gray
            elif len(img.shape) == 3:
                # Just use first channel
                image_volume[i] = img[:,:,0]
            else:
                image_volume[i] = img
                
            # Load mask
            mask_data = np.load(slice_dir / "mask.npz")
            mask = mask_data['mask'] if 'mask' in mask_data else np.zeros_like(first_mask)
            mask_volume[i] = mask
            
        except Exception as e:
            print(f"Error processing slice {slice_dir}: {e}")
            # Fill with zeros if there's an error
            continue
    
    # Create SimpleITK images and save as NIfTI
    try:
        # Use floating point representation for CT images
        image_sitk = sitk.GetImageFromArray(image_volume.astype(np.float32))
        image_sitk.SetSpacing(spacing)
        
        mask_sitk = sitk.GetImageFromArray(mask_volume)
        mask_sitk.SetSpacing(spacing)
        
        # Save files
        sitk.WriteImage(image_sitk, str(output_image_path))
        sitk.WriteImage(mask_sitk, str(output_label_path))
        
        return True
    except Exception as e:
        print(f"Error saving NIfTI files: {e}")
        return False

def convert_cpaisd_to_flare():
    """Main function to convert the CPAISD dataset to FLARE format"""
    # Create output directories
    output_base, images_dir, labels_dir = create_output_directories()
    
    # Process each split
    source_base = Path("external/MedSAM/data/CPAISD_dataset")
    splits = ["train", "val", "test"]
    
    # Keep track of successful conversions for each split
    converted_counts = {split: 0 for split in splits}
    
    for split in splits:
        split_dir = source_base / split
        if not split_dir.exists():
            print(f"Split directory {split_dir} not found!")
            continue
            
        patient_dirs = list(split_dir.glob("*"))
        patient_dirs = [d for d in patient_dirs if d.is_dir()]
        
        print(f"Processing {split} split: {len(patient_dirs)} patients")
        
        for patient_dir in tqdm(patient_dirs, desc=f"Converting {split}"):
            # Create output filenames
            prefix = f"CPAISD_{split.upper()}"  # e.g., CPAISD_TRAIN
            patient_id = patient_dir.name.replace(".", "_")  # Replace dots in ID
            
            image_filename = f"{prefix}_{patient_id}_0000.nii.gz"
            label_filename = f"{prefix}_{patient_id}.nii.gz"
            
            output_image_path = images_dir / image_filename
            output_label_path = labels_dir / label_filename
            
            # Convert patient data to NIfTI
            success = convert_patient_to_nifti(patient_dir, output_image_path, output_label_path)
            
            if success:
                converted_counts[split] += 1
    
    # Print summary
    print("\nConversion complete!")
    for split in splits:
        print(f"{split}: {converted_counts[split]} patients converted")
    
    print(f"\nConverted dataset saved to: {output_base}")
    print("To use this dataset with pre_CT_MR.py, update the following variables:")
    print("nii_path = \"data/CPAISD_FLARE_format/images\"")
    print("gt_path = \"data/CPAISD_FLARE_format/labels\"")
    print("And consider setting the modality to 'CT' and anatomy to 'Head'")

if __name__ == "__main__":
    # First analyze both datasets
    analyze_datasets()
    
    # Confirm and proceed with conversion
    response = input("\nContinue with conversion? (y/n): ")
    if response.lower() == 'y':
        convert_cpaisd_to_flare()
    else:
        print("Conversion cancelled.") 