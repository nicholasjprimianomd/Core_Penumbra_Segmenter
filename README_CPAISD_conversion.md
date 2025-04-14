# CPAISD to FLARE Format Conversion

This directory contains scripts to convert the CPAISD dataset to a format compatible with MedSAM preprocessing.

## Dataset Structure

The original CPAISD dataset has the following structure:
```
CPAISD_dataset/
├── train/
│   ├── [patient_id]/
│   │   ├── [slice_number]/
│   │   │   ├── image.npz
│   │   │   ├── mask.npz
│   │   │   ├── raw.dcm
│   │   │   └── metadata.json
│   │   ├── ...
│   │   └── metadata.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

The target FLARE format has this structure:
```
CPAISD_FLARE_format/
├── images/
│   ├── CPAISD_TRAIN_[patient_id]_0000.nii.gz
│   └── ...
└── labels/
    ├── CPAISD_TRAIN_[patient_id].nii.gz
    └── ...
```

## Conversion Process

The conversion process involves:

1. First analyzing both datasets to understand their structure
2. Converting each patient's 2D slices into a single 3D NIFTI volume
3. Maintaining train/val/test splits from the original dataset
4. Preserving metadata like spacing from the DICOM files if available

## Scripts

- `convert_cpaisd_to_flare.py`: Main conversion script
- `pre_CT_MR_CPAISD.py`: Modified version of the original `pre_CT_MR.py` optimized for head CT scans

## Usage

1. First run the conversion script to transform the dataset:

```
python convert_cpaisd_to_flare.py
```

This will:
- Analyze both datasets to verify compatibility
- Create a new directory structure at `external/MedSAM/data/CPAISD_FLARE_format`
- Convert all patients to the required format

2. After conversion, run the preprocessing script:

```
python pre_CT_MR_CPAISD.py
```

This will:
- Process the converted dataset using parameters appropriate for head CT
- Apply window level/width settings for brain CT (40/80)
- Output processed files to `external/MedSAM/data/npy/CT_Head`
- Maintain the train/validation split (80/20)

## Customization

The preprocessing script is customized for head CT with appropriate window settings:
- `WINDOW_LEVEL = 40` (brain window)
- `WINDOW_WIDTH = 80` (narrower than abdominal window)

## Requirements

- Python 3.7+
- NumPy
- PyDICOM
- SimpleITK
- Connected-Components-3D (cc3d)
- scikit-image
- tqdm

Install requirements with:
```
pip install numpy pydicom SimpleITK connected-components-3d scikit-image tqdm
``` 