import numpy as np
import os
import glob

def check_masks(path_pattern, limit=20):
    non_empty = 0
    total_checked = 0
    
    for mask_file in glob.glob(path_pattern)[:limit]:
        total_checked += 1
        data = np.load(mask_file)
        mask = data['mask']
        unique = np.unique(mask)
        if len(unique) > 1:
            non_empty += 1
            print(f'{mask_file}: Unique values: {unique}')
        
    print(f'Found {non_empty} non-empty masks out of {total_checked} checked')

if __name__ == "__main__":
    # Check multiple cases
    base_dir = '/mnt/c/Users/nprim/Downloads/10892316/dataset/dataset/train'
    cases = glob.glob(f'{base_dir}/*')[:5]  # Check first 5 cases
    
    for case in cases:
        case_id = os.path.basename(case)
        print(f"\nChecking case: {case_id}")
        check_masks(f'{case}/*/mask.npz', 20)
        
    # Also check a few validation cases
    val_dir = '/mnt/c/Users/nprim/Downloads/10892316/dataset/dataset/val'
    val_cases = glob.glob(f'{val_dir}/*')[:2]  # Check first 2 validation cases
    
    for case in val_cases:
        case_id = os.path.basename(case)
        print(f"\nChecking validation case: {case_id}")
        check_masks(f'{case}/*/mask.npz', 20) 