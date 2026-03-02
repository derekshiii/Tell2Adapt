import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
from PIL import Image

def load_nii_gz(file_path):
    try:
        nii_img = nib.load(file_path)
        return nii_img.get_fdata()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_png(file_path):
    try:
        img = Image.open(file_path)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_image_file(file_path):
    suffix = file_path.suffix.lower()
    if suffix == '.gz' and file_path.name.endswith('.nii.gz'):
        return load_nii_gz(file_path)
    elif suffix == '.png':
        data = load_png(file_path)
        if data is not None and data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        return data
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def find_matching_label_file(img_file, labels_dir):
    if img_file.name.endswith('_0000.nii.gz'):
        base_name = img_file.name.replace("_0000.nii.gz", "")
        label_file = labels_dir / f"{base_name}.nii.gz"
        if label_file.exists():
            return label_file
        label_file = labels_dir / f"{base_name}.png"
        if label_file.exists():
            return label_file
    
    elif img_file.name.endswith('_0000.png'):
        base_name = img_file.name.replace("_0000.png", "")
        label_file = labels_dir / f"{base_name}.png"
        if label_file.exists():
            return label_file
        label_file = labels_dir / f"{base_name}.nii.gz"
        if label_file.exists():
            return label_file
    
    return None

def get_base_name(img_file):
    if img_file.name.endswith('_0000.nii.gz'):
        return img_file.name.replace("_0000.nii.gz", "")
    elif img_file.name.endswith('_0000.png'):
        return img_file.name.replace("_0000.png", "")
    else:
        return img_file.stem

def process_single_file(args):
    img_file, labels_dir, output_path, text_prompts = args
    
    base_name = get_base_name(img_file)
    
    label_file = find_matching_label_file(img_file, labels_dir)
    
    if label_file is None or not label_file.exists():
        return f"Warning: Label file not found for {img_file.name}", False
    
    try:
        img_data = load_image_file(img_file)
        label_data = load_image_file(label_file)
        
        if img_data is None or label_data is None:
            return f"Skipping {base_name} due to loading error", False
        
        if img_data.shape != label_data.shape:
            return f"Skipping {base_name}: shape mismatch (img: {img_data.shape}, label: {label_data.shape})", False
        
        img_data = img_data.astype(np.uint8)
        label_data = label_data.astype(np.uint8)
        
        label_data[label_data == 3] = 0
        
        img_data = np.transpose(img_data, (0, 1, 2))
        label_data = np.transpose(label_data, (0, 1, 2))

        output_data = {
            'imgs': img_data,
            'gts': label_data,
            'text_prompts': text_prompts
        }
        
        output_file = output_path / f"{base_name}.npz"
        np.savez_compressed(output_file, **output_data)
        
        return f"Successfully processed: {base_name}", True
        
    except Exception as e:
        return f"Error processing {base_name}: {str(e)}", False

def convert_nnunet_to_biomedparse(input_dir, output_dir, text_prompts, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    
    images_dir = Path(input_dir) / "imagesTr"
    labels_dir = Path(input_dir) / "labelsTr"
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    image_files = []
    image_files.extend(images_dir.glob("*_0000.nii.gz"))
    image_files.extend(images_dir.glob("*_0000.png"))
    
    if not image_files:
        print("No supported image files found in imagesTr directory")
        print("Looking for files with pattern: *_0000.nii.gz or *_0000.png")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Using {num_processes} processes for conversion")
    
    nii_count = sum(1 for f in image_files if f.name.endswith('.nii.gz'))
    png_count = sum(1 for f in image_files if f.name.endswith('.png'))
    print(f"File format distribution: {nii_count} .nii.gz, {png_count} .png")
    
    process_args = [(img_file, labels_dir, output_path, text_prompts) for img_file in image_files]
    
    processed_count = 0
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_single_file, process_args), 
                            total=len(image_files), 
                            desc="Converting files"))
    
    for message, success in results:
        if success:
            processed_count += 1
        if not success:
            print(message)
    
    print(f"\nConversion completed! Processed {processed_count}/{len(image_files)} files.")
    print(f"Output directory: {output_path}")

def verify_conversion(npz_file_path):
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        
        print(f"\nVerification for: {npz_file_path}")
        print(f"Keys in file: {list(data.keys())}")
        
        if 'imgs' in data:
            print(f"Images shape: {data['imgs'].shape}")
            print(f"Images dtype: {data['imgs'].dtype}")
            print(f"Images range: [{data['imgs'].min():.2f}, {data['imgs'].max():.2f}]")
        
        if 'gts' in data:
            print(f"Labels shape: {data['gts'].shape}")
            print(f"Labels dtype: {data['gts'].dtype}")
            print(f"Unique label values: {np.unique(data['gts'])}")
        
        if 'text_prompts' in data:
            print(f"Text prompts: {data['text_prompts'].item()}")
        
        data.close()
        
    except Exception as e:
        print(f"Error verifying {npz_file_path}: {e}")

if __name__ == "__main__":
    input_directory = ""
    output_directory = ""
    
    text_prompts = {
        '1': '', 
        '2': '', 
        '3': '',
        'instance_label': 0
    }
    
    convert_nnunet_to_biomedparse(input_directory, output_directory, text_prompts)
    
    output_files = list(Path(output_directory).glob("*.npz"))
    if output_files:
        verify_conversion(output_files[0])