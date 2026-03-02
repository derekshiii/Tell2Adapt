import os
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm

def save_as_nifti(data, output_path, affine=None):
    if affine is None:
        affine = np.eye(4)
    
    original_dtype = data.dtype
    tqdm.write(f"Original data type: {original_dtype}")
    
    if data.dtype == np.int64:
        data_min, data_max = data.min(), data.max()
        tqdm.write(f"Data range: {data_min} ~ {data_max}")
        
        if data_min >= 0 and data_max <= 255:
            data = data.astype(np.uint8)
            tqdm.write(f"Converted to uint8")
        elif data_min >= -32768 and data_max <= 32767:
            data = data.astype(np.int16)
            tqdm.write(f"Converted to int16")
        elif data_min >= 0 and data_max <= 65535:
            data = data.astype(np.uint16)
            tqdm.write(f"Converted to uint16")
        else:
            data = data.astype(np.int32)
            tqdm.write(f"Converted to int32")
    
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
        tqdm.write(f"Converted to float32")
    
    nifti_img = nib.Nifti1Image(data, affine)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nib.save(nifti_img, output_path)
    tqdm.write(f"Saved: {output_path} (data type: {data.dtype})")

def process_prediction_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npz_files = list(input_path.glob("*.npz"))
    
    if not npz_files:
        tqdm.write(f"No npz files found in directory {input_dir}")
        return
    
    tqdm.write(f"Found {len(npz_files)} prediction files")
    
    for npz_file in tqdm(npz_files, desc="Processing prediction files"):
        try:
            data = np.load(npz_file)
            
            if 'segs' not in data:
                tqdm.write(f"Warning: 'segs' key not found in {npz_file.name}")
                continue
            
            segs = data['segs']
            tqdm.write(f"Processing file: {npz_file.name}, segs shape: {segs.shape}")
            
            output_filename = npz_file.stem + ".nii.gz"
            output_file_path = output_path / output_filename
            
            save_as_nifti(segs, str(output_file_path))
            
        except Exception as e:
            tqdm.write(f"Error processing file {npz_file.name}: {str(e)}")

def process_image_files(input_dir, output_img_dir, output_gts_dir):
    input_path = Path(input_dir)
    output_img_dir = Path(output_img_dir)
    output_gts_dir = Path(output_gts_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_gts_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = list(input_path.glob("*.npz"))
    
    if not npz_files:
        tqdm.write(f"No npz files found in directory {input_dir}")
        return
    
    tqdm.write(f"Found {len(npz_files)} original image files")
    
    for npz_file in tqdm(npz_files, desc="Processing original image files"):
        try:
            data = np.load(npz_file)
            
            if 'imgs' in data:
                imgs = data['imgs']
                tqdm.write(f"Processing file: {npz_file.name}, imgs shape: {imgs.shape}")
                
                output_filename = npz_file.stem + "_0000.nii.gz"
                output_file_path = output_img_dir / output_filename
                save_as_nifti(imgs, str(output_file_path))
            else:
                tqdm.write(f"Warning: 'imgs' key not found in {npz_file.name}")
            
            if 'gts' in data:
                gts = data['gts']
                tqdm.write(f"Processing file: {npz_file.name}, gts shape: {gts.shape}")
                
                output_filename = npz_file.stem + ".nii.gz"
                output_file_path = output_gts_dir / output_filename
                save_as_nifti(gts, str(output_file_path))
            else:
                tqdm.write(f"Warning: 'gts' key not found in {npz_file.name}")
                
        except Exception as e:
            tqdm.write(f"Error processing file {npz_file.name}: {str(e)}")


if __name__ == "__main__":
    
    pred_input_dir = ""
    pred_output_dir = ""
    process_prediction_files(pred_input_dir, pred_output_dir)
