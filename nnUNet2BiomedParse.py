import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
from PIL import Image
from npz_utils import load_npz

def load_nii_gz(file_path):
    """
    加载.nii.gz文件并返回numpy数组
    """
    try:
        nii_img = nib.load(file_path)
        # 注意：nibabel加载的数据维度顺序通常是 (W, H, D)
        return nii_img.get_fdata()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_png(file_path):
    """
    加载.png文件并返回numpy数组
    对于2D图像返回(H, W)，保持与单slice一致
    """
    try:
        img = Image.open(file_path)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_image_file(file_path):
    """
    根据文件扩展名自动选择加载方法
    """
    suffix = file_path.suffix.lower()
    if suffix == '.gz' and file_path.name.endswith('.nii.gz'):
        return load_nii_gz(file_path)
    elif suffix == '.png':
        data = load_png(file_path)
        # 如果是2D图像，添加第三维度使其变为 (H, W, 1)
        if data is not None and data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        return data
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def find_matching_label_file(img_file, labels_dir):
    """
    查找与图像文件对应的标签文件
    支持 .nii.gz 和 .png 格式
    """
    # 处理 .nii.gz 格式
    if img_file.name.endswith('_0000.nii.gz'):
        base_name = img_file.name.replace("_0000.nii.gz", "")
        # 先尝试 .nii.gz
        label_file = labels_dir / f"{base_name}.nii.gz"
        if label_file.exists():
            return label_file
        # 再尝试 .png
        label_file = labels_dir / f"{base_name}.png"
        if label_file.exists():
            return label_file

    # 处理 .png 格式
    elif img_file.name.endswith('_0000.png'):
        base_name = img_file.name.replace("_0000.png", "")
        # 先尝试 .png
        label_file = labels_dir / f"{base_name}.png"
        if label_file.exists():
            return label_file
        # 再尝试 .nii.gz
        label_file = labels_dir / f"{base_name}.nii.gz"
        if label_file.exists():
            return label_file

    return None

def get_base_name(img_file):
    """
    从图像文件名获取基础名称
    """
    if img_file.name.endswith('_0000.nii.gz'):
        return img_file.name.replace("_0000.nii.gz", "")
    elif img_file.name.endswith('_0000.png'):
        return img_file.name.replace("_0000.png", "")
    else:
        return img_file.stem

def process_single_file(args):
    """
    处理单个文件的函数，用于多进程调用
    """
    img_file, labels_dir, output_path, text_prompts = args

    # 获取基础文件名
    base_name = get_base_name(img_file)

    # 查找对应的标签文件
    label_file = find_matching_label_file(img_file, labels_dir)

    if label_file is None or not label_file.exists():
        return f"Warning: Label file not found for {img_file.name}", False

    try:
        # 加载影像和标签数据
        img_data = load_image_file(img_file)
        label_data = load_image_file(label_file)

        if img_data is None or label_data is None:
            return f"Skipping {base_name} due to loading error", False

        # 确保维度匹配
        if img_data.shape != label_data.shape:
            return f"Skipping {base_name}: shape mismatch (img: {img_data.shape}, label: {label_data.shape})", False

        # 确保数据类型一致
        img_data = img_data.astype(np.uint8)
        label_data = label_data.astype(np.uint8)

        # ==================== 开始修改 ====================

        # 1. 将标签3置零
        # 使用布尔索引找到所有值为3的元素并赋值为0
        label_data[label_data == 3] = 0

        # 2. 调整维度顺序从 (W, H, D) 到 (D, W, H)
        # 原始轴: 0=W, 1=H, 2=D
        # 目标轴顺序: 2, 0, 1
        img_data = np.transpose(img_data, (0, 1, 2))
        label_data = np.transpose(label_data, (0, 1, 2))

        # ==================== 修改结束 ====================

        # 创建输出数据字典
        output_data = {
            'imgs': img_data,
            'gts': label_data,
            'text_prompts': text_prompts
        }

        # 保存为npz文件
        output_file = output_path / f"{base_name}.npz"
        np.savez_compressed(output_file, **output_data)

        return f"Successfully processed: {base_name}", True

    except Exception as e:
        return f"Error processing {base_name}: {str(e)}", False

def convert_nnunet_to_biomedparse(input_dir, output_dir, text_prompts, num_processes=None):
    """
    将nnUNet格式数据转换为biomedparse格式（多进程版本）
    支持 .nii.gz 和 .png 格式
    """
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

    # 查找所有支持的图像文件格式
    image_files = []
    image_files.extend(images_dir.glob("*_0000.nii.gz"))
    image_files.extend(images_dir.glob("*_0000.png"))

    if not image_files:
        print("No supported image files found in imagesTr directory")
        print("Looking for files with pattern: *_0000.nii.gz or *_0000.png")
        return

    print(f"Found {len(image_files)} image files")
    print(f"Using {num_processes} processes for conversion")

    # 统计文件格式
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
        # 打印错误信息以方便调试
        if not success:
            print(message)

    print(f"\nConversion completed! Processed {processed_count}/{len(image_files)} files.")
    print(f"Output directory: {output_path}")

def verify_conversion(npz_file_path):
    """
    验证转换后的npz文件
    """
    try:
        data = load_npz(npz_file_path)

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
            print(f"Text prompts: {data['text_prompts']}")

    except Exception as e:
        print(f"Error verifying {npz_file_path}: {e}")

# 使用示例
if __name__ == "__main__":
    # 设置参数
    input_directory = "/path/to/nnUNet_raw/Dataset151_RIGA"  # 包含imagesTr和labelsTr的目录
    output_directory = "/path/to/Datasets/RIGA_processed"

    # 设置标签提示字典
    text_prompts = {
        '1': 'Optic cup in retinal fundus',
        '2': 'Optic disc in retinal fundus',
        #'3': 'right ventricle', # 这个提示可以保留，即使我们在gts中已经将其置零
        'instance_label': 0
    }

    # 执行转换，使用所有可用核心
    convert_nnunet_to_biomedparse(input_directory, output_directory, text_prompts)

    # 验证第一个转换后的文件
    output_files = list(Path(output_directory).glob("*.npz"))
    if output_files:
        verify_conversion(output_files[0])
