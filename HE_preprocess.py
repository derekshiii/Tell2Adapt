import os
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle

def load_npz(file_path):
    """加载npz文件"""
    return np.load(file_path, allow_pickle=True)

def save_npz(data_dict, file_path):
    """保存npz文件，保持所有原始键值"""
    np.savez(file_path, **data_dict)

def apply_histogram_equalization_3d_local(image_data):
    """
    对3D图像应用局部直方图均衡化（基于单个3D体的全局统计）
    输入: image_data shape (slices, height, width) - 3D灰度图像 (uint8格式)
    输出: 均衡化后的图像数据 (保持uint8格式)
    """
    if len(image_data.shape) != 3:
        raise ValueError(f"Expected 3D image shape (slices, height, width), got {image_data.shape}")

    if image_data.dtype != np.uint8:
        print(f"Warning: Expected uint8 data type, got {image_data.dtype}")
        image_data = image_data.astype(np.uint8)

    # 将3D数据展平为1D来计算全局直方图
    flat_data = image_data.flatten()

    # 计算累积分布函数（CDF）
    hist, bins = np.histogram(flat_data, bins=256, range=(0, 256))
    cdf = hist.cumsum()

    # 归一化CDF
    cdf_normalized = cdf * 255 / cdf[-1]

    # 创建查找表
    lut = np.uint8(cdf_normalized)

    # 应用查找表到整个3D数据
    equalized_image = lut[image_data]

    return equalized_image

def compute_global_histogram(file_list, sample_ratio=1.0):
    """
    计算所有文件的全局直方图

    Args:
        file_list: 文件路径列表
        sample_ratio: 采样比例，用于加速（1.0表示使用所有数据）

    Returns:
        全局累积直方图
    """
    print(f"\nComputing global histogram from {len(file_list)} files...")

    global_hist = np.zeros(256, dtype=np.int64)

    files_to_sample = file_list
    if sample_ratio < 1.0:
        n_samples = max(1, int(len(file_list) * sample_ratio))
        files_to_sample = np.random.choice(file_list, n_samples, replace=False)
        print(f"Sampling {len(files_to_sample)} files for histogram computation")

    for file_path in tqdm(files_to_sample, desc="Computing histogram"):
        try:
            data = load_npz(file_path)
            if 'imgs' not in data:
                continue

            image_data = data['imgs']
            if image_data.dtype != np.uint8:
                image_data = image_data.astype(np.uint8)

            # 计算当前文件的直方图并累加
            hist, _ = np.histogram(image_data.flatten(), bins=256, range=(0, 256))
            global_hist += hist

        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    return global_hist

def create_global_lut(global_hist):
    """
    根据全局直方图创建查找表（LUT）

    Args:
        global_hist: 全局直方图

    Returns:
        查找表（LUT）
    """
    # 计算累积分布函数
    cdf = global_hist.cumsum()

    # 归一化到0-255
    cdf_normalized = cdf * 255 / cdf[-1]

    # 创建查找表
    lut = np.uint8(cdf_normalized)

    return lut

def apply_global_equalization(image_data, lut):
    """
    使用全局查找表对3D图像进行均衡化

    Args:
        image_data: 3D图像数据 (slices, height, width)
        lut: 查找表

    Returns:
        均衡化后的图像
    """
    if image_data.dtype != np.uint8:
        image_data = image_data.astype(np.uint8)

    # 应用查找表
    equalized_image = lut[image_data]

    return equalized_image

def process_single_file_local(args):
    """
    处理单个npz文件（使用局部统计的直方图均衡化）
    """
    file_path, dest_dir = args

    try:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)

        if os.path.exists(dest_path):
            return f"Already exists: {file_name}"

        data = load_npz(file_path)

        if 'imgs' not in data:
            return f"Warning: {file_name} missing 'imgs' key, skipping..."

        image_data = data['imgs']

        if len(image_data.shape) != 3:
            return f"Warning: {file_name} has unexpected image shape {image_data.shape}, expected 3D, skipping..."

        # 使用局部统计进行均衡化
        equalized_image = apply_histogram_equalization_3d_local(image_data)

        new_data = {}
        for key in data.files:
            if key == 'imgs':
                new_data[key] = equalized_image
            else:
                new_data[key] = data[key]

        save_npz(new_data, dest_path)

        return f"Success: {file_name}"

    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"

def process_single_file_global(args):
    """
    处理单个npz文件（使用全局统计的直方图均衡化）
    """
    file_path, dest_dir, lut = args

    try:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)

        if os.path.exists(dest_path):
            return f"Already exists: {file_name}"

        data = load_npz(file_path)

        if 'imgs' not in data:
            return f"Warning: {file_name} missing 'imgs' key, skipping..."

        image_data = data['imgs']

        if len(image_data.shape) != 3:
            return f"Warning: {file_name} has unexpected image shape {image_data.shape}, expected 3D, skipping..."

        # 使用全局查找表进行均衡化
        equalized_image = apply_global_equalization(image_data, lut)

        new_data = {}
        for key in data.files:
            if key == 'imgs':
                new_data[key] = equalized_image
            else:
                new_data[key] = data[key]

        save_npz(new_data, dest_path)

        return f"Success: {file_name}"

    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"

def get_file_list(source_dir):
    """获取所有npz文件列表"""
    npz_files = []
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.npz'):
            npz_files.append(os.path.join(source_dir, file_name))
    return sorted(npz_files)

def process_files_multicore_global(source_dir, dest_dir, num_processes=None,
                                   process_ratio=1.0, histogram_sample_ratio=0.1,
                                   lut_cache_path=None):
    """
    使用全局直方图统计进行多核处理

    Args:
        source_dir: 源目录
        dest_dir: 目标目录
        num_processes: 进程数
        process_ratio: 处理文件的比例
        histogram_sample_ratio: 用于计算全局直方图的采样比例
        lut_cache_path: LUT缓存路径，如果提供则保存/加载LUT
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    all_files = get_file_list(source_dir)

    if not all_files:
        print(f"No .npz files found in {source_dir}")
        return

    num_files_to_process = int(len(all_files) * process_ratio)
    files_to_process = all_files[:num_files_to_process]

    print(f"Processing {process_ratio*100:.1f}% of files: {num_files_to_process} of {len(all_files)}")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")

    # 步骤1: 计算或加载全局LUT
    lut = None
    if lut_cache_path and os.path.exists(lut_cache_path):
        print(f"\nLoading cached LUT from {lut_cache_path}")
        with open(lut_cache_path, 'rb') as f:
            lut = pickle.load(f)
    else:
        # 计算全局直方图
        global_hist = compute_global_histogram(all_files, sample_ratio=histogram_sample_ratio)

        # 创建全局LUT
        print("\nCreating global lookup table...")
        lut = create_global_lut(global_hist)

        # 保存LUT
        if lut_cache_path:
            print(f"Saving LUT to {lut_cache_path}")
            with open(lut_cache_path, 'wb') as f:
                pickle.dump(lut, f)

    # 步骤2: 使用全局LUT处理文件
    args_list = [(file_path, dest_dir, lut) for file_path in files_to_process]

    if num_processes is None:
        num_processes = cpu_count()

    print(f"\nUsing {num_processes} processes to process {len(files_to_process)} files...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file_global, args_list),
            total=len(args_list),
            desc="Processing files",
            unit="file"
        ))

    # 统计结果
    success_count = sum(1 for result in results if result.startswith("Success"))
    error_count = sum(1 for result in results if result.startswith("Error"))
    warning_count = sum(1 for result in results if result.startswith("Warning"))
    exists_count = sum(1 for result in results if result.startswith("Already exists"))

    print(f"\nProcessing completed!")
    print(f"Success: {success_count}")
    print(f"Already exists: {exists_count}")
    print(f"Warnings: {warning_count}")
    print(f"Errors: {error_count}")

    if error_count > 0 or warning_count > 0:
        print("\nDetailed messages:")
        for result in results:
            if result.startswith("Error") or result.startswith("Warning"):
                print(result)

def process_files_multicore_local(source_dir, dest_dir, num_processes=None, process_ratio=1.0):
    """
    使用局部直方图统计进行多核处理（每个3D体独立统计）
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    all_files = get_file_list(source_dir)

    if not all_files:
        print(f"No .npz files found in {source_dir}")
        return

    num_files_to_process = int(len(all_files) * process_ratio)
    files_to_process = all_files[:num_files_to_process]

    print(f"Processing {process_ratio*100:.1f}% of files: {num_files_to_process} of {len(all_files)}")

    args_list = [(file_path, dest_dir) for file_path in files_to_process]

    if num_processes is None:
        num_processes = cpu_count()

    print(f"Using {num_processes} processes to process {len(files_to_process)} files...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file_local, args_list),
            total=len(args_list),
            desc="Processing files",
            unit="file"
        ))

    success_count = sum(1 for result in results if result.startswith("Success"))
    error_count = sum(1 for result in results if result.startswith("Error"))
    warning_count = sum(1 for result in results if result.startswith("Warning"))
    exists_count = sum(1 for result in results if result.startswith("Already exists"))

    print(f"\nProcessing completed!")
    print(f"Success: {success_count}")
    print(f"Already exists: {exists_count}")
    print(f"Warnings: {warning_count}")
    print(f"Errors: {error_count}")

def main():
    source_dir = "/path/to/Datasets/Cardiac_US_processed-2T"
    dest_dir = "/path/to/Datasets/Cardiac_US_processed-2T-HE"
    lut_cache_path = "/path/to/Datasets/Cardiac_US_processed-2T-HE/global_histogram_lut.pkl"

    print("Starting 3D medical image histogram equalization processing...")
    print("\n" + "="*60)
    print("选择处理模式:")
    print("1. 局部模式: 每个3D体独立计算直方图")
    print("2. 全局模式: 使用整个数据集的全局直方图")
    print("="*60)

    mode = '2'

    if mode == "1":
        print("\n使用局部模式处理...")
        process_files_multicore_local(source_dir, dest_dir, process_ratio=0.05)
    else:
        print("\n使用全局模式处理...")
        # 全局模式：先统计10%的文件来计算直方图，然后处理5%的文件
        process_files_multicore_global(
            source_dir,
            dest_dir,
            process_ratio=1,
            histogram_sample_ratio=1,  # 使用10%的文件计算全局直方图
            lut_cache_path=lut_cache_path  # 缓存LUT以便后续使用
        )

if __name__ == "__main__":
    main()
