import os
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm

def save_as_nifti(data, output_path, affine=None):
    """
    将numpy数组保存为NIfTI格式文件

    Args:
        data: numpy数组
        output_path: 输出路径
        affine: 仿射变换矩阵，如果为None则使用默认单位矩阵
    """
    if affine is None:
        # 创建默认的仿射变换矩阵
        affine = np.eye(4)

    # 数据类型转换处理
    original_dtype = data.dtype
    tqdm.write(f"原始数据类型: {original_dtype}")

    # 根据数据类型和数据范围选择合适的目标类型
    if data.dtype == np.int64:
        # 检查数据范围来选择合适的类型
        data_min, data_max = data.min(), data.max()
        tqdm.write(f"数据范围: {data_min} ~ {data_max}")

        if data_min >= 0 and data_max <= 255:
            # 如果是0-255范围，转为uint8（常用于分割标签）
            data = data.astype(np.uint8)
            tqdm.write(f"转换为 uint8")
        elif data_min >= -32768 and data_max <= 32767:
            # 如果在int16范围内，转为int16
            data = data.astype(np.int16)
            tqdm.write(f"转换为 int16")
        elif data_min >= 0 and data_max <= 65535:
            # 如果在uint16范围内，转为uint16
            data = data.astype(np.uint16)
            tqdm.write(f"转换为 uint16")
        else:
            # 否则转为int32
            data = data.astype(np.int32)
            tqdm.write(f"转换为 int32")

    elif data.dtype == np.float64:
        # float64转为float32以节省空间
        data = data.astype(np.float32)
        tqdm.write(f"转换为 float32")

    # 创建NIfTI图像对象
    nifti_img = nib.Nifti1Image(data, affine)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存文件
    nib.save(nifti_img, output_path)
    tqdm.write(f"已保存: {output_path} (数据类型: {data.dtype})")

def process_prediction_files(input_dir, output_dir):
    """
    处理预测文件，将segs键值保存为NIfTI格式

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有npz文件
    npz_files = list(input_path.glob("*.npz"))

    if not npz_files:
        tqdm.write(f"在目录 {input_dir} 中未找到npz文件")
        return

    tqdm.write(f"找到 {len(npz_files)} 个预测文件")

    for npz_file in tqdm(npz_files, desc="处理预测文件"):
        try:
            # 加载npz文件
            data = np.load(npz_file)

            # 检查是否存在segs键
            if 'segs' not in data:
                tqdm.write(f"警告: {npz_file.name} 中未找到'segs'键")
                continue

            segs = data['segs']
            tqdm.write(f"处理文件: {npz_file.name}, segs形状: {segs.shape}")

            # 生成输出文件名
            output_filename = npz_file.stem + ".nii.gz"
            output_file_path = output_path / output_filename

            # 保存为NIfTI格式
            save_as_nifti(segs, str(output_file_path))

        except Exception as e:
            tqdm.write(f"处理文件 {npz_file.name} 时出错: {str(e)}")

def process_image_files(input_dir, output_img_dir, output_gts_dir):
    """
    处理原始影像文件，将imgs和gts保存为NIfTI格式

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    input_path = Path(input_dir)
    output_img_dir = Path(output_img_dir)
    output_gts_dir = Path(output_gts_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_gts_dir.mkdir(parents=True, exist_ok=True)
    # 查找所有npz文件
    npz_files = list(input_path.glob("*.npz"))

    if not npz_files:
        tqdm.write(f"在目录 {input_dir} 中未找到npz文件")
        return

    tqdm.write(f"找到 {len(npz_files)} 个原始影像文件")

    for npz_file in tqdm(npz_files, desc="处理原始影像文件"):
        try:
            # 加载npz文件
            data = np.load(npz_file)

            # 处理imgs
            if 'imgs' in data:
                imgs = data['imgs']
                tqdm.write(f"处理文件: {npz_file.name}, imgs形状: {imgs.shape}")

                output_filename = npz_file.stem + "_0000.nii.gz"
                output_file_path = output_img_dir / output_filename
                save_as_nifti(imgs, str(output_file_path))
            else:
                tqdm.write(f"警告: {npz_file.name} 中未找到'imgs'键")

            # 处理gts
            if 'gts' in data:
                gts = data['gts']
                tqdm.write(f"处理文件: {npz_file.name}, gts形状: {gts.shape}")

                output_filename = npz_file.stem + ".nii.gz"
                output_file_path = output_gts_dir / output_filename
                save_as_nifti(gts, str(output_file_path))
            else:
                tqdm.write(f"警告: {npz_file.name} 中未找到'gts'键")

        except Exception as e:
            tqdm.write(f"处理文件 {npz_file.name} 时出错: {str(e)}")


if __name__ == "__main__":

    pred_input_dir = "/path/to/Inference_results/RIGA"
    pred_output_dir = "/path/to/nnUNet_raw/Dataset154_BioP-RIGA/labelsTr"
    process_prediction_files(pred_input_dir, pred_output_dir)

    # # 处理原始影像文件示例
    # img_input_dir = "/path/to/Datasets/ACDC_processed-2T-HE"
    # img_output_dir = "/path/to/nnUNet_raw/Dataset120_ACDC-2T-HE/imagesTr"
    # gts_output_dir = "/path/to/nnUNet_raw/Dataset120_ACDC-2T-HE/labelsTr"
    # process_image_files(img_input_dir, img_output_dir, gts_output_dir)
