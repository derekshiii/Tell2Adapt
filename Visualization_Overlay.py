import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合多进程
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy import ndimage
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

def create_colormap_for_labels(n_labels=15):
    """为15个标签创建独特的颜色"""
    colors = [
        [0, 0, 0],  # 0: 背景
        [0.8, 0.2, 0.2],  # 1: 深红
        [0.2, 0.8, 0.2],  # 2: 深绿
        [0.2, 0.2, 0.8],  # 3: 深蓝
        [0.8, 0.8, 0.2],  # 4: 深黄
        [0.8, 0.2, 0.8],  # 5: 品红
        [0.2, 0.8, 0.8],  # 6: 青色
        [1.0, 0.5, 0.0],  # 7: 橙色
        [0.5, 0.0, 0.8],  # 8: 紫色
        [0.0, 0.5, 0.0],  # 9: 深绿
        [1.0, 0.6, 0.8],  # 10: 粉色
        [0.6, 0.4, 0.2],  # 11: 棕色
        [0.5, 0.5, 0.5],  # 12: 灰色
        [1.0, 0.8, 0.0],  # 13: 金色
        [0.5, 0.9, 0.5],  # 14: 浅绿
        [1.0, 0.3, 0.0],  # 15: 橙红
    ]
    return colors

def normalize_image(img):
    """归一化图像到0-1范围"""
    img = img.astype(float)
    img_min, img_max = np.percentile(img, [1, 99])
    img = np.clip(img, img_min, img_max)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img

def get_label_boundaries(label_slice):
    """获取每个标签的边界"""
    boundaries = np.zeros_like(label_slice, dtype=bool)

    for label_id in np.unique(label_slice):
        if label_id == 0:
            continue
        mask = (label_slice == label_id)
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        boundary = mask & ~eroded
        boundaries |= boundary

    return boundaries

def get_label_centers(label_slice):
    """获取每个标签的中心点坐标"""
    centers = {}
    for label_id in np.unique(label_slice):
        if label_id == 0:
            continue
        mask = (label_slice == label_id)
        if np.any(mask):
            coords = np.argwhere(mask)
            center = coords.mean(axis=0)
            centers[label_id] = center
    return centers

def overlay_labels_on_slice(image_slice, label_slice, colors, alpha=0.4):
    """在单个切片上叠加半透明标签"""
    # 归一化图像
    img_norm = normalize_image(image_slice)

    # 创建RGB图像
    rgb_image = np.stack([img_norm] * 3, axis=-1)

    # 叠加每个标签（半透明）
    for label_id in range(1, len(colors)):
        mask = label_slice == label_id
        if np.any(mask):
            color = colors[label_id]
            for c in range(3):
                rgb_image[:, :, c][mask] = (1 - alpha) * rgb_image[:, :, c][mask] + alpha * color[c]

    return rgb_image

def save_single_slice(image_slice, label_slice, colors, output_path, alpha=0.4, dpi=600):
    """
    保存单个切片的可视化

    参数:
        image_slice: 图像切片
        label_slice: 标签切片
        colors: 颜色映射
        output_path: 输出路径
        alpha: 透明度 (0-1)
        dpi: 输出DPI
    """
    # 创建overlay
    overlay = overlay_labels_on_slice(image_slice, label_slice, colors, alpha=alpha)

    # 创建图形，不留边距
    fig, ax = plt.subplots(figsize=(10, 10))

    # 显示图像
    ax.imshow(overlay, origin='lower', interpolation='bilinear')

    # 获取边界
    boundaries = get_label_boundaries(label_slice)

    # 绘制黑色边界
    boundary_coords = np.argwhere(boundaries)
    if len(boundary_coords) > 0:
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0],
                  c='black', s=1, alpha=0.8)

    # 获取标签中心并添加编号（颜色与overlay一致）
    centers = get_label_centers(label_slice)
    for label_id, (y, x) in centers.items():
        if label_id < len(colors):
            # 使用标签对应的颜色
            text_color = colors[label_id]
            ax.text(x, y, str(label_id),
                   color=text_color, fontsize=12, weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='black', alpha=0.7, edgecolor='white', linewidth=1))

    ax.axis('off')

    # 移除所有边距
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    # 保存
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_single_slice(args):
    """
    处理单个切片的包装函数，用于多进程

    参数:
        args: (img_slice, label_slice, output_path, alpha, dpi)
    """
    img_slice, label_slice, output_path, alpha, dpi = args
    colors = create_colormap_for_labels(15)

    try:
        save_single_slice(img_slice, label_slice, colors, output_path, alpha=alpha, dpi=dpi)
        return True
    except Exception as e:
        tqdm.write(f"处理切片 {output_path} 时出错: {e}")
        return False

def visualize_all_slices(image_path, label_path, output_dir, alpha=0.4, dpi=600, n_workers=None):
    """
    为每个切片保存单独的可视化图像（多进程版本）

    参数:
        image_path: 原始影像路径
        label_path: 标签路径
        output_dir: 输出目录
        alpha: 透明度 (0-1，0为完全透明，1为完全不透明)
        dpi: 输出图像DPI
        n_workers: 使用的CPU核心数，None表示使用所有可用核心
    """
    # 加载数据
    tqdm.write(f"加载影像: {image_path}")
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()

    tqdm.write(f"加载标签: {label_path}")
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata().astype(int)

    tqdm.write(f"影像形状: {img_data.shape}")

    # 找到切片数最少的轴
    slice_dim = np.argmin(img_data.shape)
    total_slices = img_data.shape[slice_dim]
    tqdm.write(f"选择轴 {slice_dim} (共 {total_slices} 个切片)")

    # 创建输出目录
    base_name = Path(image_path).name.replace('.nii.gz', '').replace('.nii', '')
    case_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(case_output_dir, exist_ok=True)

    # 准备所有切片的参数
    slice_args = []
    for slice_idx in range(total_slices):
        # 提取切片
        if slice_dim == 0:
            img_slice = img_data[slice_idx, :, :]
            label_slice = label_data[slice_idx, :, :]
        elif slice_dim == 1:
            img_slice = img_data[:, slice_idx, :]
            label_slice = label_data[:, slice_idx, :]
        else:  # slice_dim == 2
            img_slice = img_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

        # 旋转90度使图像横向显示
        img_slice = np.rot90(img_slice, k=1)
        label_slice = np.rot90(label_slice, k=1)

        output_path = os.path.join(case_output_dir, f'slice_{slice_idx:04d}.png')
        slice_args.append((img_slice, label_slice, output_path, alpha, dpi))

    # 使用多进程处理
    if n_workers is None:
        n_workers = cpu_count()

    tqdm.write(f"使用 {n_workers} 个CPU核心处理 {total_slices} 个切片...")

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_slice, slice_args),
            total=total_slices,
            desc=f"处理 {base_name}",
            ncols=100
        ))

    success_count = sum(results)
    tqdm.write(f"完成! {success_count}/{total_slices} 个切片已保存至: {case_output_dir}")

    return success_count == total_slices

def batch_process(image_dir, label_dir, output_dir, alpha=0.4, dpi=600, n_workers=None):
    """
    批量处理目录下的所有NIfTI文件

    参数:
        image_dir: 原始影像目录
        label_dir: 标签目录
        output_dir: 输出目录
        alpha: 透明度 (0-1)
        dpi: 输出DPI
        n_workers: 使用的CPU核心数，None表示使用所有可用核心
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # 查找所有影像文件
    image_files = sorted(image_dir.glob('*_0000.nii.gz'))

    if not image_files:
        image_files = sorted(image_dir.glob('*.nii.gz'))

    tqdm.write(f"找到 {len(image_files)} 个影像文件")

    if n_workers is None:
        n_workers = cpu_count()

    tqdm.write(f"使用 {n_workers} 个CPU核心进行并行处理")

    success_count = 0
    failed_files = []

    for idx, img_path in enumerate(image_files, 1):
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"处理文件 {idx}/{len(image_files)}: {img_path.name}")
        tqdm.write(f"{'='*80}")

        # 构建对应的标签路径
        base_name = img_path.stem.replace('.nii', '').replace('_0000', '')
        label_path = label_dir / f"{base_name}.nii.gz"

        if not label_path.exists():
            tqdm.write(f"警告: 找不到标签文件 {label_path}, 跳过")
            failed_files.append(img_path.name)
            continue

        try:
            result = visualize_all_slices(
                str(img_path),
                str(label_path),
                output_dir,
                alpha=alpha,
                dpi=dpi,
                n_workers=n_workers
            )
            if result:
                success_count += 1
            else:
                failed_files.append(img_path.name)
        except Exception as e:
            tqdm.write(f"处理 {img_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(img_path.name)
            continue

    tqdm.write(f"\n{'='*80}")
    tqdm.write(f"全部完成!")
    tqdm.write(f"成功处理: {success_count}/{len(image_files)} 个文件")
    if failed_files:
        tqdm.write(f"失败的文件: {', '.join(failed_files)}")
    tqdm.write(f"{'='*80}")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    IMAGE_DIR = "path/to/images"  # 包含 *_0000.nii.gz 的目录
    LABEL_DIR = "path/to/labels"  # 包含标签 .nii.gz 的目录
    OUTPUT_DIR = "output_visualizations"

    # 批量处理 - 每个切片保存为单独的图像
    batch_process(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        alpha=0.5,  # 可以调整透明度：0.0(完全透明) 到 1.0(完全不透明)
        dpi=600,
        n_workers=None  # None表示使用所有可用CPU核心，或指定具体数字如 n_workers=8
    )

    # 或者处理单个文件的所有切片
    # visualize_all_slices(
    #     image_path="AMOS_CT_0001_0000.nii.gz",
    #     label_path="AMOS_CT_0001.nii.gz",
    #     output_dir="output",
    #     alpha=0.5,
    #     dpi=600,
    #     n_workers=8  # 使用8个CPU核心
    # )

# 使用示例
if __name__ == "__main__":
    # 设置路径
    IMAGE_DIR = "/path/to/nnUNet_raw/Dataset049_BraTS-T2w/imagesTs"  # 包含 *_0000.nii.gz 的目录
    LABEL_DIR = "/path/to/nnUNet_inference/FINETUNED_PRED_T2w_FROM_T1n-HE"  # 包含标签 .nii.gz 的目录
    OUTPUT_DIR = "/path/to/visualizations/BraTs-T2w_Overlay_Visualization"

    # 批量处理 - 每个切片保存为单独的图像
    batch_process(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        alpha=0.3,  # 可以调整透明度：0.0(完全透明) 到 1.0(完全不透明)
        dpi=600,
        n_workers=None  # None表示使用所有可用CPU核心，或指定具体数字如 n_workers=8
    )

    # 或者处理单个文件的所有切片
    # visualize_all_slices(
    #     image_path="AMOS_CT_0001_0000.nii.gz",
    #     label_path="AMOS_CT_0001.nii.gz",
    #     output_dir="output",
    #     alpha=0.5,
    #     dpi=600,
    #     n_workers=8  # 使用8个CPU核心
    # )
