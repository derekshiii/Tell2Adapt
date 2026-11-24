import numpy as np
import os
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def compute_dice(pred, gt):
    """
    计算Dice系数
    pred: 预测标签 (3D numpy array)
    gt: 真实标签 (3D numpy array)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = 2.0 * intersection / union
    return dice

def compute_surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    计算表面距离
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 如果预测或真实标签为空，返回None
    if not pred.any() and not gt.any():
        return []
    if not pred.any() or not gt.any():
        return None

    # 获取表面点（使用腐蚀方法）
    from scipy.ndimage import binary_erosion

    # 预测的表面
    pred_surface = pred & ~binary_erosion(pred, structure=np.ones((3, 3, 3)))
    # 真实标签的表面
    gt_surface = gt & ~binary_erosion(gt, structure=np.ones((3, 3, 3)))

    if not pred_surface.any() or not gt_surface.any():
        return None

    # 计算距离变换
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)

    # 获取表面点到另一个表面的距离
    distances_pred_to_gt = dt_gt[pred_surface]
    distances_gt_to_pred = dt_pred[gt_surface]

    # 合并所有距离
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])

    return all_distances

def compute_asd(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    计算平均表面距离 (Average Surface Distance)
    """
    distances = compute_surface_distances(pred, gt, spacing)

    if distances is None:
        return np.nan
    if len(distances) == 0:
        return 0.0

    return np.mean(distances)

def compute_metrics_per_label(pred, gt, label_value, spacing=(1.0, 1.0, 1.0)):
    """
    计算特定标签值的指标
    """
    pred_mask = (pred == label_value)
    gt_mask = (gt == label_value)

    dice = compute_dice(pred_mask, gt_mask)
    asd = compute_asd(pred_mask, gt_mask, spacing)

    return dice, asd

def process_single_file(args):
    """
    处理单个文件的函数（用于多进程）
    """
    pred_file, gt_dir, spacing = args
    filename = pred_file.name
    gt_file = gt_dir / filename

    if not gt_file.exists():
        print(f"警告: 找不到对应的标签文件 {filename}")
        return None

    try:
        # 加载数据
        pred_data = np.load(pred_file)
        gt_data = np.load(gt_file)

        pred = pred_data['segs'].astype(np.int32)
        gt = gt_data['gts'].astype(np.uint8)

        # 获取所有唯一的标签值（排除背景0）
        unique_labels = np.unique(gt)
        unique_labels = unique_labels[unique_labels > 0]

        # 计算每个标签的指标
        label_metrics = {}
        for label in unique_labels:
            dice, asd = compute_metrics_per_label(pred, gt, label, spacing)
            label_metrics[int(label)] = {'dice': dice, 'asd': asd}

        result = {
            'filename': filename,
            'shape': str(pred.shape),
            'label_metrics': label_metrics
        }

        print(f"已处理: {filename} - 标签数: {len(unique_labels)}")

        return result

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(pred_dir, gt_dir, output_csv='metrics_results.csv',
                     spacing=(1.0, 1.0, 1.0), n_processes=None):
    """
    批量计算Dice和ASD指标（多进程版本）

    参数:
    pred_dir: 预测文件目录
    gt_dir: 标签文件目录
    output_csv: 输出CSV文件名
    spacing: 体素间距 (z, y, x)，默认为(1.0, 1.0, 1.0)
    n_processes: 进程数，默认为CPU核心数
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    # 获取所有npz文件
    pred_files = sorted(pred_dir.glob('*.npz'))

    if n_processes is None:
        n_processes = cpu_count()

    print(f"找到 {len(pred_files)} 个预测文件")
    print(f"使用 {n_processes} 个进程进行计算")
    print("=" * 80)

    # 准备多进程参数
    args_list = [(pred_file, gt_dir, spacing) for pred_file in pred_files]

    # 多进程处理
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_file, args_list)

    # 过滤掉None结果
    results = [r for r in results if r is not None]

    if len(results) == 0:
        print("没有成功处理任何文件")
        return None

    # 收集所有标签
    all_labels = set()
    for result in results:
        all_labels.update(result['label_metrics'].keys())
    all_labels = sorted(all_labels)

    print(f"\n检测到的标签: {all_labels}")
    print("=" * 80)

    # 组织数据：每个文件一行
    rows = []
    for result in results:
        row = {'filename': result['filename'], 'shape': result['shape']}
        for label in all_labels:
            if label in result['label_metrics']:
                row[f'label_{label}_dice'] = result['label_metrics'][label]['dice']
                row[f'label_{label}_asd'] = result['label_metrics'][label]['asd']
            else:
                row[f'label_{label}_dice'] = np.nan
                row[f'label_{label}_asd'] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # 计算每个标签的统计信息
    print("\n各标签的统计信息:")
    print("=" * 80)

    summary_data = []
    for label in all_labels:
        dice_col = f'label_{label}_dice'
        asd_col = f'label_{label}_asd'

        dice_values = df[dice_col].dropna()
        asd_values = df[asd_col].dropna()

        # 过滤掉inf值
        asd_values = asd_values[~np.isinf(asd_values)]

        summary_data.append({
            'Label': label,
            'Dice_Mean': dice_values.mean() if len(dice_values) > 0 else np.nan,
            'Dice_Std': dice_values.std() if len(dice_values) > 0 else np.nan,
            'ASD_Mean': asd_values.mean() if len(asd_values) > 0 else np.nan,
            'ASD_Std': asd_values.std() if len(asd_values) > 0 else np.nan,
            'Count': len(dice_values)
        })

        print(f"标签 {label}:")
        print(f"  Dice: {dice_values.mean():.4f} ± {dice_values.std():.4f}")
        print(f"  ASD:  {asd_values.mean():.4f} ± {asd_values.std():.4f}")
        print(f"  样本数: {len(dice_values)}")
        print("-" * 80)

    summary_df = pd.DataFrame(summary_data)

    # 计算总体平均（所有标签）
    all_dice_cols = [col for col in df.columns if col.endswith('_dice')]
    all_asd_cols = [col for col in df.columns if col.endswith('_asd')]

    all_dice_values = df[all_dice_cols].values.flatten()
    all_dice_values = all_dice_values[~np.isnan(all_dice_values)]

    all_asd_values = df[all_asd_cols].values.flatten()
    all_asd_values = all_asd_values[~np.isnan(all_asd_values)]
    all_asd_values = all_asd_values[~np.isinf(all_asd_values)]

    print("=" * 80)
    print("总体统计 (所有标签):")
    print(f"  Dice: {all_dice_values.mean():.4f} ± {all_dice_values.std():.4f}")
    print(f"  ASD:  {all_asd_values.mean():.4f} ± {all_asd_values.std():.4f}")
    print("=" * 80)

    # 保存详细结果
    df.to_csv(output_csv, index=False)
    print(f"\n详细结果已保存到: {output_csv}")

    # 保存汇总结果
    summary_csv = output_csv.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"汇总结果已保存到: {summary_csv}")

    return df, summary_df

# 使用示例
if __name__ == "__main__":
    # 设置预测和标签目录
    pred_directory = "/path/to/Inference_results/AMOS_MR-CHAOS-32.17-correct"
    gt_directory = "/path/to/Datasets/AMOS_MR_processed"

    # 体素间距（如果知道的话）
    spacing = (1.0, 1.0, 1.0)

    # 计算指标（使用所有CPU核心）
    results_df, summary_df = calculate_metrics(
        pred_dir=pred_directory,
        gt_dir=gt_directory,
        output_csv=f'{pred_directory}/dice_asd_results.csv',
        spacing=spacing,
        n_processes=None  # None表示使用所有CPU核心
    )

    # 显示汇总表格
    if summary_df is not None:
        print("\n汇总结果:")
        print(summary_df.to_string(index=False))
