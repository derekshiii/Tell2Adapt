import os
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
import gc
import json
from collections import defaultdict
from utils import process_input, process_output, slice_nms
from tqdm import tqdm
import nibabel as nib
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import time

# ==================== 配置选项 ====================
# GPU配置
CUDA_DEVICES = ['0', '1']  # 使用的GPU设备号列表
NUM_WORKERS = len(CUDA_DEVICES)  # 工作进程数量，通常等于GPU数量

# 数据路径配置
INPUT_DIR = ""
OUTPUT_DIR = ""
CHECKPOINT_PATH = ""

# 保存选项
SAVE_NII = False           # 设置为True时保存nii格式的预测结果
SAVE_PROB = False          # 设置为True时保存概率分布
NII_OUTPUT_DIR = None      # NII输出目录，None表示使用默认子目录 "nii_predictions"
PROB_OUTPUT_DIR = None     # 概率分布输出目录，None表示使用默认子目录 "probability_maps"

# 模型推理配置
SLICE_BATCH_SIZE = 4       # 切片批次大小
INTERPOLATE_SIZE = 512     # 插值大小
NMS_IOU_THRESHOLD = 0.5    # NMS的IOU阈值
SCORE_THRESHOLD = 0.5      # 分数阈值
# ==================================================

def load_case(file_path):
    data = np.load(file_path, allow_pickle=True)
    image = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gt = data["gts"] if "gts" in data else None
    return image, text_prompts, gt

def merge_multiclass_masks(masks, ids):
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    class_mask = keep_masks.argmax(dim=0)

    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]

    return class_mask

def postprocess(model_outputs, object_existence, threshold=SCORE_THRESHOLD, do_nms=True):
    if do_nms and model_outputs.shape[0] > 1:
        return slice_nms(model_outputs.sigmoid(), object_existence.sigmoid(), 
                        iou_threshold=NMS_IOU_THRESHOLD, score_threshold=threshold)
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask

def compute_dice_coefficient(mask_gt, mask_pred):
    """计算Dice系数"""
    if torch.is_tensor(mask_gt):
        mask_gt = mask_gt.cpu().numpy()
    if torch.is_tensor(mask_pred):
        mask_pred = mask_pred.cpu().numpy()
    
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_dice_per_class(gt_mask, pred_mask, class_ids):
    """计算每个类别的Dice系数"""
    dice_scores = {}
    
    for class_id in class_ids:
        gt_class_mask = (gt_mask == class_id)
        pred_class_mask = (pred_mask == class_id)
        dice_score = compute_dice_coefficient(gt_class_mask, pred_class_mask)
        dice_scores[class_id] = dice_score
    
    return dice_scores

def save_as_nii(mask_array, output_path, affine=None):
    """将mask数组保存为NIfTI格式"""
    if torch.is_tensor(mask_array):
        mask_array = mask_array.cpu().numpy()
    
    if affine is None:
        affine = np.eye(4)
    
    mask_array = mask_array.astype(np.int16)
    nii_img = nib.Nifti1Image(mask_array, affine)
    nib.save(nii_img, output_path)

def save_probability_maps(prob_maps, output_path, class_ids):
    """
    保存概率分布图
    
    Args:
        prob_maps: torch.Tensor, shape (num_classes, D, H, W) 或 numpy array
        output_path: str, 输出文件路径
        class_ids: list, 类别ID列表
    """
    if torch.is_tensor(prob_maps):
        prob_maps = prob_maps.cpu().numpy()
    
    # 将概率图转换为float16以节省空间
    prob_maps = prob_maps.astype(np.float16)
    
    # 保存为压缩的npz文件
    np.savez_compressed(
        output_path,
        probability_maps=prob_maps,
        class_ids=np.array(class_ids)
    )

def worker_process(gpu_id, file_queue, result_queue, progress_queue, input_dir, output_dir, 
                   checkpoint_path, nii_dir, prob_dir):
    """工作进程函数，在指定GPU上处理文件"""
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda:0")  # 因为设置了CUDA_VISIBLE_DEVICES，所以使用cuda:0
    
    try:
        # 初始化模型
        GlobalHydra.instance().clear()
        hydra.initialize(config_path="./configs/model", job_name=f"prediction_gpu_{gpu_id}")
        cfg = compose(config_name="biomedparse_3D")
        model = hydra.utils.instantiate(cfg, _convert_="object")
        model.load_pretrained(checkpoint_path)

        model.to(device)
        model.eval()
        
        # 处理文件队列中的文件
        processed_count = 0
        while True:
            try:
                file = file_queue.get(timeout=5)  # 5秒超时
                if file is None:  # 结束信号
                    break
                    
                file_path = os.path.join(input_dir, file)
                
                # 处理单个文件
                result = process_single_file(
                    file_path, file, output_dir, nii_dir, prob_dir,
                    model, device, gpu_id
                )
                
                # 将结果放入结果队列
                if result:
                    result_queue.put(result)
                
                processed_count += 1
                
                # 更新进度
                progress_queue.put({'gpu_id': gpu_id, 'file': file, 'status': 'completed'})
                
                # 内存清理
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                progress_queue.put({'gpu_id': gpu_id, 'file': file, 'status': 'error', 'error': str(e)})
                continue
    
    except Exception as e:
        progress_queue.put({'gpu_id': gpu_id, 'status': 'worker_error', 'error': str(e)})

def process_single_file(file_path, filename, output_dir, nii_output_dir, 
                       prob_output_dir, model, device, gpu_id):
    """处理单个文件"""
    try:
        # 加载数据
        npz_data = np.load(file_path, allow_pickle=True)
        imgs = npz_data["imgs"]
        text_prompts = npz_data["text_prompts"].item()
        
        # 检查是否有ground truth
        has_gt = "gts" in npz_data
        gt_masks = npz_data["gts"] if has_gt else None

        ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
        ids.sort()
        text = "[SEP]".join([text_prompts[str(i)] for i in ids])
        
        # 预处理输入
        imgs, pad_width, padded_size, valid_axis = process_input(imgs, INTERPOLATE_SIZE)
        imgs = imgs.to(device).int()

        # 模型推理
        input_tensor = {
            "image": imgs.unsqueeze(0),
            "text": [text],
        }

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=SLICE_BATCH_SIZE)

        # 获取原始概率分布（在后处理之前）
        mask_logits = output["predictions"]["pred_gmasks"]
        object_existence = output["predictions"]["object_existence"]
        
        # 插值到指定大小
        mask_logits_resized = F.interpolate(
            mask_logits,
            size=(INTERPOLATE_SIZE, INTERPOLATE_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        
        # 保存概率分布（如果启用）
        if SAVE_PROB and prob_output_dir:
            # 计算概率图（sigmoid后的结果）
            prob_maps = mask_logits_resized.sigmoid() * object_existence.sigmoid().unsqueeze(-1).unsqueeze(-1)
            
            # 对每个类别分别还原到原始尺寸
            prob_maps_list = []
            for i in range(prob_maps.shape[0]):
                prob_map_single = process_output(prob_maps[i], pad_width, padded_size, valid_axis)
                prob_maps_list.append(prob_map_single)
            
            # 堆叠所有类别的概率图
            if torch.is_tensor(prob_maps_list[0]):
                prob_maps_original = torch.stack(prob_maps_list, dim=0)
            else:
                prob_maps_original = np.stack(prob_maps_list, axis=0)
            
            base_filename = os.path.splitext(filename)[0]
            prob_save_path = os.path.join(prob_output_dir, f"{base_filename}_prob.npz")
            
            try:
                save_probability_maps(prob_maps_original, prob_save_path, ids)
            except Exception as e:
                pass  # 静默失败

        # 后处理得到最终mask
        mask_preds = postprocess(mask_logits_resized, object_existence)
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        # 保存预测结果为npz格式
        save_path = os.path.join(output_dir, filename)
        np.savez_compressed(save_path, segs=mask_preds)

        # 如果启用SAVE_NII选项，同时保存为NII格式
        if SAVE_NII and nii_output_dir:
            base_filename = os.path.splitext(filename)[0]
            nii_save_path = os.path.join(nii_output_dir, f"{base_filename}_pred.nii.gz")
            
            try:
                save_as_nii(mask_preds, nii_save_path)
            except Exception as e:
                pass  # 静默失败

        # 计算Dice系数（如果有ground truth）
        if has_gt and gt_masks is not None:
            if torch.is_tensor(mask_preds):
                mask_preds_np = mask_preds.cpu().numpy()
            else:
                mask_preds_np = mask_preds
                
            dice_scores = compute_dice_per_class(gt_masks, mask_preds_np, ids)
            valid_dice_scores = [score for score in dice_scores.values() if not np.isnan(score)]
            mean_dice = np.mean(valid_dice_scores) if valid_dice_scores else np.NaN
            
            # 构建结果
            result = {
                'filename': filename,
                'mean_dice': mean_dice,
            }
            
            for class_id in ids:
                result[f'dice_class_{class_id}'] = dice_scores.get(class_id, np.NaN)
            
            return result
        else:
            return None
        
    except Exception as e:
        return None

def main():
    """主函数"""
    tqdm.write("=" * 60)
    tqdm.write("Starting Multi-GPU Inference")
    tqdm.write("=" * 60)
    tqdm.write(f"GPU Devices: {CUDA_DEVICES}")
    tqdm.write(f"Number of Workers: {NUM_WORKERS}")
    tqdm.write(f"Input Directory: {INPUT_DIR}")
    tqdm.write(f"Output Directory: {OUTPUT_DIR}")
    tqdm.write(f"Checkpoint: {CHECKPOINT_PATH}")
    tqdm.write(f"SAVE_NII: {SAVE_NII}")
    tqdm.write(f"SAVE_PROB: {SAVE_PROB}")
    tqdm.write("-" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置NII输出目录
    nii_dir = None
    if SAVE_NII:
        if NII_OUTPUT_DIR is None:
            nii_dir = os.path.join(OUTPUT_DIR, "nii_predictions")
        else:
            nii_dir = NII_OUTPUT_DIR
        os.makedirs(nii_dir, exist_ok=True)
        tqdm.write(f"NII Output Directory: {nii_dir}")
    
    # 设置概率分布输出目录
    prob_dir = None
    if SAVE_PROB:
        if PROB_OUTPUT_DIR is None:
            prob_dir = os.path.join(OUTPUT_DIR, "probability_maps")
        else:
            prob_dir = PROB_OUTPUT_DIR
        os.makedirs(prob_dir, exist_ok=True)
        tqdm.write(f"Probability Maps Output Directory: {prob_dir}")
    
    # 获取所有需要处理的文件
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]
    tqdm.write(f"\nFound {len(all_files)} files to process")
    
    if not all_files:
        tqdm.write("No .npz files found in input directory")
        return
    
    # 创建进程间通信的队列
    file_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()
    
    # 将文件放入队列
    for file in all_files:
        file_queue.put(file)
    
    # 添加结束信号（每个worker一个None）
    for _ in range(NUM_WORKERS):
        file_queue.put(None)
    
    # 启动工作进程
    processes = []
    for i, gpu_id in enumerate(CUDA_DEVICES):
        p = Process(
            target=worker_process,
            args=(gpu_id, file_queue, result_queue, progress_queue, INPUT_DIR, OUTPUT_DIR, 
                  CHECKPOINT_PATH, nii_dir, prob_dir),
            name=f"GPU-{gpu_id}-Worker"
        )
        p.start()
        processes.append(p)
    
    tqdm.write("\nStarting inference...\n")
    
    # 收集结果
    all_results = []
    class_dice_accumulator = defaultdict(list)
    
    # 使用tqdm显示进度
    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        completed = 0
        while completed < len(all_files):
            try:
                # 检查进度更新
                if not progress_queue.empty():
                    progress_info = progress_queue.get(timeout=1)
                    if progress_info.get('status') == 'completed':
                        pbar.update(1)
                        completed += 1
                        file = progress_info.get('file', 'Unknown')
                        gpu = progress_info.get('gpu_id', 'Unknown')
                        tqdm.write(f"[GPU {gpu}] Completed: {file}")
                    elif progress_info.get('status') == 'error':
                        pbar.update(1)
                        completed += 1
                        file = progress_info.get('file', 'Unknown')
                        gpu = progress_info.get('gpu_id', 'Unknown')
                        error = progress_info.get('error', 'Unknown error')
                        tqdm.write(f"[GPU {gpu}] Error processing {file}: {error}")
                
                # 检查结果
                if not result_queue.empty():
                    result = result_queue.get_nowait()
                    if result:
                        all_results.append(result)
                        
                        # 累积类别Dice分数
                        for key, value in result.items():
                            if key.startswith('dice_class_'):
                                class_id = int(key.split('_')[-1])
                                class_dice_accumulator[class_id].append(value)
                
                time.sleep(0.1)
                
            except Exception as e:
                continue
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集剩余结果
    while not result_queue.empty():
        result = result_queue.get_nowait()
        if result and result not in all_results:
            all_results.append(result)
            for key, value in result.items():
                if key.startswith('dice_class_'):
                    class_id = int(key.split('_')[-1])
                    class_dice_accumulator[class_id].append(value)
    
    tqdm.write("\n" + "=" * 60)
    tqdm.write("Inference Complete - Computing Statistics")
    tqdm.write("=" * 60)
    
    # 计算统计信息并保存为JSON
    if all_results:
        # 准备统计结果
        statistics = {
            'overall': {},
            'per_class': {},
            'per_sample': all_results,
            'config': {
                'num_samples': len(all_results),
                'cuda_devices': CUDA_DEVICES,
                'checkpoint': CHECKPOINT_PATH,
                'save_nii': SAVE_NII,
                'save_prob': SAVE_PROB
            }
        }
        
        # 计算总体统计
        all_mean_scores = [score for score in [r['mean_dice'] for r in all_results] if not np.isnan(score)]
        if all_mean_scores:
            statistics['overall']['mean'] = float(np.mean(all_mean_scores))
            statistics['overall']['std'] = float(np.std(all_mean_scores))
            statistics['overall']['min'] = float(np.min(all_mean_scores))
            statistics['overall']['max'] = float(np.max(all_mean_scores))
            statistics['overall']['median'] = float(np.median(all_mean_scores))
        else:
            statistics['overall'] = {'mean': None, 'std': None, 'min': None, 'max': None, 'median': None}
        
        # 计算每个类别的统计
        for class_id in sorted(class_dice_accumulator.keys()):
            class_scores = [score for score in class_dice_accumulator[class_id] if not np.isnan(score)]
            if class_scores:
                statistics['per_class'][f'class_{class_id}'] = {
                    'mean': float(np.mean(class_scores)),
                    'std': float(np.std(class_scores)),
                    'min': float(np.min(class_scores)),
                    'max': float(np.max(class_scores)),
                    'median': float(np.median(class_scores))
                }
            else:
                statistics['per_class'][f'class_{class_id}'] = {
                    'mean': None, 'std': None, 'min': None, 'max': None, 'median': None
                }
        
        # 保存JSON文件
        json_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        tqdm.write(f"\nResults saved to: {json_path}")
        tqdm.write(f"Processed {len(all_results)} files with {NUM_WORKERS} GPUs")
        
        if SAVE_NII:
            tqdm.write(f"NII predictions saved to: {nii_dir}")
        
        if SAVE_PROB:
            tqdm.write(f"Probability maps saved to: {prob_dir}")
        
        # 打印统计信息
        tqdm.write("\n" + "-" * 60)
        tqdm.write("Overall Statistics:")
        tqdm.write("-" * 60)
        if statistics['overall']['mean'] is not None:
            tqdm.write(f"  Mean Dice:   {statistics['overall']['mean']:.4f} ± {statistics['overall']['std']:.4f}")
            tqdm.write(f"  Median Dice: {statistics['overall']['median']:.4f}")
            tqdm.write(f"  Min Dice:    {statistics['overall']['min']:.4f}")
            tqdm.write(f"  Max Dice:    {statistics['overall']['max']:.4f}")
        else:
            tqdm.write("  No valid Dice scores")
        
        tqdm.write("\n" + "-" * 60)
        tqdm.write("Per-Class Statistics:")
        tqdm.write("-" * 60)
        for class_name, stats in sorted(statistics['per_class'].items()):
            class_id = class_name.split('_')[1]
            if stats['mean'] is not None:
                tqdm.write(f"  Class {class_id}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"(median: {stats['median']:.4f}, range: [{stats['min']:.4f}, {stats['max']:.4f}])")
            else:
                tqdm.write(f"  Class {class_id}: No valid scores")
    else:
        tqdm.write("\nNo results to save - no files with ground truth were found.")
    
    tqdm.write("\n" + "=" * 60)
    tqdm.write("All Done!")
    tqdm.write("=" * 60)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()