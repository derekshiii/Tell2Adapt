#!/bin/bash

# nnUNet 双GPU并行预测脚本 - 8个任务版本
# 使用方法: bash this_script.sh

echo "开始 nnUNet 预测任务..."
echo "使用 GPU 0 和 GPU 1 进行并行预测"
echo "================================"

# # 任务5 - GPU 0
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
#     -i /path/to/nnUNet_raw/Dataset178_BiomedParse-Kvasir/imagesTs \
#     -o /path/to/nnUNet_inference/FINETUNED_PRED_Kvasir_FROM_CVCDB \
#     -d 178 \
#     -c 2d \
#     -f all \
#     -p nnUNetResEncUNetPlans &
# PID5=$!

# # 任务6 - GPU 1
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
#     -i /path/to/nnUNet_raw/Dataset179_BiomedParse-CVCDB/imagesTs \
#     -o /path/to/nnUNet_inference/FINETUNED_PRED_CVCDB_FROM_Kvasir \
#     -d 179 \
#     -c 2d \
#     -f all \
#     -p nnUNetResEncUNetPlans &
# PID6=$!

# echo "任务 5 和 6 已启动"
# wait $PID5 $PID6
# echo "任务 5 和 6 完成"

# 任务7 - GPU 0
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -i /path/to/nnUNet_raw/Dataset118_ACDC-2T/imagesTs \
    -o /path/to/nnUNet_inference/BASELINE_PRED_ACDC_FROM_CAMUS \
    -d 27 \
    -c 3d_fullres \
    -f all \
    -p nnUNetResEncUNetPlans &
PID7=$!

# 任务8 - GPU 1
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
    -i /path/to/nnUNet_raw/Dataset027_Cardiac-US-2T/imagesTs \
    -o /path/to/nnUNet_inference/BASELINE_PRED_CAMUS_FROM_ACDC \
    -d 118 \
    -c 3d_fullres \
    -f all \
    -p nnUNetResEncUNetPlans &
PID8=$!

# echo "任务 7 和 8 已启动"
# wait $PID7 $PID8
# echo "任务 7 和 8 完成"

echo "================================"
echo "所有 8 个预测任务完成!"
echo "完成时间: $(date)"
