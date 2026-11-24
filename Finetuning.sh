#!/bin/bash

# ========================================
# nnU-Net 预训练和微调脚本
# ========================================

# -----------------
# 参数定义区域
# -----------------

# 数据集ID
FINETUNING_DATASET=178  # 微调数据集ID（目标数据集
PRETRAINING_DATASET=182  # 预训练数据集ID

# Plans标识符
FINETUNING_PLANS_IDENTIFIER="nnUNetResEncUNetPlans"  # 微调数据集的plans名称
PRETRAINING_PLANS_IDENTIFIER="nnUNetResEncUNetPlans_CVCDB2Kvasir"  # 预训练数据集的plans名称（建议自定义）
# 训练配置
CONFIG="2d"  # 配置类型: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
FOLD="all"  # 预训练使用 'all'（所有数据），微调可以使用 0,1,2,3,4 或 all

# 预训练checkpoint路径（用于微调时加载） 应是预训练的Dataset ID
PRETRAINED_CHECKPOINT="/path/to/nnUNet_trained_models/Dataset182_CVCDB/nnUNetTrainer__${PRETRAINING_PLANS_IDENTIFIER}__${CONFIG}/fold_all/checkpoint_final.pth"  # 需要替换为实际路径

# 执行模式选择
MODE="full"  # full: 完整流程, pretrain: 仅预训练, finetune: 仅微调

# 自动模式设置
AUTO_MODE=true  # true: 完全自动执行, false: 需要交互确认
AUTO_CHECKPOINT_PATH=""  # 自动模式下的checkpoint路径（留空则自动推断）

# -----------------
# 函数定义
# -----------------

# 打印分隔线
print_separator() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# 检查命令是否成功
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ 成功: $1"
    else
        echo "✗ 失败: $1"
        exit 1
    fi
}

# -----------------
# 预训练流程
# -----------------
run_pretraining() {
    print_separator "开始预训练流程"

    # 步骤1: 对微调数据集进行实验规划和预处理
    print_separator "步骤1: 规划和预处理微调数据集"
    nnUNetv2_plan_and_preprocess -d ${FINETUNING_DATASET} -c ${CONFIG} -pl ResEncUNetPlanner
    check_status "微调数据集规划和预处理"

    # 步骤2: 提取预训练数据集的fingerprint
    print_separator "步骤2: 提取预训练数据集fingerprint"
    nnUNetv2_extract_fingerprint -d ${PRETRAINING_DATASET}
    check_status "提取预训练数据集fingerprint"

    # 步骤3: 将plans从微调数据集转移到预训练数据集
    print_separator "步骤3: 转移plans配置"
    nnUNetv2_move_plans_between_datasets \
        -s ${FINETUNING_DATASET} \
        -t ${PRETRAINING_DATASET} \
        -sp ${FINETUNING_PLANS_IDENTIFIER} \
        -tp ${PRETRAINING_PLANS_IDENTIFIER}
    check_status "Plans转移"

    # 步骤4: 预处理预训练数据集
    print_separator "步骤4: 预处理预训练数据集"
    nnUNetv2_preprocess -d ${PRETRAINING_DATASET} -plans_name ${PRETRAINING_PLANS_IDENTIFIER} -c ${CONFIG}
    check_status "预训练数据集预处理"

    # 步骤5: 训练预训练模型
    print_separator "步骤5: 训练预训练模型"
    nnUNetv2_train ${PRETRAINING_DATASET} ${CONFIG} all -p ${PRETRAINING_PLANS_IDENTIFIER} #-num_gpus 2
    check_status "预训练模型训练"

    print_separator "预训练流程完成"
}

# 自动推断checkpoint路径
auto_infer_checkpoint() {
    # 构建可能的checkpoint路径
    local dataset_name=$(printf "Dataset%03d" ${PRETRAINING_DATASET})
    local possible_path="/path/to/nnUNet_trained_models/${dataset_name}_*/nnUNetTrainer__${PRETRAINING_PLANS_IDENTIFIER}__${CONFIG}/fold_all/checkpoint_final.pth"

    # 查找匹配的文件
    local found_checkpoint=$(find ${nnUNet_results:-./nnUNet_results} -path "*${dataset_name}*/nnUNetTrainer__${PRETRAINING_PLANS_IDENTIFIER}__${CONFIG}/fold_all/checkpoint_final.pth" 2>/dev/null | head -n 1)

    if [ -n "${found_checkpoint}" ] && [ -f "${found_checkpoint}" ]; then
        echo "${found_checkpoint}"
        return 0
    else
        return 1
    fi
}

# -----------------
# 微调流程
# -----------------
run_finetuning() {
    print_separator "开始微调流程"

    local checkpoint_to_use="${PRETRAINED_CHECKPOINT}"

    # 如果是自动模式且checkpoint路径为空或不存在，尝试自动推断
    if [ "${AUTO_MODE}" = true ]; then
        if [ -z "${checkpoint_to_use}" ] || [ ! -f "${checkpoint_to_use}" ]; then
            echo "尝试自动推断checkpoint路径..."
            checkpoint_to_use=$(auto_infer_checkpoint)
            if [ $? -eq 0 ]; then
                echo "✓ 找到checkpoint: ${checkpoint_to_use}"
                PRETRAINED_CHECKPOINT="${checkpoint_to_use}"
            else
                echo "✗ 无法自动找到checkpoint"
                echo "请手动设置PRETRAINED_CHECKPOINT变量"
                exit 1
            fi
        fi
    fi

    # 检查预训练权重是否存在
    if [ ! -f "${checkpoint_to_use}" ]; then
        echo "错误: 预训练checkpoint不存在: ${checkpoint_to_use}"
        echo "请确保预训练已完成，或更新PRETRAINED_CHECKPOINT路径"
        exit 1
    fi

    # 使用预训练权重进行微调训练
    print_separator "使用预训练权重进行微调"
    echo "使用checkpoint: ${checkpoint_to_use}"
    nnUNetv2_train ${FINETUNING_DATASET} ${CONFIG} ${FOLD} -pretrained_weights ${checkpoint_to_use} -p nnUNetResEncUNetPlans #-num_gpus 2
    check_status "微调训练"

    print_separator "微调流程完成"
}

# -----------------
# 主程序
# -----------------
main() {
    print_separator "nnU-Net 预训练和微调脚本"
    echo "模式: ${MODE}"
    echo "自动执行: ${AUTO_MODE}"
    echo "预训练数据集: ${PRETRAINING_DATASET}"
    echo "微调数据集: ${FINETUNING_DATASET}"
    echo "配置: ${CONFIG}"
    echo ""

    case ${MODE} in
        full)
            run_pretraining
            echo ""

            if [ "${AUTO_MODE}" = true ]; then
                # 自动模式：直接继续微调
                echo ">>> 自动模式：继续执行微调 <<<"
                sleep 2  # 给用户2秒查看输出
                run_finetuning
            else
                # 交互模式：询问用户
                echo "提示: 请更新PRETRAINED_CHECKPOINT变量为实际的checkpoint路径"
                echo "例如: nnUNet_results/DatasetXXX_YourDataset/nnUNetTrainer__PretrainPlans__${CONFIG}/fold_all/checkpoint_final.pth"
                echo ""
                read -p "预训练完成。是否继续微调? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    read -p "请输入预训练checkpoint完整路径（留空自动推断）: " user_checkpoint
                    if [ -n "${user_checkpoint}" ]; then
                        PRETRAINED_CHECKPOINT="${user_checkpoint}"
                    fi
                    run_finetuning
                else
                    echo "跳过微调步骤"
                fi
            fi
            ;;
        pretrain)
            run_pretraining
            ;;
        finetune)
            run_finetuning
            ;;
        *)
            echo "错误: 未知的执行模式 '${MODE}'"
            echo "可选模式: full, pretrain, finetune"
            exit 1
            ;;
    esac

    print_separator "所有任务完成"
}

# 执行主程序
main
