#!/bin/bash

# Dataset IDs
FINETUNING_DATASET=178  # Fine-tuning dataset ID (target dataset)
SOURCE_DATASET=182  # Source domain dataset ID (pretrained model)

# Plans identifiers
FINETUNING_PLANS_IDENTIFIER="nnUNetResEncUNetPlans"  # Plans name for fine-tuning dataset
SOURCE_PLANS_IDENTIFIER="nnUNetResEncUNetPlans_CVCDB2Kvasir"  # Plans name for source model

# Training configuration
CONFIG="2d"  # Configuration type: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
FOLD="all"  # Use 'all' or specific fold: 0,1,2,3,4

# Source model checkpoint path (already trained on source domain)
SOURCE_CHECKPOINT="/8TB_HDD_2/nnUNetFrame/nnUNet_trained_models/Dataset182_CVCDB/nnUNetTrainer__${SOURCE_PLANS_IDENTIFIER}__${CONFIG}/fold_all/checkpoint_final.pth"

# Automatic mode settings
AUTO_MODE=true  # true: fully automatic execution, false: requires interactive confirmation
AUTO_CHECKPOINT_PATH=""  # Checkpoint path in automatic mode (leave empty for auto-inference)

# -----------------
# Function Definitions
# -----------------

# Print separator
print_separator() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# Check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ Success: $1"
    else
        echo "✗ Failed: $1"
        exit 1
    fi
}

# Auto-infer checkpoint path
auto_infer_checkpoint() {
    local dataset_name=$(printf "Dataset%03d" ${SOURCE_DATASET})
    
    local found_checkpoint=$(find ${nnUNet_results:-./nnUNet_results} -path "*${dataset_name}*/nnUNetTrainer__${SOURCE_PLANS_IDENTIFIER}__${CONFIG}/fold_all/checkpoint_final.pth" 2>/dev/null | head -n 1)
    
    if [ -n "${found_checkpoint}" ] && [ -f "${found_checkpoint}" ]; then
        echo "${found_checkpoint}"
        return 0
    else
        return 1
    fi
}

# -----------------
# Fine-tuning Workflow
# -----------------
run_finetuning_workflow() {
    print_separator "Starting Fine-tuning Workflow"
    
    # Step 1: Plan and preprocess fine-tuning dataset
    print_separator "Step 1: Planning and preprocessing fine-tuning dataset"
    nnUNetv2_plan_and_preprocess -d ${FINETUNING_DATASET} -c ${CONFIG} -pl ResEncUNetPlanner
    check_status "Fine-tuning dataset planning and preprocessing"
    
    # Step 2: Verify or auto-infer source checkpoint
    local checkpoint_to_use="${SOURCE_CHECKPOINT}"
    
    if [ "${AUTO_MODE}" = true ]; then
        if [ -z "${checkpoint_to_use}" ] || [ ! -f "${checkpoint_to_use}" ]; then
            echo "Attempting to auto-infer checkpoint path..."
            checkpoint_to_use=$(auto_infer_checkpoint)
            if [ $? -eq 0 ]; then
                echo "✓ Found checkpoint: ${checkpoint_to_use}"
                SOURCE_CHECKPOINT="${checkpoint_to_use}"
            else
                echo "✗ Unable to automatically find source checkpoint"
                echo "Please manually set SOURCE_CHECKPOINT variable"
                exit 1
            fi
        fi
    fi
    
    if [ ! -f "${checkpoint_to_use}" ]; then
        echo "Error: Source checkpoint does not exist: ${checkpoint_to_use}"
        echo "Please ensure the source model exists, or update SOURCE_CHECKPOINT path"
        exit 1
    fi
    
    # Step 3: Fine-tune on target dataset using source model weights
    print_separator "Step 3: Fine-tuning with source domain pretrained weights"
    echo "Source dataset: ${SOURCE_DATASET}"
    echo "Target dataset: ${FINETUNING_DATASET}"
    echo "Using checkpoint: ${checkpoint_to_use}"
    echo ""
    
    nnUNetv2_train ${FINETUNING_DATASET} ${CONFIG} ${FOLD} \
        -pretrained_weights ${checkpoint_to_use} \
        -p ${FINETUNING_PLANS_IDENTIFIER} \
        -num_gpus 2
    check_status "Fine-tuning training"
    
    print_separator "Fine-tuning Workflow Complete"
}

main() {
    print_separator "nnU-Net Direct Fine-tuning Script"
    echo "Auto-execution: ${AUTO_MODE}"
    echo "Source dataset: ${SOURCE_DATASET}"
    echo "Fine-tuning dataset: ${FINETUNING_DATASET}"
    echo "Configuration: ${CONFIG}"
    echo "Fold: ${FOLD}"
    echo ""
    
    if [ "${AUTO_MODE}" = false ]; then
        read -p "Continue with fine-tuning? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting..."
            exit 0
        fi
        
        read -p "Enter source checkpoint full path (leave empty for auto-inference): " user_checkpoint
        if [ -n "${user_checkpoint}" ]; then
            SOURCE_CHECKPOINT="${user_checkpoint}"
        fi
    fi
    
    run_finetuning_workflow
    
    print_separator "All Tasks Complete"
    echo "Fine-tuned model saved in nnUNet_results/Dataset${FINETUNING_DATASET}_*/"
}

main