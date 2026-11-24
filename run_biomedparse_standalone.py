"""
Standalone BiomedParse inference script — called via subprocess from Search2Adapt.

This script is designed to be run under the `biomedparsev2` conda environment,
analogous to how run_sam3_standalone.py is run under the `sam3` env.

Usage:
    python run_biomedparse_standalone.py \
        --input_dir /path/to/npz_files \
        --output_dir /path/to/output \
        --checkpoint /path/to/checkpoint.ckpt \
        --gpus 0,1 \
        --slice_batch_size 8 \
        --interpolate_size 512 \
        --save_nii

Input:
    Directory of .npz files, each containing:
      - imgs: uint8 array (shape depends on modality)
      - text_prompts: dict {"1": "organ description", "instance_label": 0}
      - gts: (optional) ground truth label mask

Output:
    - NPZ predictions in output_dir/{casename}.npz  (always)
    - NIfTI predictions in output_dir/nii_predictions/{casename}_pred.nii.gz  (if --save_nii)
"""

import os
import sys
import argparse
import multiprocessing as mp


# Must be set before any CUDA imports
def main():
    parser = argparse.ArgumentParser(description="BiomedParse standalone inference")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .npz input files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for output predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get(
            "BIOMEDPARSE_CHECKPOINT",
            "model_weights/biomedparse_3D_AllData_MultiView_edge.ckpt",
        ),
        help="Path to BiomedParse checkpoint (or set BIOMEDPARSE_CHECKPOINT)",
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1", help="Comma-separated GPU IDs"
    )
    parser.add_argument(
        "--slice_batch_size",
        type=int,
        default=8,
        help="Number of slices per forward pass",
    )
    parser.add_argument(
        "--interpolate_size",
        type=int,
        default=512,
        help="Spatial resolution for inference",
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Object existence score threshold",
    )
    parser.add_argument(
        "--save_nii",
        action="store_true",
        help="Also save predictions as NIfTI (.nii.gz)",
    )
    parser.add_argument(
        "--save_prob", action="store_true", help="Also save probability maps"
    )
    args = parser.parse_args()

    # Add BiomedParsev2 to path so its modules can be imported
    biomedparse_root = os.path.dirname(os.path.abspath(__file__))
    if biomedparse_root not in sys.path:
        sys.path.insert(0, biomedparse_root)

    # Now we can patch the global config in inference.py and call main
    import inference as bp_inference

    # Patch global config variables
    gpu_list = [g.strip() for g in args.gpus.split(",")]
    bp_inference.CUDA_DEVICES = gpu_list
    bp_inference.NUM_WORKERS = len(gpu_list)
    bp_inference.INPUT_DIR = args.input_dir
    bp_inference.OUTPUT_DIR = args.output_dir
    bp_inference.CHECKPOINT_PATH = args.checkpoint
    bp_inference.SLICE_BATCH_SIZE = args.slice_batch_size
    bp_inference.INTERPOLATE_SIZE = args.interpolate_size
    bp_inference.NMS_IOU_THRESHOLD = args.nms_threshold
    bp_inference.SCORE_THRESHOLD = args.score_threshold
    bp_inference.SAVE_NII = args.save_nii
    bp_inference.SAVE_PROB = args.save_prob
    bp_inference.NII_OUTPUT_DIR = None  # will default to output_dir/nii_predictions
    bp_inference.PROB_OUTPUT_DIR = None

    print(f"[run_biomedparse_standalone] Config:")
    print(f"  GPUs: {gpu_list}")
    print(f"  Input: {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Batch size: {args.slice_batch_size}")
    print(f"  Interpolate: {args.interpolate_size}")
    print(f"  Save NII: {args.save_nii}")

    bp_inference.main()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
