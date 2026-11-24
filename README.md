# Tell2Adapt

Tell2Adapt is a unified source-free unsupervised domain adaptation (SFUDA)
framework for medical image segmentation. It uses a vision foundation model
(BiomedParse) guided by regularized text prompts to generate pseudo-labels,
distills the knowledge into a lightweight nnU-Net student, and refines the
student's predictions with visual plausibility checks.


## Pipeline

1. **CAPR** (Context-Aware Prompts Regularization) sends the complete
   dataset-level prompt list to an LLM once, infers the global anatomical site
   and modality, and rewrites every prompt in the canonical form
   `[Target] in [Anatomical Site] [Modality]`.
2. **BiomedParse** receives the canonical prompts and generates pseudo-labels
   for the unlabeled target domain.
3. **Histogram equalization (HE)** is applied to target images before student
   training.
4. **nnU-Net distillation** trains the lightweight student on
   `(HE(target image), pseudo-label)` pairs.
5. **VPR** (Visual Plausibility Refinement) filters anatomically implausible
   connected components using BiomedParse-derived Beta priors. The current
   code release implements it as `vpr.py` and `apply_vpr.py`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Prompt_regulization.py` | CAPR client: Meta-Prompt, canonicalization, validation, provenance |
| `inject_capr_prompts.py` | Dataset-level CAPR injection into NPZ files by explicit label ID |
| `npz_utils.py` | NPZ loader compatible with numpy 1.x and numpy 2.x pickles |
| `Meta-Prompt.txt` | Unified CAPR system prompt described in the paper |
| `inference.py` | Multi-GPU BiomedParse 3D inference |
| `run_biomedparse_standalone.py` | CLI wrapper for BiomedParse inference |
| `src/`, `configs/` | BiomedParse v2 model, dataset, and training configuration |
| `vpr.py`, `apply_vpr.py` | Paper-faithful Visual Plausibility Refinement (VPR) and CLI |
| `assets/target_dist.json` | BiomedParse Beta priors used by VPR |
| `HE_preprocess.py` | Histogram equalization for target images |
| `nnUNet2BiomedParse.py` | NPZ / nnU-Net format conversion utilities |
| `Inference_metrics.py` | Dice / ASD evaluation utilities |
| `Finetuning.sh`, `nnUnet_predict.sh` | nnU-Net distillation and prediction examples |
| `tests/` | CAPR and VPR unit tests |
| `SECURITY.md`, `NOTICE` | Vulnerability reporting and third-party notices |

## Setup

Use Python 3.10 or newer. The reference command sequence is:

```bash
conda create -n tell2adapt python=3.10.14
conda activate tell2adapt
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

`detectron2` is not distributed on PyPI and must be installed from source.

For CAPR prompt processing only (no torch/nnU-Net):

```bash
pip install -r requirements-capr.txt
```

The 3D model uses a deformable-attention CUDA extension. Build it once if
your environment does not already provide it:

```bash
cd src/model/pixel_decoder/ops && sh make.sh
```

Download the BiomedParse v2 checkpoint from the upstream
[BiomedParse](https://github.com/microsoft/BiomedParse) release and place it at
`model_weights/biomedparse_3D_AllData_MultiView_edge.ckpt`, or pass
`--checkpoint /path/to/checkpoint.ckpt` explicitly. No checkpoint is included
in this repository.

## Data format

`inject_capr_prompts.py` and the inference scripts consume a flat directory of
`.npz` files. Each file contains:

| Key | Description |
| --- | --- |
| `imgs` | image volume, `uint8`, expected as `(D, H, W)` for 3D inference |
| `gts` | integer label mask (optional during inference) |
| `text_prompts` | dict mapping label ID to prompt, plus `instance_label: 0` |
| `spacing` | optional voxel spacing |

The input directory is never modified. Outputs are written to a new directory.


## CAPR

CAPR sends the complete dataset-level prompt list to an LLM once and rewrites
each prompt as `[Target] in [Anatomical Site] [Modality]`. The injector maps the
returned prompts back to explicit label IDs and writes a new NPZ dataset.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAPR_API_KEY` | required, or `OPENAI_API_KEY` | OpenAI-compatible API key |
| `CAPR_BASE_URL` | unset | API base URL |
| `CAPR_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | model used by the paper |
| `CAPR_API_STYLE` | auto-detected | `openai` or `anthropic` |
| `ANTHROPIC_AUTH_TOKEN` | unset | Anthropic-compatible token |
| `ANTHROPIC_BASE_URL` | unset | Anthropic-compatible base URL |
| `ANTHROPIC_MODEL` | unset | Anthropic-compatible model name |
| `CAPR_TIMEOUT` | `120` | request timeout in seconds |
| `CAPR_MAX_TOKENS` | `4096` | output token budget |
| `CAPR_VALIDATION_RETRIES` | `3` | retries after invalid responses |
| `CAPR_META_PROMPT` | `Meta-Prompt.txt` | unified meta-prompt path |
| `CAPR_EXTRA_BODY` | unset | optional provider-specific JSON `extra_body` |

```bash
export CAPR_API_KEY=...
export CAPR_MODEL=Qwen/Qwen3-VL-8B-Instruct
export CAPR_BASE_URL=...

python inject_capr_prompts.py \
  --input_dir /path/to/Datasets/MR_BraTS-T1c_processed \
  --output_dir /path/to/Datasets/MR_BraTS-T1c_CAPR \
  --context "head post-contrast T1-weighted MR"
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--dry-run` | validate and print provenance without writing NPZ files |
| `--limit N` | process only the first N cases |
| `--prompts_json` | explicit `{label_id: raw_prompt}` mapping |
| `--dataset_json` | nnU-Net `dataset.json` for label names |
| `--store-json` | store `text_prompts` as JSON instead of a pickled dict |
| `--overwrite` | allow writing into an existing output directory |

Responses are rejected and retried when the prompt count is wrong, a prompt is
empty or duplicated, a placeholder remains, or the canonical
`[Target] in [Site] [Modality]` structure is invalid. The output includes
`capr_provenance.json` with model, raw/canonical prompts, label IDs, variants,
and retry information.

Only run the injector on trusted NPZ files. Loading pickled `text_prompts` can
execute constructors; use `--store-json` for new outputs when possible.

## Inference

```bash
python run_biomedparse_standalone.py \
  --input_dir /path/to/Dataset_CAPR \
  --output_dir /path/to/predictions \
  --checkpoint model_weights/biomedparse_3D_AllData_MultiView_edge.ckpt \
  --gpus 0 \
  --save_nii \
  --save_prob
```

`inference.py` now validates that the number of prompts equals the number of
label IDs and that the model returns exactly one mask/existence score per
prompt. A mismatch raises an error instead of silently assigning masks to the
wrong classes.

## VPR

VPR follows the paper's Eq. (2)–(4):

1. extract each connected component and resolve its anatomical target;
2. compute four features: mean prediction probability and mean R/G/B intensity;
3. score the features with the target's four Beta priors;
4. remove components below the global
   `tau = mean(raw scores) - 2 * std(raw scores)` threshold.

The threshold is computed with shifted exponentials to avoid underflow while
remaining equivalent to the raw-score formula. Components without a matching
prior are kept unchanged. Every scored target requires a real BiomedParse
probability map; missing probabilities are an error.

```bash
python apply_vpr.py \
  --pred_dir /path/to/pred_with_probability_maps \
  --image_dir /path/to/original_input_npz_dir \
  --output_dir /path/to/pred_vpr \
  --priors assets/target_dist.json \
  --modality CT-Abdomen \
  --dataset_json /path/to/Dataset125_AMOS-CT/dataset.json
```

Prediction files contain `segs`. Matching `<case>_prob.npz` files under
`probability_maps/` contain `probability_maps` with shape `(N,D,H,W)` and
`class_ids` with shape `(N,)`. Outputs include refined NPZ files and
`vpr_stats.json` with component scores and Dice before/after. Dice is
macro-averaged over the complete target list.

On five MR→CT and five CT→MR AMOS cases from the 30-epoch student predictions,
the paper implementation removed no components and left mean Dice unchanged at
0.769637 and 0.834133, respectively. These cases do not demonstrate a
meaningful VPR benefit.

## Citation

```bibtex
@InProceedings{Shi_2026_CVPR,
    author    = {Shi, Yulong and Li, Shijie and Li, Ziyi and Qi, Lin},
    title     = {Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {6941-6950}
}
```

## License and acknowledgements

This repository is released under the Apache License 2.0 (see `LICENSE`). It
contains code derived from Microsoft BiomedParse v2 and other third-party
projects; see `NOTICE` for copyright and license notices. Dataset and model
assets remain subject to their original licenses and terms of use.
