#!/usr/bin/env python3
"""Apply Visual Plausibility Refinement (VPR) to saved predictions.

The intended stage is the adapted nnU-Net/student prediction, but the same
NPZ interface also works with BiomedParse prediction/probability folders.

Example (BiomedParse pseudo-label stage):
    python apply_vpr.py \
        --pred_dir /tmp/pred_with_probs \
        --image_dir /path/to/original_input_npz_dir \
        --output_dir /tmp/pred_vpr \
        --priors assets/target_dist.json \
        --modality CT-Abdomen \
        --label_targets_json amos_targets.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from vpr import VisualPlausibilityRefiner, load_priors


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Tell2Adapt VPR to predictions")
    parser.add_argument("--pred_dir", required=True, help="directory of prediction .npz files")
    parser.add_argument(
        "--prob_dir",
        default=None,
        help="directory of *_prob.npz; defaults to <pred_dir>/probability_maps",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="directory of matching image .npz files (keys: imgs, optionally gts)",
    )
    parser.add_argument("--output_dir", required=True, help="output directory")
    parser.add_argument(
        "--priors",
        default=str(Path(__file__).with_name("assets") / "target_dist.json"),
        help="BiomedParse target_dist.json",
    )
    parser.add_argument(
        "--modality",
        required=True,
        help="prior modality, e.g. CT-Abdomen or MRI-T1-Gd-Brain",
    )
    parser.add_argument(
        "--label_targets_json",
        default=None,
        help="JSON mapping label ID -> BiomedParse prior target name",
    )
    parser.add_argument(
        "--dataset_json",
        default=None,
        help="nnU-Net dataset.json used when --label_targets_json is absent",
    )
    parser.add_argument("--threshold-sigma", type=float, default=2.0)
    parser.add_argument("--min-voxels", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_targets(args: argparse.Namespace) -> Dict[int, str]:
    if args.label_targets_json:
        payload = json.loads(Path(args.label_targets_json).read_text(encoding="utf-8"))
        result: Dict[int, str] = {}
        for key, value in payload.items():
            result[int(key)] = str(value)
        return result

    if args.dataset_json:
        info = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
        labels = info.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError("dataset.json labels must be a mapping")
        pairs = list(labels.items())
        numeric_keys = sum(1 for key, _ in pairs if str(key).strip().isdigit())
        numeric_values = sum(1 for _, value in pairs if str(value).strip().isdigit())
        result = {}
        for key, value in pairs:
            if numeric_keys >= numeric_values:
                if not str(key).strip().isdigit():
                    continue
                label_id, name = int(key), str(value)
            else:
                if not str(value).strip().isdigit():
                    continue
                label_id, name = int(value), str(key)
            if label_id != 0:
                result[label_id] = name
        return result

    raise ValueError("provide --label_targets_json or --dataset_json")


def _load_numeric_npz(path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            try:
                data[key] = npz[key]
            except ValueError:
                # Skip pickled object arrays (e.g. legacy text_prompts); VPR
                # only needs numeric image/label/probability arrays.
                continue
    return data


def load_probability_maps(
    pred_path: Path,
    prob_dir: Optional[Path],
    image_shape: Tuple[int, ...],
) -> Optional[Dict[int, np.ndarray]]:
    if prob_dir is None:
        return None
    base = pred_path.stem
    candidate = prob_dir / f"{base}_prob.npz"
    if not candidate.exists():
        candidate = prob_dir / pred_path.name
    if not candidate.exists():
        return None
    data = _load_numeric_npz(candidate)
    if "probability_maps" not in data:
        raise ValueError(f"{candidate}: no probability_maps array")
    maps = np.asarray(data["probability_maps"])
    class_ids = np.asarray(data.get("class_ids", []))
    if maps.shape[1:] != image_shape:
        raise ValueError(
            f"{candidate}: probability shape {maps.shape[1:]} != image shape {image_shape}"
        )
    if class_ids.ndim == 0:
        class_ids = class_ids.reshape(-1)
    if len(class_ids) != maps.shape[0]:
        raise ValueError(f"{candidate}: class_ids length {len(class_ids)} != maps {maps.shape[0]}")
    return {int(label_id): maps[index] for index, label_id in enumerate(class_ids)}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * int(np.logical_and(a, b).sum()) / denominator)


def evaluate(
    labels: np.ndarray,
    gt: Optional[np.ndarray],
    label_ids: Iterable[int],
) -> Dict[str, Any]:
    if gt is None:
        return {}
    scores: Dict[str, float] = {}
    values: List[float] = []
    for label_id in sorted(set(int(x) for x in label_ids)):
        value = dice(labels == label_id, gt == label_id)
        scores[f"class_{label_id}"] = value
        if value is not None:
            values.append(value)
    return {
        "mean_dice": float(np.mean(values)) if values else None,
        "valid_classes": len(values),
        "per_class": scores,
    }


def apply_case(
    pred_path: Path,
    image_path: Path,
    prob_dir: Optional[Path],
    refiner: VisualPlausibilityRefiner,
    targets: Mapping[int, str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred = _load_numeric_npz(pred_path)
    image = _load_numeric_npz(image_path)
    if "segs" not in pred:
        raise ValueError(f"{pred_path}: missing 'segs'")
    if "imgs" not in image:
        raise ValueError(f"{image_path}: missing 'imgs'")
    labels = np.asarray(pred["segs"]).astype(np.int64, copy=False)
    img = np.asarray(image["imgs"])
    if labels.shape != img.shape[: labels.ndim]:
        raise ValueError(
            f"{pred_path}: segs shape {labels.shape} incompatible with imgs {img.shape}"
        )
    probabilities = load_probability_maps(pred_path, prob_dir, labels.shape)

    label_ids = sorted({int(x) for x in np.unique(labels) if int(x) > 0})
    gt = np.asarray(image["gts"]).astype(np.int64, copy=False) if "gts" in image else None

    case_targets = {
        int(label_id): targets.get(int(label_id), str(label_id))
        for label_id in label_ids
    }
    refined, stats = refiner.refine(
        labels,
        image=img,
        label_targets=case_targets,
        probabilities=probabilities,
    )
    evaluation_label_ids = sorted(int(label_id) for label_id in targets)
    original_eval = evaluate(labels, gt, evaluation_label_ids)
    refined_eval = evaluate(refined, gt, evaluation_label_ids)
    stats.update(
        {
            "prediction": str(pred_path),
            "image": str(image_path),
            "probabilities_used": probabilities is not None,
            "labels": label_ids,
            "evaluated_labels": evaluation_label_ids,
            "label_targets": {str(k): v for k, v in case_targets.items()},
            "original": original_eval,
            "refined": refined_eval,
        }
    )
    return refined, stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    prob_dir = Path(args.prob_dir) if args.prob_dir else pred_dir / "probability_maps"
    if not prob_dir.exists():
        raise ValueError(f"probability directory does not exist: {prob_dir}")

    pred_files = sorted(pred_dir.glob("*.npz"))
    pred_files = [
        p for p in pred_files if p.parent != prob_dir and "probability" not in p.name.lower()
    ]
    if args.limit and args.limit > 0:
        pred_files = pred_files[: args.limit]
    if not pred_files:
        raise ValueError(f"no prediction .npz files under {pred_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    priors = load_priors(args.priors)
    refiner = VisualPlausibilityRefiner(
        priors,
        args.modality,
        threshold_sigma=args.threshold_sigma,
        min_voxels=args.min_voxels,
    )
    targets = load_targets(args)
    all_stats: List[Dict[str, Any]] = []

    for pred_path in pred_files:
        image_path = image_dir / pred_path.name
        if not image_path.exists():
            raise FileNotFoundError(f"missing matching image NPZ: {image_path}")
        refined, stats = apply_case(
            pred_path,
            image_path,
            prob_dir,
            refiner,
            targets,
        )
        out_path = output_dir / pred_path.name
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite")
        np.savez_compressed(out_path, segs=refined.astype(np.uint16))
        compact = dict(stats)
        components = compact.pop("components", [])
        compact["components"] = [
            {
                "label_id": c["label_id"],
                "target": c["target"],
                "prior_target": c.get("prior_target"),
                "size": c["size"],
                "features": c["features"],
                "log_score": c.get("log_score"),
                "keep": c["keep"],
            }
            for c in components
        ]
        all_stats.append(compact)
        od = stats.get("original", {}).get("mean_dice")
        rd = stats.get("refined", {}).get("mean_dice")
        print(
            f"{pred_path.name}: components={stats['component_count']} "
            f"removed={stats['removed_components']} "
            f"dice={'' if od is None else f'{od:.4f}'} -> "
            f"{'' if rd is None else f'{rd:.4f}'}"
        )

    summary_path = output_dir / "vpr_stats.json"
    summary_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    original_values = [
        s["original"]["mean_dice"]
        for s in all_stats
        if s.get("original", {}).get("mean_dice") is not None
    ]
    refined_values = [
        s["refined"]["mean_dice"]
        for s in all_stats
        if s.get("refined", {}).get("mean_dice") is not None
    ]
    print(f"VPR summary: {summary_path}")
    if original_values and refined_values:
        print(
            f"mean Dice original={np.mean(original_values):.4f} "
            f"refined={np.mean(refined_values):.4f} "
            f"delta={np.mean(refined_values) - np.mean(original_values):+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
