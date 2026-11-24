#!/usr/bin/env python3
"""Apply CAPR to a BiomedParse NPZ dataset and inject canonical prompts.

The paper runs CAPR once on the complete, dataset-level prompt list and then
uses the resulting canonical prompts for every case. This script follows that
contract:

* all unique labels in the input directory are collected and sent to the LLM
  in one call;
* the returned canonical prompts are validated (count, format, placeholders);
* every output NPZ receives the canonical prompt for each label it contains,
  keyed by the explicit label ID; prompts are never mapped by dict order;
* a sidecar ``capr_provenance.json`` records the model, raw/canonical prompts,
  variants, and validation outcome.

Usage:
    export CAPR_API_KEY=...
    python inject_capr_prompts.py \
        --input_dir /path/to/Datasets/MR_BraTS-T1c_processed \
        --output_dir /path/to/Datasets/MR_BraTS-T1c_CAPR \
        --context "head post-contrast T1-weighted MR"

The input directory is never modified. It must be a flat directory of ``.npz``
files (for example ``Datasets/MR_BraTS-T1c_processed``), not a tree of prompt
batches.

Input NPZ files are trusted local experiment artifacts; ``text_prompts`` is
read with ``allow_pickle=True`` because the existing datasets store a pickled
dict. New outputs can be written as JSON strings with ``--store-json``.

If every raw prompt contains ``Unknown`` (the historical SFUDA_Agent failure),
pass ``--context "abdomen CT"`` so CAPR has a real global context to work with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from Prompt_regulization import CAPRClient, CAPRResult
from npz_utils import decode_text_prompts, load_npz, load_text_prompts


def load_label_names(dataset_json: Optional[Union[str, Path]]) -> Dict[int, str]:
    """Read the nnU-Net ``labels`` mapping, excluding background."""

    if not dataset_json:
        return {}
    path = Path(dataset_json)
    if not path.exists():
        raise FileNotFoundError(f"dataset.json not found: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    labels = info.get("labels", {})
    if isinstance(labels, list):
        pairs = list(enumerate(labels))
    elif isinstance(labels, dict):
        pairs = list(labels.items())
    else:
        raise ValueError("dataset.json labels must be a mapping or a list")

    # nnU-Net stores name -> ID ({"spleen": "1"}), while other tools store
    # ID -> name ({"1": "spleen"}). Detect the orientation instead of assuming.
    numeric_keys = sum(1 for key, _ in pairs if str(key).strip().isdigit())
    numeric_values = sum(1 for _, value in pairs if str(value).strip().isdigit())
    result: Dict[int, str] = {}
    for key, value in pairs:
        if numeric_keys >= numeric_values:
            if not str(key).strip().isdigit():
                continue
            label_id, name = int(key), str(value)
        else:
            if not str(value).strip().isdigit():
                continue
            label_id, name = int(value), str(key)
        if label_id == 0:
            continue
        result[label_id] = name
    return result


def load_prompt_overrides(
    prompts_json: Optional[Union[str, Path]],
) -> Dict[int, str]:
    """Load an optional explicit ``label ID -> raw prompt`` mapping."""

    if not prompts_json:
        return {}
    path = Path(prompts_json)
    if not path.exists():
        raise FileNotFoundError(f"prompts JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prompts JSON must be an object mapping label IDs to prompts")
    result: Dict[int, str] = {}
    for key, value in payload.items():
        try:
            label_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid label ID in prompts JSON: {key!r}") from exc
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompt for label {label_id} must be a non-empty string")
        result[label_id] = value.strip()
    return result


def _prompt_keys(prompts: Mapping[Any, Any]) -> Set[int]:
    result: Set[int] = set()
    for key in prompts:
        if str(key).strip().lower() == "instance_label":
            continue
        try:
            result.add(int(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-integer prompt key: {key!r}") from exc
    return result


def collect_prompts(
    input_dir: Union[str, Path],
    *,
    label_names: Optional[Mapping[int, str]] = None,
    prompt_overrides: Optional[Mapping[int, str]] = None,
    limit: int = 0,
) -> Tuple[List[Path], Dict[int, str], Dict[str, Set[int]], Dict[int, Counter]]:
    """Collect one raw prompt per label and the label set of every case."""

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"input directory not found: {input_path}")

    files = sorted(input_path.glob("*.npz"))
    if limit and limit > 0:
        files = files[:limit]
    if not files:
        raise ValueError(f"no .npz files found in {input_path}")

    label_names = dict(label_names or {})
    prompt_overrides = dict(prompt_overrides or {})
    variants: Dict[int, Counter] = defaultdict(Counter)
    case_labels: Dict[str, Set[int]] = {}

    for path in files:
        data = load_npz(path, include_prompts=True)
        prompts: Dict[str, Any] = data.get("text_prompts", {})

        labels = _prompt_keys(prompts)
        gt_labels: Set[int] = set()
        if "gts" in data:
            gts = np.asarray(data["gts"])
            gt_labels = {int(value) for value in np.unique(gts) if int(value) > 0}

            missing_gt_prompts = gt_labels - labels
            if missing_gt_prompts:
                raise ValueError(
                    f"{path.name}: ground-truth labels without prompts: "
                    f"{sorted(missing_gt_prompts)}"
                )
            labels = labels | gt_labels

            if not labels:
                raise ValueError(f"{path.name} contains no non-background labels")
            case_labels[path.name] = set(labels)

            for label_id in labels:
                if label_id in prompt_overrides:
                    variants[label_id][prompt_overrides[label_id]] += 1
                    continue
                if str(label_id) in prompts:
                    value = prompts[str(label_id)]
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"{path.name}: prompt for label {label_id} is empty"
                        )
                    variants[label_id][value.strip()] += 1
                    continue
                if label_id in label_names:
                    variants[label_id][label_names[label_id]] += 1
                    continue
                raise ValueError(
                    f"{path.name}: label {label_id} has no prompt and no label name; "
                    "provide --prompts_json or --dataset_json"
                )

    all_labels: Set[int] = set()
    for labels in case_labels.values():
        all_labels.update(labels)
    missing = sorted(all_labels - set(variants))
    if missing:
        raise ValueError(f"labels without prompts: {missing}")

    raw_prompts = {
        label_id: counter.most_common(1)[0][0]
        for label_id, counter in sorted(variants.items())
    }
    return files, raw_prompts, case_labels, variants


def inject_prompts(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    capr_client: Optional[CAPRClient] = None,
    dataset_json: Optional[Union[str, Path]] = None,
    prompts_json: Optional[Union[str, Path]] = None,
    context_hint: Optional[str] = None,
    limit: int = 0,
    store_json: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Regularize prompts once and write a CAPR-injected copy of the dataset."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    label_names = load_label_names(dataset_json)
    prompt_overrides = load_prompt_overrides(prompts_json)

    files, raw_prompts, case_labels, variants = collect_prompts(
        input_path,
        label_names=label_names,
        prompt_overrides=prompt_overrides,
        limit=limit,
    )

    if not context_hint and all(
        "unknown" in prompt.lower() for prompt in raw_prompts.values()
    ):
        raise ValueError(
            "All raw prompts contain 'Unknown'; provide --context (for example "
            "'abdomen CT') instead of asking CAPR to infer a missing context"
        )

    client = capr_client or CAPRClient()
    result: CAPRResult = client.regularize(raw_prompts, context_hint=context_hint)

    expected_labels = set(raw_prompts)
    if set(result.prompts) != expected_labels:
        raise ValueError(
            "CAPR returned a different label set: "
            f"expected {sorted(expected_labels)}, got {sorted(result.prompts)}"
        )

    provenance: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "input_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "num_cases": len(files),
        "store_json": bool(store_json),
        "context_hint": context_hint,
        "dataset_json": str(dataset_json) if dataset_json else None,
        "prompts_json": str(prompts_json) if prompts_json else None,
        "labels": {},
        "capr": result.to_dict(),
        "warnings": [],
    }
    for label_id in sorted(raw_prompts):
        provenance["labels"][str(label_id)] = {
            "raw": raw_prompts[label_id],
            "canonical": result.prompts[label_id],
            "variants": variants[label_id].most_common(),
        }
        if len(variants[label_id]) > 1:
            provenance["warnings"].append(
                f"label {label_id} had {len(variants[label_id])} raw prompt variants; "
                "used the most frequent one"
            )

    if dry_run:
        return provenance

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_path} (use --overwrite)"
            )
    output_path.mkdir(parents=True, exist_ok=True)

    for path in files:
        data = load_npz(path, include_prompts=False)
        try:
            original_prompts = load_text_prompts(path)
        except KeyError:
            original_prompts = {}
        present = case_labels[path.name]
        new_prompts: Dict[str, Any] = {
            str(label_id): result.prompts[label_id] for label_id in sorted(present)
        }
        new_prompts["instance_label"] = original_prompts.get("instance_label", 0)
        data["text_prompts"] = (
            json.dumps(new_prompts, ensure_ascii=False) if store_json else new_prompts
        )
        np.savez_compressed(output_path / path.name, **data)

    provenance_path = output_path / "capr_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return provenance


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply CAPR to a BiomedParse NPZ dataset"
    )
    parser.add_argument("--input_dir", required=True, help="directory of input .npz files")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="output directory (default: <input_dir>_CAPR)",
    )
    parser.add_argument(
        "--dataset_json",
        default=None,
        help="nnU-Net dataset.json used for label names when prompts are absent",
    )
    parser.add_argument(
        "--prompts_json",
        default=None,
        help="optional JSON mapping of label ID to raw prompt",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="optional global context hint; CAPR still infers it from the list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N cases (for a smoke test)",
    )
    parser.add_argument(
        "--store-json",
        action="store_true",
        help="store text_prompts as a JSON string instead of a pickled dict",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow an existing output dir")
    parser.add_argument("--dry-run", action="store_true", help="print provenance; write nothing")
    parser.add_argument("--model", default=None, help="override CAPR_MODEL")
    parser.add_argument("--base_url", default=None, help="override CAPR_BASE_URL")
    parser.add_argument("--meta_prompt", default=None, help="override Meta-Prompt.txt")
    args = parser.parse_args(argv)

    input_path = Path(args.input_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / f"{input_path.name}_CAPR"
    )

    client = CAPRClient(
        base_url=args.base_url,
        model=args.model,
        meta_prompt_path=args.meta_prompt,
    )
    provenance = inject_prompts(
        input_path,
        output_dir,
        capr_client=client,
        dataset_json=args.dataset_json,
        prompts_json=args.prompts_json,
        context_hint=args.context,
        limit=args.limit,
        store_json=args.store_json,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(json.dumps(provenance, ensure_ascii=False, indent=2))
        return 0

    print(
        f"CAPR applied to {provenance['num_cases']} case(s); "
        f"output: {provenance['output_dir']}",
        file=sys.stderr,
    )
    print(
        f"provenance: {Path(provenance['output_dir']) / 'capr_provenance.json'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
