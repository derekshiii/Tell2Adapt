"""Visual Plausibility Refinement (VPR) for Tell2Adapt.

The paper refines final predictions by:

1. identifying the general anatomical category of each predicted target;
2. retrieving BiomedParse's four Beta priors for that category
   (pixel probability, R, G, B);
3. computing a plausibility score for every connected component;
4. retaining only components whose score is above
   ``tau = mean(scores) - 2 * std(scores)``.

Raw-score statistics use shifted exponentials to evaluate Eq.4 without underflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.stats import beta

LabelId = int
PriorParams = Sequence[Sequence[float]]

# Medical label names vary between nnU-Net, CAPR prompts, and BiomedParse's
# prior database. These aliases preserve the prior's naming without changing
# the input annotation IDs.
_TARGET_ALIASES = {
    "arota": "aorta",
    "aortic": "aorta",
    "inferior vena cava": "postcava",
    "vena cava": "postcava",
    "gall bladder": "gallbladder",
    "right heart ventricle": "right ventricle",
    "left heart ventricle": "left ventricle",
    "myocardium": "myocardium",
    "prostate uterus": "prostate/uterus",
    "prostate or uterus": "prostate/uterus",
    "non enhancing tumor core": "tumor core",
    "non enhancing tumor": "non-enhancing tumor",
    "enhancing tissue": "enhancing tumor",
    "enhancing tumour": "enhancing tumor",
}


class VPRError(ValueError):
    pass


def _clean_name(name: Any) -> str:
    value = str(name).lower()
    value = "".join(ch if ch.isalnum() or ch in "/ " else " " for ch in value)
    value = " ".join(value.split())
    return value


def normalize_target_name(name: Any) -> str:
    cleaned = _clean_name(name)
    return _TARGET_ALIASES.get(cleaned, cleaned)


def load_priors(path: Union[str, Path]) -> Dict[str, Dict[str, PriorParams]]:
    raw = Path(path).read_text(encoding="utf-8")
    if raw.startswith("version https://git-lfs"):
        raise VPRError(
            f"{path} is a Git LFS pointer, not the target prior JSON. Download "
            "microsoft/BiomedParse:target_dist.json before running VPR."
        )
    try:
        priors = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VPRError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(priors, dict):
        raise VPRError("target prior JSON must be an object")
    for modality, targets in priors.items():
        if not isinstance(targets, dict):
            raise VPRError(f"prior modality {modality!r} must map to target names")
        for target, params in targets.items():
            if not isinstance(params, list) or len(params) != 4:
                raise VPRError(f"prior {modality}/{target} must contain 4 Beta pairs")
            for pair in params:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise VPRError(f"invalid Beta pair in prior {modality}/{target}")
                try:
                    alpha, beta_value = map(float, pair)
                except (TypeError, ValueError) as exc:
                    raise VPRError(
                        f"invalid Beta pair in prior {modality}/{target}"
                    ) from exc
                if (
                    not np.isfinite(alpha)
                    or not np.isfinite(beta_value)
                    or alpha <= 0
                    or beta_value <= 0
                ):
                    raise VPRError(f"invalid Beta pair in prior {modality}/{target}")
    return priors


def _normalize_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        rgb = np.stack([arr.astype(float)] * 3, axis=-1)
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        rgb = arr.astype(float)
    else:
        raise VPRError(f"image must be (D,H,W) or (D,H,W,3), got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        rgb /= 256.0
    else:
        maximum = float(np.nanmax(rgb))
        if maximum <= 1.0001:
            pass
        elif maximum <= 255.0001:
            rgb /= 256.0
        else:
            raise VPRError(
                "image values must be in [0, 1] or [0, 255]. Convert original "
                "HU/intensity NIfTI files to the BiomedParse uint8 image space "
                "before applying VPR priors."
            )
    return np.clip(rgb, 0.0, 1.0)


def _component_structure(ndim: int) -> np.ndarray:
    return np.ones((3,) * ndim, dtype=bool)


class VisualPlausibilityRefiner:
    def __init__(
        self,
        priors: Mapping[str, Mapping[str, PriorParams]],
        modality: str,
        *,
        threshold_sigma: float = 2.0,
        min_voxels: int = 1,
    ) -> None:
        if modality not in priors:
            raise VPRError(
                f"modality {modality!r} is absent from target priors; "
                f"known modalities: {sorted(priors)}"
            )
        if not np.isfinite(threshold_sigma) or threshold_sigma < 0:
            raise VPRError("threshold_sigma must be finite and non-negative")
        if min_voxels < 1:
            raise VPRError("min_voxels must be at least 1")
        self.priors = priors
        self.modality = modality
        self.modality_priors = priors[modality]
        self.threshold_sigma = float(threshold_sigma)
        self.min_voxels = int(min_voxels)

    @classmethod
    def from_file(
        cls,
        priors_path: Union[str, Path],
        modality: str,
        **kwargs: Any,
    ) -> "VisualPlausibilityRefiner":
        return cls(load_priors(priors_path), modality, **kwargs)

    def resolve_prior(self, target_name: Any) -> Optional[Tuple[str, PriorParams]]:
        wanted = normalize_target_name(target_name)
        exact = {normalize_target_name(key): key for key in self.modality_priors}
        if wanted in exact:
            key = exact[wanted]
            return key, self.modality_priors[key]
        candidates = [
            key
            for key, normalized in exact.items()
            if wanted and (wanted in normalized or normalized in wanted)
        ]
        if len(candidates) == 1:
            return candidates[0], self.modality_priors[candidates[0]]
        return None

    @staticmethod
    def _feature_logscore(
        features: Sequence[float], params: PriorParams
    ) -> Tuple[float, List[float]]:
        logscore = 0.0
        clipped: List[float] = []
        for value, (alpha, beta_value) in zip(features, params):
            x = float(np.clip(value, 1e-6, 1.0 - 1e-6))
            clipped.append(x)
            logscore += float(beta.logpdf(x, float(alpha), float(beta_value)))
        return logscore, clipped

    @staticmethod
    def _paper_threshold_logscore(
        log_scores: Sequence[float], sigma: float
    ) -> Optional[float]:
        if not log_scores:
            return None
        values = np.asarray(log_scores, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise VPRError("VPR component scores must be finite")
        shift = float(np.max(values))
        scaled_scores = np.exp(values - shift)
        margin = float(np.mean(scaled_scores) - sigma * np.std(scaled_scores))
        if margin <= 0:
            return None
        return shift + float(np.log(margin))

    def refine(
        self,
        labels: np.ndarray,
        *,
        image: np.ndarray,
        label_targets: Mapping[LabelId, Any],
        probabilities: Optional[Mapping[LabelId, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        labels = np.asarray(labels).astype(np.int64, copy=False)
        if labels.ndim not in (2, 3):
            raise VPRError("labels must be 2D or 3D")
        rgb = _normalize_image(image)
        if rgb.shape[: labels.ndim] != labels.shape:
            raise VPRError(
                f"label/image shape mismatch: labels {labels.shape}, image {rgb.shape}"
            )
        if probabilities is None:
            probabilities = {}

        structure = _component_structure(labels.ndim)
        candidates: List[Dict[str, Any]] = []
        for label_id, target_name in sorted(label_targets.items()):
            class_mask = labels == int(label_id)
            if not np.any(class_mask):
                continue
            resolved = self.resolve_prior(target_name)
            prob = probabilities.get(int(label_id))
            if prob is None:
                prob_array = None
            else:
                prob_array = np.asarray(prob, dtype=float)
                if prob_array.shape != labels.shape:
                    raise VPRError(
                        f"probability map for label {label_id} has shape "
                        f"{prob_array.shape}, expected {labels.shape}"
                    )
                if not np.all(np.isfinite(prob_array)) or np.any(
                    (prob_array < 0.0) | (prob_array > 1.0)
                ):
                    raise VPRError(
                        f"probability map for label {label_id} must be finite "
                        "and within [0, 1]"
                    )
            if resolved is not None and prob_array is None:
                raise VPRError(
                    f"label {label_id} ({target_name}) requires a probability "
                    "map for paper VPR"
                )

            connected, component_count = ndimage.label(class_mask, structure=structure)
            for component in range(1, component_count + 1):
                component_mask = connected == component
                size = int(np.count_nonzero(component_mask))
                if size < self.min_voxels:
                    continue
                probability = (
                    float(np.mean(prob_array[component_mask]))
                    if prob_array is not None
                    else 1.0
                )
                intensities = rgb[component_mask].reshape(-1, rgb.shape[-1]).mean(axis=0)
                features = [probability, float(intensities[0]), float(intensities[1]), float(intensities[2])]
                item = {
                    "label_id": int(label_id),
                    "target": str(target_name),
                    "component": component,
                    "size": size,
                    "features": features,
                    "log_score": None,
                    "prior_target": resolved[0] if resolved else None,
                    "keep": True,
                }
                if resolved is not None:
                    item["log_score"], item["clipped_features"] = self._feature_logscore(
                        features, resolved[1]
                    )
                candidates.append(item)

        scored = [item for item in candidates if item.get("log_score") is not None]
        threshold = self._paper_threshold_logscore(
            [float(item["log_score"]) for item in scored],
            self.threshold_sigma,
        )
        if threshold is not None:
            for item in scored:
                if float(item["log_score"]) < threshold:
                    item["keep"] = False

        refined = labels.copy()
        removed = 0
        for item in candidates:
            if not item["keep"]:
                # Re-identify the original connected component by the source
                # label array, so filtering one class does not shift another.
                component_mask = (
                    ndimage.label(labels == item["label_id"], structure=structure)[0]
                    == item["component"]
                )
                refined[component_mask] = 0
                removed += 1

        return refined, {
            "modality": self.modality,
            "threshold_log_score": threshold,
            "threshold_scope": "all_scored_components",
            "threshold_sigma": self.threshold_sigma,
            "component_count": len(candidates),
            "scored_component_count": len(scored),
            "removed_components": removed,
            "components": candidates,
        }
