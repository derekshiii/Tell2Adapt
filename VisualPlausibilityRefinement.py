import json
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.special import betaln
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import warnings

class PriorDistributionRefiner:

    def __init__(self, prior_json_path: str):
        with open(prior_json_path, 'r') as f:
            self.priors = json.load(f)
        
    def find_matching_category(self, object_name: str, modality: str) -> Optional[str]:
        if modality not in self.priors:
            return None
        
        object_lower = object_name.lower().strip()
        
        if object_name in self.priors[modality]:
            return object_name
        
        for key in self.priors[modality].keys():
            if key.lower() == object_lower:
                return key
        
        for key in self.priors[modality].keys():
            key_lower = key.lower()
            if object_lower in key_lower or key_lower in object_lower:
                return key
        
        return None
    
    def get_prior_params(self, modality: str, object_type: str) -> Optional[List[List[float]]]:
        matched_category = self.find_matching_category(object_type, modality)
        if matched_category:
            return self.priors[modality][matched_category]
        return None
    
    def extract_features(self, component_mask: np.ndarray,
                        prob_map_255: np.ndarray,
                        image_255: np.ndarray) -> np.ndarray:

        pixels_in = component_mask > 0
        prob_vals = prob_map_255[pixels_in]

        if prob_vals.max() <= 127:
            # No confident pixel — return the zero-vector sentinel
            # (same as mask_stats when mask.max() <= 127).
            return np.array([0.0, 0.0, 0.0, 0.0])

        confident = prob_vals >= 128
        avg_prob = float(prob_vals[confident].mean() / 256)
        avg_r    = float(image_255[pixels_in, 0][confident].mean() / 256)
        avg_g    = float(image_255[pixels_in, 1][confident].mean() / 256)
        avg_b    = float(image_255[pixels_in, 2][confident].mean() / 256)

        return np.array([avg_prob, avg_r, avg_g, avg_b])
    
    def compute_plausibility_score(self, features: np.ndarray, 
                                   prior_params: List[List[float]]) -> float:
        """
        Compute the anatomical plausibility score (in log-space) for a component.

        Implements Eq. (3) from the paper in log-space to avoid numerical
        overflow/underflow caused by the large alpha/beta values in the priors:

            log S(p_i) = sum_{k=1}^{4} [
                (alpha_k - 1) * log(f_k) +
                (beta_k  - 1) * log(1 - f_k) -
                log B(alpha_k, beta_k)
            ]

        Using scipy.special.betaln (= log B) keeps the computation stable even
        when alpha > 1000.  A higher log-score means the component is more
        plausible under the learned anatomical prior.
        """
        log_score = 0.0

        for k in range(4):
            alpha, beta_val = prior_params[k]
            f_k = float(np.clip(features[k], 1e-10, 1 - 1e-10))

            try:
                log_pdf = (
                    (alpha   - 1) * np.log(f_k)
                  + (beta_val - 1) * np.log(1.0 - f_k)
                  - betaln(alpha, beta_val)
                )
                log_score += log_pdf
            except (ValueError, FloatingPointError) as e:
                warnings.warn(f"Numerical issue in log-Beta PDF computation: {e}")
                return -np.inf

        return log_score
    
    def refine_prediction(self,
                         initial_prediction: np.ndarray,
                         prob_map_255: np.ndarray,
                         image_255: np.ndarray,
                         modality: str,
                         object_type: str,
                         min_component_size: int = 10) -> Tuple[np.ndarray, Dict]:
        """
        Refine the initial binary prediction by removing anatomically implausible
        connected components, following Eq. (3)–(4) of the paper.

        Parameters
        ----------
        initial_prediction : bool ndarray
            Binary prediction mask (True = foreground).
        prob_map_255 : float ndarray, values in [0, 255]
            Softmax probability map scaled by 255 (= raw_prob * 255).
        image_255 : float ndarray (*spatial, 3), values in [0, 255]
            RGB image (grayscale triplicated by the caller).
        modality : str
            Key in target_dist.json, e.g. ``"CT-Abdomen"``.
        object_type : str
            Target name to look up inside the modality, e.g. ``"liver"``.
        min_component_size : int
            Connected components smaller than this are always discarded.

        Plausibility scores are computed in log-space (see compute_plausibility_score).
        The retention threshold is τ = μ_S − 2σ_S, where μ_S and σ_S are the mean
        and standard deviation of the log-scores over all N components.
        """
        stats = {
            'total_components': 0,
            'kept_components': 0,
            'prior_found': False,
            'matched_category': None,
            'threshold_used': 0.0,
            'mean_score': 0.0,
            'std_score': 0.0
        }
        
        prior_params = self.get_prior_params(modality, object_type)
        if prior_params is None:
            stats['prior_found'] = False
            return initial_prediction, stats
        
        stats['prior_found'] = True
        stats['matched_category'] = self.find_matching_category(object_type, modality)
        
        labeled_array, num_components = ndimage.label(initial_prediction)
        stats['total_components'] = num_components
        
        if num_components == 0:
            return initial_prediction, stats
        
        log_scores = []
        component_info = []
        
        for i in range(1, num_components + 1):
            component_mask = (labeled_array == i)
            component_size = np.sum(component_mask)
            
            if component_size < min_component_size:
                continue
            
            features = self.extract_features(component_mask, prob_map_255, image_255)
            log_score = self.compute_plausibility_score(features, prior_params)
            
            log_scores.append(log_score)
            component_info.append((i, component_mask, log_score))
        
        if len(log_scores) == 0:
            return np.zeros_like(initial_prediction), stats
        
        log_scores_arr = np.array(log_scores)

        # Filter out degenerate (-inf) scores before computing statistics so that
        # a single numerical failure does not corrupt the threshold (Eq. 4).
        finite_mask = np.isfinite(log_scores_arr)
        if finite_mask.sum() == 0:
            # All scores are degenerate; keep original prediction unchanged.
            return initial_prediction, stats

        mean_score = float(np.mean(log_scores_arr[finite_mask]))
        std_score  = float(np.std(log_scores_arr[finite_mask]))

        # τ = μ_S − 2σ_S  (paper Eq. 4)
        threshold = mean_score - 2 * std_score
        
        stats['mean_score']     = mean_score
        stats['std_score']      = std_score
        stats['threshold_used'] = threshold
        
        refined_prediction = np.zeros_like(initial_prediction)
        kept_components = 0
        
        for comp_idx, component_mask, log_score in component_info:
            if log_score >= threshold:
                refined_prediction[component_mask] = 1
                kept_components += 1
        
        stats['kept_components'] = kept_components
        
        return refined_prediction, stats
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         image_dir: str,
                         modality: str,
                         object_mapping: Dict[int, str],
                         min_component_size: int = 10,
                         pattern: str = "*.npz"):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        image_path = Path(image_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        pred_files = sorted(list(input_path.glob(pattern)))
        
        if len(pred_files) == 0:
            tqdm.write(f"No files found matching pattern '{pattern}' in {input_dir}")
            return
        
        tqdm.write(f"Found {len(pred_files)} files to process")
        tqdm.write(f"Modality: {modality}")
        tqdm.write(f"Object mapping: {object_mapping}")
        tqdm.write("-" * 80)
        
        for pred_file in tqdm(pred_files, desc="Processing files"):
            try:
                data = np.load(pred_file)
                if 'probabilities' not in data:
                    tqdm.write(f"Skipping {pred_file.name}: 'probabilities' key not found")
                    continue
                
                probabilities = data['probabilities']

                # Ensure probabilities has an explicit channel axis: (C, *spatial).
                # nnUNet 2D saves (C, H, W) → ndim 3, already has channel axis.
                # A legacy single-channel 3D volume saved as (H, W, D) → ndim 3
                # is ambiguous; we assume (C, *spatial) convention throughout.
                if probabilities.ndim < 2:
                    tqdm.write(f"Skipping {pred_file.name}: unexpected probabilities shape {probabilities.shape}")
                    continue

                num_channels = probabilities.shape[0]
                spatial_shape = probabilities.shape[1:]   # (H, W) or (H, W, D)
                
                image_file = image_path / pred_file.name.replace('.npz', '.nii.gz')
                if not image_file.exists():
                    image_file = image_path / pred_file.name.replace('.npz', '.nii')
                
                if not image_file.exists():
                    tqdm.write(f"Image not found for {pred_file.name}, skipping")
                    continue
                
                img_nii = nib.load(str(image_file))
                image_data = img_nii.get_fdata()
                
                # Build image_255 with shape (*spatial_shape, 3) so that
                # boolean indexing image_255[pixels_in, c] works for any
                # spatial dimensionality (2-D or 3-D).
                #
                # Normalise voxel range to [0, 255] to match the feature space
                # of Anatomical_Priors.json (priors were fitted with images
                # scaled to [0, 255] → divided by 256 inside extract_features).
                vmin, vmax = float(image_data.min()), float(image_data.max())
                image_norm = (image_data - vmin) / (vmax - vmin + 1e-10) * 255.0

                if image_norm.shape == spatial_shape:
                    # Grayscale: scalar-per-voxel, triplicate to fill R/G/B.
                    # shape: (*spatial_shape, 3)
                    image_255 = np.stack([image_norm] * 3, axis=-1)
                elif image_norm.shape == (*spatial_shape, 3):
                    # Already a 3-channel image in the right layout.
                    image_255 = image_norm
                elif image_norm.ndim > len(spatial_shape) and image_norm.shape[:len(spatial_shape)] == spatial_shape:
                    # Extra trailing dims (e.g. 4-channel MRI, or singleton depth).
                    # Use the first channel and triplicate.
                    slc = (slice(None),) * len(spatial_shape) + (0,)
                    image_255 = np.stack([image_norm[slc]] * 3, axis=-1)
                else:
                    warnings.warn(
                        f"{pred_file.name}: image shape {image_data.shape} does not match "
                        f"probability spatial shape {spatial_shape}. Skipping VPR for this file."
                    )
                    np.savez_compressed(str(output_path / pred_file.name), probabilities=probabilities)
                    continue

                refined_probabilities = np.zeros_like(probabilities)

                tqdm.write(f"\nProcessing: {pred_file.name}")

                for channel_idx in range(num_channels):
                    if channel_idx in object_mapping:
                        object_name = object_mapping[channel_idx]
                    else:
                        tqdm.write(f"Channel {channel_idx}: No object mapping, skipping")
                        refined_probabilities[channel_idx] = probabilities[channel_idx]
                        continue

                    prob_map   = probabilities[channel_idx]          # [0, 1]
                    prob_255   = prob_map * 255.0                    # [0, 255] — matches mask_stats scale
                    initial_pred = (prob_map > 0.5).astype(bool)

                    refined_pred, stats = self.refine_prediction(
                        initial_pred,
                        prob_255,
                        image_255,
                        modality,
                        object_name,
                        min_component_size
                    )
                    
                    refined_probabilities[channel_idx] = prob_map * refined_pred
                    
                    if stats['prior_found']:
                        tqdm.write(f"Channel {channel_idx} ({object_name} → {stats['matched_category']}): "
                                 f"{stats['kept_components']}/{stats['total_components']} components kept "
                                 f"(threshold={stats['threshold_used']:.2e}, μ={stats['mean_score']:.2e}, σ={stats['std_score']:.2e})")
                    else:
                        tqdm.write(f"Channel {channel_idx} ({object_name}): No prior found, kept original")
                        refined_probabilities[channel_idx] = probabilities[channel_idx]
                
                output_file = output_path / pred_file.name
                np.savez_compressed(str(output_file), probabilities=refined_probabilities)
                tqdm.write(f" Saved to: {output_file.name}\n")
                
            except Exception as e:
                tqdm.write(f"Error processing {pred_file.name}: {str(e)}\n")
                continue
        
        tqdm.write("-" * 80)
        tqdm.write("Processing complete!")


if __name__ == "__main__":
    refiner = PriorDistributionRefiner(prior_json_path="Anatomical_Priors.json")
    object_mapping = {
        0: "",
        1: "",
        2: "",
        3: "",
    }
    
    refiner.process_directory(
        input_dir="predictions",
        output_dir="refined_predictions",
        image_dir="images",
        modality="CT-Abdomen",
        object_mapping=object_mapping,
        min_component_size=20,
        pattern="*.npz"
    )