import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apply_vpr import dice, evaluate, main as apply_vpr_main  # noqa: E402
from vpr import VPRError, VisualPlausibilityRefiner, load_priors  # noqa: E402


# First feature is average prediction probability, so the synthetic prior is
# biased toward 1 like BiomedParse's real target_dist entries.
PRIORS = {"test-modality": {"organ": [[5, 1], [5, 5], [5, 5], [5, 5]]}}


def _labels_and_image(good_positions, bad_positions):
    labels = np.zeros((60, 20, 20), dtype=np.int64)
    image = np.zeros((60, 20, 20), dtype=float)
    for d, h, w in good_positions:
        labels[d : d + 2, h : h + 2, w : w + 2] = 1
        image[d : d + 2, h : h + 2, w : w + 2] = 0.5
    for d, h, w in bad_positions:
        labels[d : d + 2, h : h + 2, w : w + 2] = 1
        image[d : d + 2, h : h + 2, w : w + 2] = 0.02
    return labels, image


class VPRRefinerTests(unittest.TestCase):
    def test_mu_two_sigma_removes_low_intensity_components(self):
        good = [(i * 4, 0, 0) for i in range(5)] + [
            (i * 4, 4, 0) for i in range(5)
        ]
        bad = [(0, 8, 8), (4, 8, 8)]
        labels, image = _labels_and_image(good, bad)
        refiner = VisualPlausibilityRefiner(PRIORS, "test-modality")
        refined, stats = refiner.refine(
            labels,
            image=image,
            label_targets={1: "organ"},
            probabilities={1: np.ones_like(labels, dtype=float)},
        )
        self.assertEqual(stats["component_count"], 12)
        self.assertEqual(stats["removed_components"], 2)
        self.assertTrue(all(not item["keep"] for item in stats["components"] if item["features"][1] < 0.1))
        # Bad voxels are now background.
        self.assertEqual(np.count_nonzero(refined[0:2, 8:10, 8:10]), 0)
        self.assertEqual(np.count_nonzero(refined[0:2, 0:2, 0:2]), 8)

    def test_missing_prior_keeps_prediction(self):
        labels, image = _labels_and_image([(0, 0, 0)], [(4, 4, 4), (8, 8, 8)])
        refiner = VisualPlausibilityRefiner(PRIORS, "test-modality")
        refined, stats = refiner.refine(
            labels, image=image, label_targets={1: "not-in-prior"}
        )
        self.assertIsNone(stats["threshold_log_score"])
        self.assertEqual(stats["removed_components"], 0)
        np.testing.assert_array_equal(refined, labels)

    def test_requires_probability_map_for_scored_target(self):
        labels, image = _labels_and_image([(0, 0, 0)], [])
        refiner = VisualPlausibilityRefiner(PRIORS, "test-modality")
        with self.assertRaisesRegex(VPRError, "requires a probability map"):
            refiner.refine(labels, image=image, label_targets={1: "organ"})

    def test_threshold_matches_raw_score_mean_minus_two_sigma(self):
        log_scores = [-8.0, -5.0, -5.0, -5.0, -5.0]
        raw_scores = np.exp(log_scores)
        expected = float(
            np.log(np.mean(raw_scores) - 2.0 * np.std(raw_scores))
        )
        actual = VisualPlausibilityRefiner._paper_threshold_logscore(
            log_scores, 2.0
        )
        self.assertAlmostEqual(actual, expected)

    def test_threshold_is_stable_for_large_negative_log_scores(self):
        log_scores = [-8.0, -5.0, -5.0, -5.0, -5.0]
        shifted = [score - 10000.0 for score in log_scores]
        base = VisualPlausibilityRefiner._paper_threshold_logscore(log_scores, 2.0)
        actual = VisualPlausibilityRefiner._paper_threshold_logscore(
            shifted, 2.0
        )
        self.assertAlmostEqual(actual, base - 10000.0)

    def test_threshold_is_global_across_target_labels(self):
        labels = np.zeros((60, 20, 20), dtype=np.int64)
        image = np.zeros(labels.shape, dtype=float)
        probabilities = np.ones(labels.shape, dtype=float)
        for index in range(10):
            d = index * 4
            labels[d : d + 2, 0:2, 0:2] = 2
            image[d : d + 2, 0:2, 0:2] = 0.5
        labels[0:2, 8:10, 8:10] = 1
        image[0:2, 8:10, 8:10] = 0.02
        probabilities[0:2, 8:10, 8:10] = 0.02
        refiner = VisualPlausibilityRefiner(PRIORS, "test-modality")
        refined, stats = refiner.refine(
            labels,
            image=image,
            label_targets={1: "organ", 2: "organ"},
            probabilities={1: probabilities, 2: probabilities},
        )
        self.assertEqual(stats["threshold_scope"], "all_scored_components")
        self.assertEqual(stats["removed_components"], 1)
        self.assertEqual(np.count_nonzero(refined[0:2, 8:10, 8:10]), 0)
        self.assertEqual(np.count_nonzero(refined[0:2, 0:2, 0:2]), 8)

    def test_probability_feature_can_filter_component(self):
        good = [(i * 4, 0, 0) for i in range(10)]
        labels, image = _labels_and_image(good, [])
        probabilities = np.ones_like(labels, dtype=float)
        probabilities[0:2, 0:2, 0:2] = 0.02
        refiner = VisualPlausibilityRefiner(PRIORS, "test-modality")
        refined, stats = refiner.refine(
            labels,
            image=image,
            label_targets={1: "organ"},
            probabilities={1: probabilities},
        )
        self.assertEqual(stats["removed_components"], 1)
        self.assertEqual(np.count_nonzero(refined[0:2, 0:2, 0:2]), 0)

    def test_uint8_image_is_normalized_like_biomedparse(self):
        labels, _ = _labels_and_image([(0, 0, 0)], [(4, 4, 4)])
        image = np.full(labels.shape, 128, dtype=np.uint8)
        refiner = VisualPlausibilityRefiner(PRIORS, "test-modality")
        _, stats = refiner.refine(
            labels,
            image=image,
            label_targets={1: "organ"},
            probabilities={1: np.ones_like(labels, dtype=float)},
        )
        self.assertAlmostEqual(stats["components"][0]["features"][1], 128 / 256)


class ApplyVPRTests(unittest.TestCase):
    def test_evaluation_scores_all_targets_and_empty_masks(self):
        labels = np.zeros((4, 4, 4), dtype=np.int64)
        gt = np.zeros_like(labels)
        labels[0, 0, 0] = 2
        self.assertEqual(dice(labels == 1, gt == 1), 1.0)
        result = evaluate(labels, gt, [1, 2])
        self.assertEqual(result["per_class"], {"class_1": 1.0, "class_2": 0.0})
        self.assertEqual(result["mean_dice"], 0.5)

    def test_cli_refines_prediction_and_evaluates_dice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors_path = root / "priors.json"
            priors_path.write_text(json.dumps(PRIORS), encoding="utf-8")
            pred_dir = root / "pred"
            prob_dir = pred_dir / "probability_maps"
            image_dir = root / "images"
            out_dir = root / "out"
            pred_dir.mkdir()
            prob_dir.mkdir()
            image_dir.mkdir()

            good = [(i * 4, 0, 0) for i in range(5)] + [(i * 4, 4, 0) for i in range(5)]
            bad = [(0, 8, 8), (4, 8, 8)]
            labels, image = _labels_and_image(good, bad)
            gt = labels.copy()
            for d, h, w in bad:
                gt[d : d + 2, h : h + 2, w : w + 2] = 0
            np.savez_compressed(pred_dir / "case.npz", segs=labels)
            np.savez_compressed(
                prob_dir / "case_prob.npz",
                probability_maps=np.stack([np.ones_like(labels, dtype=np.float16)]),
                class_ids=np.array([1], dtype=np.int16),
            )
            np.savez_compressed(image_dir / "case.npz", imgs=image, gts=gt)
            (root / "targets.json").write_text(json.dumps({"1": "organ"}), encoding="utf-8")

            rc = apply_vpr_main(
                [
                    "--pred_dir",
                    str(pred_dir),
                    "--image_dir",
                    str(image_dir),
                    "--output_dir",
                    str(out_dir),
                    "--priors",
                    str(priors_path),
                    "--modality",
                    "test-modality",
                    "--label_targets_json",
                    str(root / "targets.json"),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "case.npz").exists())
            self.assertTrue((out_dir / "vpr_stats.json").exists())
            with np.load(out_dir / "case.npz") as refined_npz:
                refined = refined_npz["segs"]
            self.assertEqual(np.count_nonzero(refined[0:2, 8:10, 8:10]), 0)
            stats = json.loads((out_dir / "vpr_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats[0]["removed_components"], 2)
            self.assertGreater(stats[0]["refined"]["mean_dice"], stats[0]["original"]["mean_dice"])

    def test_load_real_biomedparse_priors(self):
        priors = load_priors(ROOT / "assets/target_dist.json")
        self.assertIn("CT-Abdomen", priors)
        self.assertIn("MRI-T1-Gd-Brain", priors)
        self.assertEqual(len(priors["CT-Abdomen"]["liver"]), 4)


if __name__ == "__main__":
    unittest.main()
