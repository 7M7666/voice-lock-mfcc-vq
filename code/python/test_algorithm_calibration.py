import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

import srdp


class AlgorithmCalibrationTests(unittest.TestCase):
    def test_new_registration_uses_the_calibrated_phrase_threshold_once(self):
        import gui

        base = srdp.load_model()
        recordings = [
            srdp.extract_one_audio_features(srdp.PROJECT_ROOT / f"data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_00{index}.wav", base).features
            for index in (1, 2, 3)
        ]
        profile = gui.build_session_enrollment_model("new_user", base, recordings)
        self.assertEqual(profile.passphrase_accept_score, 1.0)
        held_out = [
            srdp.passphrase_template_distance(matrix, profile.passphrase_templates, profile.passphrase_aggregate, exclude_index=index)
            for index, matrix in enumerate(profile.passphrase_templates)
        ]
        self.assertTrue(all(distance <= profile.passphrase_threshold * profile.passphrase_accept_score for distance in held_out))

    def test_lloyd_refines_until_cluster_assignments_stabilise(self):
        samples = np.array([[0.0], [2.0], [3.0], [10.0], [11.0], [12.0]], dtype=np.float32)
        initial = np.array([[0.0], [2.0]], dtype=np.float32)
        centers = srdp.lloyd_refine(samples, initial, max_iter=80, tol=1e-5)
        np.testing.assert_allclose(centers.ravel(), [5.0 / 3.0, 11.0], rtol=1e-6)

    def test_voice_calibration_excludes_the_entire_held_out_recording(self):
        recordings = [np.full((2, 1), value, dtype=np.float32) for value in (0.0, 2.0, 10.0)]
        distances = srdp.leave_one_out_voice_distances(recordings, codebook_size=1)
        np.testing.assert_allclose(distances, [6.0, 3.0, 9.0])

    def test_calibration_trials_ignore_evaluation_audio_and_labels(self):
        names = [srdp.FEATURE_NAMES[0]]
        rows, features = [], []
        for speaker, offset in (("A", 0.0), ("B", 10.0)):
            for index in range(3):
                rows.append({"speaker_id": speaker, "split": "train", "file_path": f"{speaker}{index}.wav"})
                features.append(np.array([[offset + index], [offset + index + 0.2]], dtype=np.float32))
        expected = srdp.collect_calibration_trials(names, rows, features, codebook_size=1)
        rows.append({"speaker_id": "A", "split": "test", "file_path": "never_fit.wav"})
        features.append(np.full((10, 1), 1e6, dtype=np.float32))
        actual = srdp.collect_calibration_trials(names, rows, features, codebook_size=1)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 12)
        self.assertEqual(sum(row["genuine"] for row in actual), 6)

    def test_calibrated_fusion_rejects_a_trial_that_passes_both_gates(self):
        trials = [
            {"genuine": True, "voice_score": 0.6, "passphrase_score": 0.9},
            {"genuine": True, "voice_score": 0.9, "passphrase_score": 0.6},
            {"genuine": False, "voice_score": 0.9, "passphrase_score": 0.9},
        ]
        policy = srdp.fit_score_fusion(trials)
        self.assertLess(policy["fusion_threshold"], 1.0)
        self.assertEqual(policy["calibration_false_accept"], 0)
        self.assertEqual(policy["calibration_false_reject"], 0)
        rows = [dict(row, pred_speaker="A") for row in trials]
        self.assertEqual(
            srdp.fusion_decisions(rows, policy["fusion_voice_weight"], policy["fusion_threshold"], 0.0),
            ["A", "A", "REJECT"],
        )

    def test_fusion_does_not_claim_a_gain_when_the_gates_already_separate_trials(self):
        policy = srdp.fit_score_fusion([
            {"genuine": True, "voice_score": 0.6, "passphrase_score": 0.6},
            {"genuine": False, "voice_score": 1.2, "passphrase_score": 0.1},
        ])
        self.assertEqual(policy["fusion_threshold"], 1.0)
        self.assertEqual(policy["changed_from_two_layer"], 0)

    def test_fusion_prefers_rejection_when_positive_and_negative_scores_overlap(self):
        policy = srdp.fit_score_fusion([
            {"genuine": True, "voice_score": 0.7, "passphrase_score": 0.7},
            {"genuine": False, "voice_score": 0.7, "passphrase_score": 0.7},
        ])
        self.assertEqual(policy["calibration_false_accept"], 0)
        self.assertEqual(policy["calibration_false_reject"], 1)

    def test_cohort_scoring_compares_other_users_and_is_scale_invariant(self):
        expected = {"A": 0.5, "B": 2.0}
        self.assertEqual(srdp.speaker_distance_scores({"A": 2.0, "B": 4.0}, "cohort_ratio"), expected)
        self.assertEqual(srdp.speaker_distance_scores({"A": 6.0, "B": 12.0}, "cohort_ratio"), expected)
        with self.assertRaises(ValueError):
            srdp.speaker_distance_scores({"A": 2.0}, "cohort_ratio")

    def test_shared_decision_keeps_identity_and_phrase_guards_and_uses_fusion(self):
        self.assertTrue(srdp.verification_decision(0.6, 0.6, "fusion", 0.5, 0.8)[0])
        self.assertFalse(srdp.verification_decision(0.9, 0.9, "fusion", 0.5, 0.8)[0])
        self.assertTrue(srdp.verification_decision(0.9, 0.9, "two_layer", 0.5, 0.8)[0])
        self.assertFalse(srdp.verification_decision(1.1, 0.1, "fusion", 0.5, 0.8)[0])
        self.assertFalse(srdp.verification_decision(0.1, 1.1, "fusion", 0.5, 0.8)[0])

    def test_calibrated_model_keeps_parameters_without_external_feature_archive(self):
        model = replace(srdp.load_model(), voice_metric="cohort_ratio", calibration_mode="leave_one_recording_out")
        with tempfile.TemporaryDirectory(prefix="srdp_calibration_model_") as directory:
            path = Path(directory) / "model.npz"
            srdp.save_calibrated_model(path, model)
            restored = srdp.load_model(path, Path(directory) / "missing_features.npz")
            self.assertEqual(restored.voice_metric, model.voice_metric)
            self.assertEqual(restored.calibration_mode, model.calibration_mode)
            self.assertEqual(restored.thresholds, model.thresholds)
            np.testing.assert_array_equal(restored.zscore_mean, model.zscore_mean)
            for speaker in model.codebooks:
                np.testing.assert_array_equal(restored.codebooks[speaker], model.codebooks[speaker])
                for left, right in zip(restored.passphrase_train_by_speaker[speaker], model.passphrase_train_by_speaker[speaker]):
                    np.testing.assert_array_equal(left, right)

    def test_app_profile_keeps_calibrated_fusion_and_background_codebooks(self):
        import gui

        base = replace(srdp.load_model(), voice_metric="cohort_ratio", calibration_mode="leave_one_recording_out", decision_mode="fusion", fusion_threshold=0.9)
        recordings = [
            srdp.extract_one_audio_features(srdp.PROJECT_ROOT / f"data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_00{index}.wav", base).features
            for index in (1, 2, 3)
        ]
        profile = gui.build_session_enrollment_model("S01", base, recordings)
        sample = srdp.PROJECT_ROOT / "data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav"
        expected = gui.predict_with_session_model_baseline(sample, profile)
        with tempfile.TemporaryDirectory(prefix="srdp_calibrated_user_") as directory:
            path = Path(directory) / "user.npz"
            gui.save_user_model(profile, path)
            restored = gui.load_user_model(path)
            self.assertEqual(restored.decision_mode, "fusion")
            self.assertEqual(restored.voice_metric, "cohort_ratio")
            self.assertEqual(restored.passphrase_accept_score, 1.0)
            self.assertEqual(set(restored.background_codebooks), {"S02", "S03", "S04"})
            self.assertEqual(gui.predict_with_session_model_baseline(sample, restored), expected)


if __name__ == "__main__":
    unittest.main()
