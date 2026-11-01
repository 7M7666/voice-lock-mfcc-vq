import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

import srdp
from dtw_paths import alignment, path_statistics


class AlgorithmFollowupTests(unittest.TestCase):
    def test_local_dtw_cost_preserves_a_bad_interval_despite_repeated_cheap_frames(self):
        from live_structure_experiments import local_costs
        template = np.zeros((4, 1))
        sample = np.array([[0.], [0.], [0.], [8.], [0.], [0.], [0.]])
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 1], [4, 2], [5, 3], [6, 3]])
        np.testing.assert_array_equal(local_costs(sample, template, path), [0., 8., 0., 0.])
        with self.assertRaises(ValueError):
            local_costs(sample, template, path[path[:, 1] != 1])

    def test_structural_and_pitch_gates_keep_original_voice_and_phrase_limits(self):
        from live_structure_experiments import fit, apply
        row = {"voice_mfcc": .7, "phrase_mfcc": .8, "local_mfcc": .6,
            "local_delta": .6, "identity_ratio": .8, "pitch_identity": .5, "pitch_relative": .5}
        for variant, extra in (("cohort_local_fusion_fixed_1p1", "identity_ratio"),
                ("pitch_local_fusion_fixed_1p1", "pitch_identity"),
                ("pitch_relative_local_fusion_fixed_1p1", "pitch_relative")):
            policy = fit([row], variant)
            self.assertTrue(apply([row], policy)[0])
            for key, value in (("voice_mfcc", 1.000001), ("phrase_mfcc", 1.100001),
                    (extra, 1.000001), ("local_mfcc", 1.5)):
                self.assertFalse(apply([{**row, key: value}], policy)[0])
            self.assertFalse(apply([{**row, "input_rejected": True}], policy)[0])

    def test_structural_folds_exclude_both_claims_and_do_not_fit_held_values(self):
        from live_structure_experiments import grouped_predictions
        rows = []
        for group in ("A", "B", "C"):
            for genuine in (True, False):
                value = .6 if genuine else 1.2
                rows.append({"audio_sha256": group, "genuine": genuine,
                    "voice_mfcc": .8, "phrase_mfcc": value, "phrase_delta": value,
                    "phrase_spectral": value, "local_mfcc": value, "local_delta": value,
                    "identity_ratio": .8, "pitch_identity": .8, "pitch_relative": .8})
        _, folds = grouped_predictions(rows, "pitch_best_single_1p1")
        changed = [{**r, "local_mfcc": 999., "phrase_mfcc": 999.} if r["audio_sha256"] == "A" else r for r in rows]
        _, altered = grouped_predictions(changed, "pitch_best_single_1p1")
        self.assertEqual(folds[0], altered[0])
        for fold in folds:
            self.assertNotIn(fold["held_audio"], fold["fit_audio"])
            self.assertEqual(len(fold["fit_audio"]), 2)

    def test_live_cohort_rejects_incompatible_feature_coordinates(self):
        from live_structure_experiments import check_coordinates
        base = srdp.load_model()
        check_coordinates({"A": base, "B": replace(base)})
        with self.assertRaises(ValueError):
            check_coordinates({"A": base, "B": replace(base, zscore_mean=base.zscore_mean + .01)})

    def test_pitch_failure_closes_pitch_gate_but_preserves_original_control(self):
        from live_structure_experiments import fit, apply
        row = {"voice_mfcc": .7, "phrase_mfcc": .8, "phrase_delta": .8,
            "phrase_spectral": .8, "pitch_identity": float("inf"), "genuine": True}
        self.assertTrue(apply([row], fit([row, {**row, "genuine": False}], "two_layer"))[0])
        self.assertFalse(apply([row], fit([row], "pitch_only_strict"))[0])

    def test_score_dominance_audit_never_compares_different_target_accounts(self):
        from live_structure_experiments import score_audit, SINGLES
        positive = {**dict.fromkeys(SINGLES, .9), "genuine": True, "target": "A", "speaker": "A", "sequence": 1, "kind": "self_correct"}
        other_target = {**dict.fromkeys(SINGLES, .8), "genuine": False, "target": "B", "speaker": "A", "sequence": 1, "kind": "other_correct"}
        audit, _, _ = score_audit([positive, other_target])
        self.assertEqual(audit[0]["dominating_negatives"], [])

    def test_pitch_descriptor_measures_frequency_and_ignores_quiet_edges(self):
        from live_structure_experiments import pitch_descriptor
        model = srdp.load_model()
        with tempfile.TemporaryDirectory() as directory:
            for hz in (150., 250.):
                t = np.arange(16000) / 16000
                signal = .2 * np.sin(2 * np.pi * hz * t)
                path = Path(directory) / f"tone_{hz}.wav"
                sf.write(path, np.r_[np.zeros(8000), signal, np.zeros(8000)], 16000)
                descriptor = pitch_descriptor(path, model)
                self.assertAlmostEqual(descriptor["median_hz"], hz, delta=2.)
                self.assertGreater(descriptor["voiced_frames"], 8)

    def test_live_feature_fusion_uses_complementary_scores_without_bypassing_voice(self):
        from live_fusion_experiments import fit_policy, apply_policy
        scores = np.array([[.65, .95, .5, 1.], [.95, .65, .5, 1.], [.9, .9, 1.2, 1.], [.65, .95, 1.3, 1.]])
        labels = [True, True, False, False]
        policy = fit_policy(scores, labels, "feature_fusion_1p1")
        np.testing.assert_array_equal(apply_policy(scores, policy)[0], labels)
        self.assertFalse(apply_policy([[1.000001, .1, .1, .1]], policy)[0][0])
        self.assertFalse(apply_policy([[.1, 1.100001, .1, .1]], policy)[0][0])

    def test_live_fusion_excludes_both_account_claims_of_held_wav(self):
        from live_fusion_experiments import grouped_predictions
        scores = np.array([[.5, .6, .4, .5], [1.2, 1.3, 1.1, 1.2], [.6, .5, .5, .4], [1.3, 1.2, 1.2, 1.1], [.9, 1.1, 1.2, 1.], [1.1, 1.1, 1.1, 1.1]])
        labels = [True, False, True, False, False, False]
        groups = ["A", "A", "B", "B", "C", "C"]
        _, folds = grouped_predictions(scores, labels, groups, "weighted_1p1")
        changed = scores.copy()
        changed[:2] = 1000.
        _, changed_folds = grouped_predictions(changed, labels, groups, "weighted_1p1")
        self.assertEqual(folds[0], changed_folds[0])
        for fold in folds:
            self.assertNotIn(fold["held_audio"], fold["fit_audio"])
            self.assertEqual(len(fold["fit_audio"]), 2)

    def test_phrase_cmn_removes_constant_offset_without_mutating_voice_features(self):
        from phrase_content_experiments import phrase_matrix
        model = srdp.load_model()
        rng = np.random.default_rng(52)
        matrix = rng.normal(size=(25, 31)).astype(np.float32)
        original = matrix.copy()
        shifted = matrix + rng.normal(size=(1, 31)).astype(np.float32)
        np.testing.assert_allclose(phrase_matrix(matrix, model, True), phrase_matrix(shifted, model, True), atol=1e-6)
        np.testing.assert_array_equal(matrix, original)
        self.assertGreater(np.linalg.norm(phrase_matrix(matrix, model, False) - phrase_matrix(shifted, model, False)), 1.)

    def test_shared_phrase_calibration_excludes_entire_held_speaker(self):
        from phrase_content_experiments import calibration_references
        owners = ["A", "A", "B", "B", "C", "C"]
        for held in range(len(owners)):
            indices = calibration_references(owners, held, True)
            self.assertEqual(len(indices), 4)
            self.assertTrue(all(owners[i] != owners[held] for i in indices))
            self.assertNotIn(held, calibration_references(owners, held, False))

    def test_phrase_cmn_control_keeps_formal_and_app_threshold_rules_separate(self):
        from phrase_content_experiments import calibrate, formal_profile
        model = formal_profile(srdp.load_model(), "S01")
        rng = np.random.default_rng(62)
        matrices = [rng.normal(size=(15, 12)).astype(np.float32) * scale for scale in (1., 1., 1., 3.)]
        formal, rows = calibrate(matrices, ["A"] * 4, model, False, formal=True)
        app, _ = calibrate(matrices, ["A"] * 4, model, False)
        values = np.array([row["distance"] for row in rows])
        self.assertAlmostEqual(formal, values.mean() + values.std())
        self.assertAlmostEqual(app, max(values.mean() + 2 * values.std(), values.max() + 1e-6))

    def test_experimental_phrase_pass_cannot_bypass_frozen_voice_gate(self):
        from phrase_content_experiments import PhraseCandidate, formal_profile, evaluate
        model = formal_profile(srdp.load_model(), "S01")
        matrix = np.zeros((15, 31), dtype=np.float32)
        candidate = PhraseCandidate([matrix[:, :12]], 1., False, "min", 1., 0., 0)
        result = evaluate(candidate, model, {"voice_score": "1.000001"}, matrix)
        self.assertEqual(result["phrase_passed"], "True")
        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(result["voice_score"], "1.000001")
        rounded = evaluate(candidate, model, {"voice_score": "1.000000", "voice_passed": "False"}, matrix)
        self.assertEqual(rounded["decision"], "REJECT")

    def test_unknown_phrase_trials_are_excluded_from_error_counts(self):
        from phrase_content_experiments import summarize
        result = summarize([{"scope": "live", "variant": "personal_mfcc", "kind": "unknown",
            "expected_decision": "", "decision": "REJECT", "phrase_passed": "False"}])[0]
        self.assertEqual(result["labeled_count"], 0)
        self.assertEqual(result["false_reject"], 0)
        self.assertEqual(result["false_accept"], 0)

    def test_rejection_reason_reports_both_computed_branches(self):
        reason = srdp.format_reject_reason({"decision": "REJECT", "voice_accepted": "False",
            "passphrase_score": "1.2", "passphrase_passed": "False"})
        self.assertIn("声纹距离", reason)
        self.assertIn("口令模板DTW距离", reason)

    def test_stationary_low_level_noise_is_rejected_before_feature_scoring(self):
        rng = np.random.default_rng(20261031)
        white = rng.normal(0, .0015, 48000).astype(np.float32)
        colored = np.convolve(white, np.ones(9) / 9, mode="same").astype(np.float32)
        for samples in (white, colored):
            with self.assertRaises(srdp.NoSpeechError):
                srdp.speech_region(samples, 16000, "trim")

    def test_dtw_path_matches_exhaustive_minimum_cost_under_run_constraint(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(3, 2)).astype(np.float32)
        y = rng.normal(size=(5, 2)).astype(np.float32)
        local = np.stack([np.linalg.norm(y - frame, axis=1) for frame in x])
        for limit in (1, 2, 3):
            candidates = []

            def visit(i, j, path, previous=None, run=0):
                if (i, j) == (len(x) - 1, len(y) - 1):
                    total = sum(float(local[a, b]) for a, b in path)
                    candidates.append((total, path))
                    return
                for direction in ((1, 0), (0, 1), (1, 1)):
                    a, b = i + direction[0], j + direction[1]
                    next_run = run + 1 if direction == previous else 1
                    if direction == (1, 1):
                        next_run = 0
                    if a < len(x) and b < len(y) and next_run <= limit:
                        visit(a, b, path + [(a, b)], direction, next_run)

            visit(0, 0, [(0, 0)])
            total, expected_path = min(candidates, key=lambda item: item[0])
            distance, path = alignment(x, y, max_run=limit)
            self.assertAlmostEqual(distance, total / len(expected_path), places=10)
            self.assertLessEqual(path_statistics(path, len(x), len(y))["max_axis_run"], limit)
            np.testing.assert_array_equal(path[0], [0, 0])
            np.testing.assert_array_equal(path[-1], [len(x) - 1, len(y) - 1])

    def test_impossible_dtw_alignment_is_rejected(self):
        distance, path = alignment(np.zeros((2, 1)), np.zeros((12, 1)), max_run=2)
        self.assertTrue(np.isinf(distance))
        self.assertEqual(len(path), 0)

    def test_bounded_fusion_can_rescue_voice_rejection_with_a_phrase_guard(self):
        self.assertFalse(srdp.verification_decision(1.05, .3, "fusion", .5, 1.)[0])
        self.assertTrue(srdp.verification_decision(1.05, .3, "bounded_fusion", .5, 1., voice_guard=1.1)[0])
        self.assertFalse(srdp.verification_decision(.1, 1.01, "bounded_fusion", .5, 1., voice_guard=1.1)[0])
        self.assertFalse(srdp.verification_decision(1.11, .1, "bounded_fusion", .5, 1., voice_guard=1.1)[0])

    def test_bounded_fusion_evaluates_phrase_even_when_voice_is_above_one(self):
        base = srdp.load_model()
        matrix = np.ones((9, 31), dtype=np.float32)
        distance = float(np.sqrt(12))
        model = replace(base, codebooks={"A": np.zeros((1, 12), dtype=np.float32)}, thresholds={"A": distance / 1.05}, passphrase_thresholds={"A": 1.}, passphrase_train_by_speaker={"A": [matrix[:, :12]]}, feature_set="mfcc", passphrase_feature_set="mfcc", decision_mode="bounded_fusion", fusion_voice_weight=.5, fusion_threshold=1., voice_guard=1.1)
        result = srdp.predict_feature_matrix(matrix, model)
        self.assertEqual(result["voice_accepted"], "False")
        self.assertEqual(result["decision"], "A")
        self.assertEqual(result["reject_reason"], "")

    def test_silence_dc_and_frame_noise_are_explicit_audio_rejections(self):
        rng = np.random.default_rng(34)
        for samples in (np.zeros(48000), np.full(48000, .05)):
            for mode in ("trim", "frame"):
                with self.assertRaises(srdp.NoSpeechError):
                    srdp.speech_region(samples.astype(np.float32), 16000, mode)
        with self.assertRaises(srdp.NoSpeechError):
            srdp.speech_region(rng.normal(0, .03, 48000).astype(np.float32), 16000, "frame")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            sf.write(path, np.zeros(48000), 16000)
            result = srdp.predict_one(path, metadata={"expected_decision": "REJECT"})
            self.assertEqual(result["decision"], "REJECT")
            self.assertEqual(result["correct"], "True")
            self.assertIn("静音", result["reject_reason"])

    def test_cmn_cmvn_preserve_other_features_and_recompute_deltas(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(20, 31)).astype(np.float32)
        for mode in ("cmn", "cmvn"):
            result = srdp.normalize_cepstra(raw, mode)
            np.testing.assert_allclose(result[:, :12].mean(axis=0), 0., atol=1e-6)
            np.testing.assert_array_equal(result[:, 24:], raw[:, 24:])
            np.testing.assert_allclose(result[:, 12:24], srdp.librosa.feature.delta(result[:, :12].T).T)
            if mode == "cmvn":
                np.testing.assert_allclose(result[:, :12].std(axis=0), 1., atol=1e-6)
        np.testing.assert_array_equal(raw, srdp.normalize_cepstra(raw, "none"))

    def test_new_model_settings_survive_standalone_and_user_model_roundtrip(self):
        import gui
        model = replace(srdp.load_model(), vad_mode="frame", cepstral_normalization="cmn", dtw_max_run=3, voice_guard=1.1, passphrase_guard=1.02)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            srdp.save_calibrated_model(path, model)
            restored = srdp.load_model(path, Path(directory) / "missing_features.npz")
            for field in ("vad_mode", "cepstral_normalization", "dtw_max_run", "voice_guard", "passphrase_guard"):
                self.assertEqual(getattr(restored, field), getattr(model, field))
            samples = [srdp.extract_one_audio_features(srdp.PROJECT_ROOT / f"data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_00{i}.wav", model).features for i in (1, 2, 3)]
            profile = gui.build_session_enrollment_model("test", restored, samples)
            path = Path(directory) / "user.npz"
            gui.save_user_model(profile, path)
            loaded = gui.load_user_model(path)
            for field in ("vad_mode", "cepstral_normalization", "dtw_max_run", "voice_guard", "passphrase_guard"):
                self.assertEqual(getattr(loaded, field), getattr(model, field))
            silence = Path(directory) / "silence.wav"
            sf.write(silence, np.zeros(48000), 16000)
            self.assertEqual(gui.predict_with_session_model(silence, loaded)["decision"], "REJECT")

    def test_existing_calibrated_evaluation_decisions_are_preserved(self):
        _, rows, features = srdp.load_feature_archive(srdp.FEATURE_DIR / "srdp_features.npz")
        model = srdp.load_model("results/calibration_experiments/loo_voice/model.npz")
        reference = {row["file_path"]: row for row in srdp.read_csv_rows(srdp.RESULTS_DIR / "calibration_experiments/loo_voice/recognition_results.csv")}
        count = 0
        for row, matrix in zip(rows, features):
            if row["split"] == "train":
                continue
            actual = srdp.predict_feature_matrix(matrix, model, row, quality_score=float(row["quality_score"]))
            expected = reference[row["file_path"]]
            self.assertEqual(actual["decision"], expected["decision"])
            for key in ("voice_score", "passphrase_score", "fusion_score"):
                if expected[key]:
                    self.assertAlmostEqual(float(actual[key]), float(expected[key]), delta=2e-6)
            count += 1
        self.assertEqual(count, 63)


if __name__ == "__main__":
    unittest.main()
