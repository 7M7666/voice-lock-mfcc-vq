import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import gui
import srdp
from speech_frames import extract_branches, mask_for_region, recording_quality, sustained_activity
from frame_consistency_experiments import score_branches, replay_candidate


class SpeechFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = srdp.load_model()
        cls.paths = [srdp.PROJECT_ROOT / f"data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_00{i}.wav" for i in (1, 2, 3)]

    def test_legacy_extraction_still_matches_archived_features(self):
        _, rows, matrices = srdp.load_feature_archive(srdp.FEATURE_DIR / "srdp_features.npz")
        reference = {row["file_path"]: matrix for row, matrix in zip(rows, matrices)}
        for path in self.paths:
            result = srdp.extract_one_audio_features(path, self.base)
            np.testing.assert_allclose(result.features, reference[srdp.display_path(path)], atol=2e-5, rtol=1e-6)

    def test_activity_indices_include_the_original_crop_offset(self):
        mask = np.array([False, False, True, True, False, False])
        np.testing.assert_array_equal(mask_for_region(mask, .02, 3, 16000), [True, True, False])

    def test_voice_filter_does_not_remove_internal_pauses_from_phrase(self):
        rng = np.random.default_rng(43)
        sr = 16000
        raw = rng.normal(0, .001, 3 * sr).astype(np.float32)
        for start, end in ((.4, .7), (.95, 1.2)):
            times = np.arange(int(start * sr), int(end * sr))
            raw[times] += .1 * np.sin(2 * np.pi * 220 * times / sr)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_bursts.wav"
            sf.write(path, raw, sr, subtype="FLOAT")
            baseline = extract_branches(path, self.base)
            voice_only = extract_branches(path, self.base, voice_active=True)
            both = extract_branches(path, self.base, voice_active=True, phrase_span=True)
            np.testing.assert_array_equal(voice_only.phrase, baseline.phrase)
            self.assertLess(len(voice_only.voice), len(baseline.voice))
            self.assertGreater(len(both.phrase), len(both.voice))
            self.assertLess(both.phrase_start, .4)
            self.assertGreater(both.phrase_end, 1.2)
            self.assertLess(both.phrase_end, 1.4)

    def test_isolated_click_is_removed_without_discarding_short_adjacent_activity(self):
        active = np.zeros(200, dtype=bool)
        active[20:40] = True
        active[43:45] = True
        active[65:85] = True
        active[170:172] = True
        selected = sustained_activity(active, 16000)
        self.assertTrue(selected[43:45].all())
        self.assertTrue(selected[65:85].all())
        self.assertFalse(selected[170:172].any())

    def test_sustained_phrase_boundary_ignores_a_late_click_and_keeps_internal_pause(self):
        rng = np.random.default_rng(71)
        raw = rng.normal(0, .001, 48000).astype(np.float32)
        for start, end in ((.4, .65), (.9, 1.15), (2.5, 2.51)):
            indices = np.arange(int(start * 16000), int(end * 16000))
            raw[indices] += .1 * np.sin(2 * np.pi * 220 * indices / 16000)
        metrics = recording_quality(raw, 16000)
        self.assertIn("isolated_activity", metrics["quality_flags"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "late_click.wav"
            sf.write(path, raw, 16000, subtype="FLOAT")
            original = extract_branches(path, self.base, phrase_span=True)
            candidate = extract_branches(path, self.base, phrase_span=True, sustained=True)
            self.assertGreater(original.phrase_end, 2.4)
            self.assertLess(candidate.phrase_start, .4)
            self.assertGreater(candidate.phrase_end, 1.15)
            self.assertLess(candidate.phrase_end, 1.4)
            self.assertGreater(len(candidate.phrase), 75)

    def test_silent_registration_is_an_invalid_take_with_a_rerecord_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            sf.write(path, np.zeros(48000), 16000)
            take, features = gui.assess_enrollment_take(path, self.base)
            self.assertFalse(take.accepted)
            self.assertIn("静音", take.reason)
            self.assertIn("重录", take.reason)
            self.assertEqual(take.diagnostics["sustained_seconds"], 0)
            self.assertEqual(features.shape, (0, len(self.base.feature_names)))
            self.assertIsNone(take.estimated_snr_db)
            self.assertIsNone(take.quality_score)

    def test_clipping_and_recording_edge_produce_quality_flags(self):
        raw = np.zeros(48000, dtype=np.float32)
        raw[:8000] = np.tile([1., -1.], 4000)
        quality = recording_quality(raw, 16000)
        self.assertIn("clipping", quality["quality_flags"])
        self.assertIn("recording_edge", quality["quality_flags"])
        self.assertAlmostEqual(quality["clip_fraction"], 1 / 6)

    def test_separate_branch_enrollment_and_saved_policy_replay(self):
        matrices = [extract_branches(path, self.base, voice_active=True, phrase_span=True) for path in self.paths]
        profile = gui.build_session_enrollment_model("candidate", self.base, [item.voice for item in matrices], passphrase_features=[item.phrase for item in matrices], codebook_size=16)
        self.assertEqual(len(profile.codebook), 16)
        dims = srdp.select_dims(self.base.feature_names, self.base.passphrase_feature_set)
        for saved, item in zip(profile.passphrase_templates, matrices):
            np.testing.assert_array_equal(saved, item.phrase[:, dims])
        path = srdp.PROJECT_ROOT / "data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav"
        expected = score_branches(extract_branches(path, self.base, voice_active=True, phrase_span=True), profile)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gui.save_user_model(profile, root / "parameters.npz")
            (root / "processing_policy.json").write_text(json.dumps({"policy": "both", "model_fingerprint": gui.session_model_fingerprint(profile)}), encoding="utf-8")
            actual = replay_candidate(root, path)
            self.assertEqual(actual["decision"], expected["decision"])
            for key in ("voice_score", "passphrase_score"):
                self.assertAlmostEqual(float(actual[key]), float(expected[key]), delta=2e-6)
            (root / "processing_policy.json").write_text(json.dumps({"policy": "both", "model_fingerprint": "wrong"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                replay_candidate(root, path)

    def test_mismatched_enrollment_branches_are_not_silently_paired(self):
        matrices = [srdp.extract_one_audio_features(path, self.base).features for path in self.paths]
        with self.assertRaisesRegex(ValueError, "不一致"):
            gui.build_session_enrollment_model("candidate", self.base, matrices, passphrase_features=matrices[:2])


if __name__ == "__main__":
    unittest.main()
