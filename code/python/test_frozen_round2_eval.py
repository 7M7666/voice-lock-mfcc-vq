import hashlib
import json
import subprocess
import tempfile
import unittest
import wave
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import frozen_round2_eval as runner


PHRASES = ["芝麻开门", "芝麻开窗", "芝麻开门", "芝麻关门", "芝麻开门", "开门",
           "芝麻开门", "西瓜开门", "芝麻开门", "芝麻开灯", "芝麻开门", "芝麻开门了"]


def policies():
    selected = {"variant": "pitch_relative_delta_fixed_1p1", "phrase_guard": 1.1,
                "cohort_gate": False, "pitch_gate": True, "pitch_stream": "pitch_relative",
                "streams": ["phrase_delta"], "threshold": 1.}
    original = {"variant": "two_layer", "phrase_guard": 1., "weights": [], "threshold": 1.}
    result = {"two_layer": original, selected["variant"]: selected,
              "guard_only_1p1": {**original, "variant": "guard_only_1p1", "phrase_guard": 1.1},
              runner.ALIAS: {**selected, "variant": runner.ALIAS}}
    for name in ("best_single_1p1", "feature_fusion_1p1"):
        result[name] = {"variant": name, "phrase_guard": 1.1,
                        "weights": [.25, .25, .5, 0.], "threshold": 1.}
    result["best_relative_single_1p1"] = {**selected, "variant": "best_relative_single_1p1"}
    result["delta_only_fixed_1p1"] = {**selected, "variant": "delta_only_fixed_1p1", "pitch_gate": False}
    result["pitch_relative_only_1p1"] = {**selected, "variant": "pitch_relative_only_1p1", "streams": []}
    return result


def context():
    rules = policies()
    return {"models": {s: SimpleNamespace(speaker_id=s, voice_threshold=1.,
                passphrase_threshold=1., passphrase_accept_score=1.) for s in runner.SPEAKERS},
            "sides": {}, "thresholds": {}, "policies": rules, "variants": list(rules),
            "candidate": {"selected_variant": "pitch_relative_delta_fixed_1p1",
                          "model_fingerprints": dict.fromkeys(runner.SPEAKERS, "synthetic")},
            "protocol": {"pilot_success_rules": ["Frozen protocol rules"]}}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.holdout = {"speakers": ["lwl2", "lyh2"], "per_speaker": 12,
                        "physical_recordings": 24, "paired_account_claims": 48,
                        "phrases": PHRASES}
        self.manifest = {"recordings": []}
        for speaker in self.holdout["speakers"]:
            for sequence, phrase in enumerate(PHRASES, 1):
                path = self.root / f"{speaker}_{sequence}.wav"
                path.write_bytes(f"synthetic test fixture {speaker} {sequence}".encode())
                self.manifest["recordings"].append({"file_path": path.name, "speaker": speaker,
                    "sequence": sequence, "phrase": phrase, "confirmed": True,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def validate(self, manifest=None, excluded=None):
        return runner.validate_recordings(manifest or self.manifest, self.holdout,
                                          excluded or {}, self.root)

    def test_complete_protocol_has_twenty_four_distinct_recordings(self):
        rows = self.validate()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r["sha256"] for r in rows}), 24)

    def test_missing_wav(self):
        (self.root / "lwl2_1.wav").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate()

    def test_duplicate_path_and_duplicate_content(self):
        for field in ("path", "content"):
            with self.subTest(field=field):
                manifest = deepcopy(self.manifest)
                first, second = manifest["recordings"][:2]
                second["sha256"] = first["sha256"]
                if field == "path":
                    second["file_path"] = first["file_path"]
                else:
                    (self.root / second["file_path"]).write_bytes((self.root / first["file_path"]).read_bytes())
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    self.validate(manifest)

    def test_sha_mismatch(self):
        self.manifest["recordings"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.validate()

    def test_labels_and_confirmation_are_required(self):
        for key in ("file_path", "speaker", "sequence", "phrase", "confirmed", "sha256"):
            with self.subTest(key=key):
                manifest = deepcopy(self.manifest)
                del manifest["recordings"][0][key]
                with self.assertRaises(ValueError):
                    self.validate(manifest)
        for confirmation in (False, "True", 1):
            manifest = deepcopy(self.manifest)
            manifest["recordings"][0]["confirmed"] = confirmation
            with self.assertRaisesRegex(ValueError, "confirmed"):
                self.validate(manifest)

    def test_speaker_sequence_phrase_and_count_must_match(self):
        for changes in ({"speaker": "unknown"}, {"speaker": "lyh2"}, {"sequence": 2},
                        {"sequence": True}, {"phrase": "wrong label"}):
            with self.subTest(changes=changes):
                manifest = deepcopy(self.manifest)
                manifest["recordings"][0].update(changes)
                with self.assertRaises(ValueError):
                    self.validate(manifest)
        self.manifest["recordings"].pop()
        with self.assertRaises(ValueError):
            self.validate()

    def test_enrollment_or_calibration_content_cannot_be_reused(self):
        first = self.manifest["recordings"][0]
        for role in ("enrollment", "calibration"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "previous"):
                self.validate(excluded={str(self.root / first["file_path"]): first["sha256"]})
        copied = self.root / "copied.wav"
        copied.write_bytes((self.root / first["file_path"]).read_bytes())
        first["file_path"] = copied.name
        with self.assertRaisesRegex(ValueError, "previous"):
            self.validate(excluded={str(self.root / "original.wav"): first["sha256"]})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.context = context()
        self.sample = {"path": Path("synthetic.wav"), "file_path": "synthetic.wav",
                       "sha256": "a" * 64, "speaker": "lwl2", "sequence": 1,
                       "phrase": PHRASES[0], "confirmed": True}

    def score(self, raw_phrase=.8, *, values=None, baseline=None, replayed=None, error=None):
        values = values or {"voice_mfcc": .5, "phrase_mfcc": float(f"{raw_phrase:.6f}"),
            "phrase_delta": .8, "phrase_spectral": .8, "pitch_relative": .5}
        baseline = baseline or {"decision": "lwl2" if raw_phrase <= 1. else "REJECT",
                                "rejection_stage": "", "reject_reason": ""}
        with patch.object(runner.gui, "predict_with_session_model_baseline", return_value=baseline), \
                patch.object(runner, "raw_scores", return_value={"raw_voice_score": .5,
                    "raw_phrase_score": raw_phrase, "raw_voice_distance": .5,
                    "raw_phrase_distance": raw_phrase}), \
                patch.object(runner.structure, "score", side_effect=error,
                             return_value=(values, replayed or baseline, [])):
            return runner.score_claim(self.sample, "lwl2", self.context)

    def test_structure_baseline_disagreement_is_explicit_and_blocks_complete_batch(self):
        for raw_phrase, structure_decision in ((.8, "REJECT"), (1.2, "lwl2")):
            with self.subTest(raw_phrase=raw_phrase):
                row = self.score(raw_phrase, replayed={"decision": structure_decision})
                self.assertEqual(row["status"], "baseline_mismatch")
                self.assertEqual(row["structure_baseline"], structure_decision)
                self.assertEqual(row["baseline"], "lwl2" if raw_phrase <= 1. else "REJECT")
                self.assertEqual(row["raw_phrase_score"], raw_phrase)
                self.assertIn("structure.score", row["baseline_mismatch_reason"])
                self.assertTrue(row["boundary_issues"])
                self.assertFalse(row["baseline_consistent"])
                rows = self.complete_rows()
                rows[0] = row
                summary = runner.summarize(rows, self.context)
                self.assertEqual(summary["claims"], 48)
                self.assertFalse(summary["conclusion_eligible"])
                self.assertFalse(summary["gain_counts_usable"])

    def test_one_and_one_point_one_rounding_disagreements_are_detected(self):
        for value in (1.0000004, 1.1000004):
            with self.subTest(value=value):
                row = self.score(value)
                self.assertEqual(row["raw_phrase_score"], value)
                self.assertTrue(row["boundary_issues"])
                self.assertFalse(runner.summarize([row], self.context)["conclusion_eligible"])
                if value < 1.1:
                    self.assertFalse(row["baseline_consistent"])
                    self.assertEqual(row["decisions"]["two_layer"], "lwl2")
                    self.assertEqual(row["baseline"], "REJECT")
                else:
                    self.assertTrue(row["baseline_consistent"])
                    self.assertEqual(row["decisions"][self.context["candidate"]["selected_variant"]], "lwl2")

    def test_exact_boundary_and_nonboundary_scores_do_not_report_false_mismatches(self):
        for value in (1., 1.1, .8, 1.000002, 1.100002):
            with self.subTest(value=value):
                self.assertEqual(self.score(value)["boundary_issues"], [])

    def test_pitch_unavailable_rejects_candidate_but_keeps_original(self):
        values = {"voice_mfcc": .5, "phrase_mfcc": .8, "phrase_delta": .8,
                  "phrase_spectral": .8, "pitch_relative": float("inf")}
        row = self.score(values=values)
        self.assertEqual(row["status"], "pitch_unavailable")
        self.assertEqual(row["decisions"]["two_layer"], "lwl2")
        self.assertEqual(row["decisions"][self.context["candidate"]["selected_variant"]], "REJECT")

    def test_input_rejection_and_scoring_failure_are_retained(self):
        row = self.score(baseline={"decision": "REJECT", "rejection_stage": "input",
                                   "reject_reason": "no speech"})
        self.assertEqual(row["status"], "input_rejected")
        self.assertTrue(all(x == "REJECT" for x in row["decisions"].values()))
        row = self.score(error=RuntimeError("synthetic scoring failure"))
        self.assertEqual(row["status"], "scoring_error")
        self.assertIn("synthetic scoring failure", row["error"])
        self.assertTrue(all(x is None for x in row["decisions"].values()))

    def test_tampered_alias_policy_is_not_claimed_equivalent(self):
        self.context["policies"][runner.ALIAS]["threshold"] = .7
        row = self.score()
        summary = runner.summarize([row], self.context)
        self.assertFalse(summary["alias_rule_equal"])
        self.assertFalse(summary["conclusion_eligible"])

    def complete_rows(self):
        rows = []
        for speaker in runner.SPEAKERS:
            for sequence, phrase in enumerate(PHRASES, 1):
                for target in runner.SPEAKERS:
                    own_correct = speaker == target and phrase == PHRASES[0]
                    decisions = dict.fromkeys(self.context["variants"], target if own_correct else "REJECT")
                    rows.append({"speaker": speaker, "target": target,
                        "kind": runner.trial_kind(speaker, target, phrase), "baseline": decisions["two_layer"],
                        "decisions": decisions, "status": "ok", "baseline_consistent": True,
                        "boundary_issues": []})
        return rows

    def test_four_categories_and_per_speaker_counts_keep_failures_in_denominator(self):
        rows = self.complete_rows()
        rows[0].update(status="scoring_error", decisions=dict.fromkeys(self.context["variants"]))
        summary = runner.summarize(rows, self.context)
        self.assertEqual(summary["claims"], 48)
        self.assertEqual(summary["category_counts"], dict.fromkeys(runner.KINDS, 12))
        stats = summary["metrics"]["two_layer"]
        self.assertEqual(stats["speakers"]["lwl2"]["correct_count"], 6)
        self.assertEqual(stats["speakers"]["lwl2"]["correct_accepted"], 5)
        self.assertEqual(stats["speakers"]["lwl2"]["correct_unscored"], 1)
        self.assertEqual(stats["speakers"]["lyh2"]["correct_accepted"], 6)
        self.assertEqual(stats["categories"]["self_wrong"]["false_accept"], 0)
        self.assertFalse(summary["conclusion_eligible"])
        self.assertFalse(summary["pilot_success"])

    def test_each_negative_category_and_per_account_counts_are_reported(self):
        rows = self.complete_rows()
        selected = self.context["candidate"]["selected_variant"]
        for kind in runner.KINDS[1:]:
            row = next(r for r in rows if r["speaker"] == "lwl2" and r["kind"] == kind)
            row["decisions"][selected] = row["target"]
            row["decisions"][runner.ALIAS] = row["target"]
        summary = runner.summarize(rows, self.context)
        stats = summary["metrics"][selected]
        for kind in runner.KINDS[1:]:
            self.assertEqual(stats["categories"][kind]["false_accept"], 1)
            self.assertEqual(stats["speakers"]["lwl2"]["negative_false_accepts"][kind], 1)
        self.assertFalse(summary["benefit_conditions"]["zero_negative_accepts"])
        self.assertFalse(summary["original_method_benefit_pass"])

    def test_recovery_can_pass_original_comparison_but_never_proves_extra_fusion_gain(self):
        rows = self.complete_rows()
        rows[0]["baseline"] = rows[0]["decisions"]["two_layer"] = "REJECT"
        summary = runner.summarize(rows, self.context)
        self.assertTrue(summary["conclusion_eligible"])
        self.assertTrue(summary["original_method_benefit_pass"])
        self.assertFalse(summary["pilot_success"])
        self.assertFalse(summary["fusion_gain_proven"])
        self.assertFalse(runner.summarize(rows, self.context, development_test=True)["gain_counts_usable"])

    def test_no_person_can_regress_and_even_count_neutral_swaps_are_regressions(self):
        rows = self.complete_rows()
        selected = self.context["candidate"]["selected_variant"]
        correct = [r for r in rows if r["kind"] == "self_correct"]
        correct[0]["baseline"] = correct[0]["decisions"]["two_layer"] = "REJECT"
        correct[1]["decisions"][selected] = correct[1]["decisions"][runner.ALIAS] = "REJECT"
        summary = runner.summarize(rows, self.context)
        self.assertTrue(summary["benefit_conditions"]["each_person_not_worse"])
        self.assertFalse(summary["benefit_conditions"]["strict_improvement_without_regression"])
        self.assertEqual(summary["observed_regressed_claims"], 1)




class FrozenRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "frozen"
        self.bundle.mkdir()
        self.protocol_path = self.root / "next_round_protocol.json"
        self.manifest_path = self.root / "round2.json"
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True, capture_output=True)
        (self.root / ".gitignore").write_text("tmp*/\n", encoding="utf-8")
        self.output = self.root / "tmp_round2" / "evaluation"
        self.json_write = lambda path, data: path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        model_files = self.bundle / "models"
        model_files.mkdir()
        labels = {"models": {}, "recordings": []}
        protocol = {"holdout": {"speakers": list(runner.SPEAKERS), "per_speaker": 12,
            "physical_recordings": 24, "paired_account_claims": 48, "phrases": PHRASES,
            "genuine_correct": 12, "own_wrong": 12, "other_correct": 12, "other_wrong": 12},
            "model_freeze": {}, "pilot_success_rules": ["Synthetic fixture, not holdout evidence"]}
        manifest, thresholds = {"recordings": []}, {}
        for speaker in runner.SPEAKERS:
            model = runner.gui.SessionEnrollmentModel(speaker_id=speaker,
                codebook=np.zeros((16, 12), dtype=np.float32), voice_threshold=1.,
                passphrase_templates=[np.zeros((8, 12), dtype=np.float32) for _ in range(4)],
                passphrase_threshold=1., feature_set="mfcc", passphrase_feature_set="mfcc",
                feature_names=runner.srdp.FEATURE_NAMES,
                zscore_mean=np.zeros(31, dtype=np.float32), zscore_std=np.ones(31, dtype=np.float32),
                quality_baseline={}, passphrase_aggregate="min", dtw_window_ratio=1.,
                dtw_warp_penalty=0., passphrase_accept_score=1.)
            current = self.root / "app_data" / f"{speaker}.npz"
            runner.gui.save_user_model(model, current)
            runner.gui.save_user_model(model, model_files / f"{speaker}.npz")
            arrays = {"count": 4, "delta_threshold": 1., "spectral_threshold": 1.}
            for i in range(4):
                arrays[f"delta_{i}"] = np.zeros((8, 12))
                arrays[f"spectral_{i}"] = np.zeros((8, 7))
            np.savez_compressed(model_files / f"{speaker}_side.npz", **arrays)
            np.savez_compressed(model_files / f"{speaker}_enrollment.npz", count=4,
                                **{f"matrix_{i}": np.zeros((8, 31)) for i in range(4)})
            fp = runner.gui.session_model_fingerprint(model)
            protocol["model_freeze"][speaker] = {"fingerprint": fp, "voice_threshold": 1., "phrase_threshold": 1.}
            enrollment = []
            for i in range(4):
                old = self.root / f"enroll_{speaker}_{i}.wav"
                old.write_bytes(f"old enrollment {speaker} {i}".encode())
                enrollment.append({"path": old.name, "sha256": runner.digest(old)})
            labels["models"][speaker] = {"model_path": str(current.relative_to(self.root)),
                                         "fingerprint": fp, "enrollment": enrollment}
            thresholds[speaker] = {"local_mfcc": 1., "local_delta": 1.,
                                   "pitch": {"center": 7. if speaker == "lwl2" else 8., "threshold": .1}}
            for sequence, phrase in enumerate(PHRASES, 1):
                old = self.root / f"calibration_{speaker}_{sequence}.wav"
                old.write_bytes(f"old calibration {speaker} {sequence}".encode())
                labels["recordings"].append({"file_path": old.name, "sha256": runner.digest(old)})
                path = self.root / f"{speaker}_{sequence}.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(np.full(1600, sequence + (100 if speaker == "lyh2" else 0), dtype="<i2").tobytes())
                manifest["recordings"].append({"file_path": path.name, "speaker": speaker,
                    "sequence": sequence, "phrase": phrase, "confirmed": True, "sha256": runner.digest(path)})
        self.json_write(self.protocol_path, protocol)
        self.json_write(self.manifest_path, manifest)
        self.json_write(self.bundle / "labels.json", labels)
        self.json_write(self.bundle / "local_thresholds.json", thresholds)
        self.json_write(self.bundle / "policies.json", policies())
        for name in ("protocol.json", "pitch_protocol.json", "relative_pitch_protocol.json", "environment.json"):
            self.json_write(self.bundle / name, {"synthetic_fixture": True})
        source_dir = self.bundle / "source"
        source_dir.mkdir()
        sources = {}
        for name in ("gui.py", "srdp.py", "speech_frames.py", "dtw_paths.py",
                     "live_structure_experiments.py", "live_fusion_experiments.py"):
            path = Path(runner.__file__).parent / name
            (source_dir / name).write_bytes(path.read_bytes())
            sources[name] = runner.digest(path)
        self.json_write(self.bundle / "source_hashes.json", sources)
        bundle_hashes = {p.relative_to(self.bundle).as_posix(): runner.digest(p)
            for p in self.bundle.rglob("*") if p.is_file() and p.name != "labels.json"}
        self.json_write(self.bundle / "bundle_hashes.json", bundle_hashes)
        self.json_write(self.bundle / "frozen_candidate.json", {
            "selected_variant": "pitch_relative_delta_fixed_1p1",
            "policy": policies()["pitch_relative_delta_fixed_1p1"],
            "comparators": [name for name in policies() if name != "pitch_relative_delta_fixed_1p1"],
            "model_fingerprints": {s: r["fingerprint"] for s, r in labels["models"].items()},
            "bundle_hash_manifest_sha256": runner.digest(self.bundle / "bundle_hashes.json"),
            "labels_sha256": runner.digest(self.bundle / "labels.json"),
            "independent_protocol_sha256": runner.digest(self.protocol_path)})
        self.json_write(self.bundle / "freeze_receipt.json", {
            "candidate_sha256": runner.digest(self.bundle / "frozen_candidate.json"),
            "bundle_hashes_sha256": runner.digest(self.bundle / "bundle_hashes.json")})

    def load(self):
        return runner.load_freeze(self.bundle, self.protocol_path, self.root)

    def test_frozen_parameters_restore_and_all_bindings_are_checked(self):
        context = self.load()
        self.assertEqual(len(context["variants"]), 9)
        self.assertEqual(len(context["excluded"]), 32)
        for speaker in runner.SPEAKERS:
            np.testing.assert_array_equal(context["models"][speaker].codebook, np.zeros((16, 12)))
            self.assertEqual(context["models"][speaker].passphrase_threshold, 1.)

    def test_tampering_frozen_materials_and_actual_models_fails(self):
        paths = [self.bundle / name for name in ("frozen_candidate.json", "freeze_receipt.json",
            "labels.json", "policies.json", "local_thresholds.json", "source/gui.py", "models/lwl2.npz")]
        paths += [self.protocol_path, self.root / "app_data/lwl2.npz"]
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"invalid fixture")
                with self.assertRaises((ValueError, KeyError)):
                    self.load()
                path.write_bytes(original)

    def test_actual_source_change_is_rejected(self):
        with patch.object(runner.structure.fusion, "digest", side_effect=lambda p:
                "0" * 64 if p == Path(runner.__file__).parent / "gui.py" else runner.digest(p)):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.load()

    def mock_scores(self, path, target, models, sides, thresholds):
        speaker, sequence = path.stem.split("_")
        phrase = PHRASES[int(sequence) - 1]
        valid = speaker == target and phrase == PHRASES[0]
        values = {"voice_mfcc": .5 if speaker == target else 1.5,
            "phrase_mfcc": .8 if phrase == PHRASES[0] else 1.2,
            "phrase_delta": .8, "phrase_spectral": .8, "pitch_relative": .5}
        return values, {"decision": target if valid else "REJECT"}, []

    def mock_app(self, path, model):
        _, baseline, _ = FrozenRunnerTests.mock_scores(self, path, model.speaker_id, None, None, None)
        return {**baseline, "rejection_stage": "", "reject_reason": ""}

    def mock_raw(self, path, model):
        values, _, _ = FrozenRunnerTests.mock_scores(self, path, model.speaker_id, None, None, None)
        return {"raw_voice_distance": values["voice_mfcc"], "raw_phrase_distance": values["phrase_mfcc"],
                "raw_voice_score": values["voice_mfcc"], "raw_phrase_score": values["phrase_mfcc"]}

    def execute(self):
        with ExitStack() as stack:
            forbidden = [(runner.structure, "fit"), (runner.structure, "grouped_predictions"),
                (runner.structure.fusion, "fit_policy"), (runner.structure.fusion, "build_side_streams"),
                (runner.srdp, "lbg_codebook"), (runner.gui, "build_session_enrollment_model")]
            checks = [stack.enter_context(patch.object(module, name,
                      side_effect=AssertionError("Fitting forbidden"))) for module, name in forbidden]
            stack.enter_context(patch.object(runner.gui, "predict_with_session_model_baseline", side_effect=self.mock_app))
            stack.enter_context(patch.object(runner, "raw_scores", side_effect=self.mock_raw))
            stack.enter_context(patch.object(runner.structure, "score", side_effect=self.mock_scores))
            result = runner.run(self.manifest_path, self.bundle, self.protocol_path, self.output,
                                development_test=True, root=self.root)
            for check in checks:
                check.assert_not_called()
            return result

    def test_batch_expands_claims_writes_only_new_outputs_and_never_fits(self):
        before = {p: runner.digest(p) for p in self.root.rglob("*") if p.is_file()}
        summary = self.execute()
        self.assertEqual(summary["claims"], 48)
        self.assertEqual(summary["category_counts"], dict.fromkeys(runner.KINDS, 12))
        self.assertEqual(summary["metrics"]["two_layer"]["speakers"]["lyh2"]["correct_accepted"], 6)
        self.assertTrue(summary["alias_rule_equal"])
        self.assertTrue(summary["alias_decisions_equal"])
        self.assertFalse(summary["fusion_gain_proven"])
        self.assertFalse(summary["pilot_success"])
        self.assertEqual(summary["evidence"], "development_test")
        self.assertFalse(summary["conclusion_eligible"])
        runner.verify_files(before)
        import csv
        with (self.output / "trials.csv").open(encoding="utf-8", newline="") as handle:
            trials = list(csv.DictReader(handle))
        self.assertEqual(len(trials), 48 * 9)
        self.assertIn("raw_phrase_score", trials[0])
        self.assertIn("policy", trials[0])
        self.assertEqual(len(list(self.output.iterdir())), 3)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=self.root, check=True, capture_output=True, text=True)
        self.assertNotIn("tmp_round2/", status.stdout)
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.execute()

    def test_project_outputs_reject_unignored_and_other_ignored_directories(self):
        for relative in ("evaluation", "reports/run", "results/round2", ".git/private", "tmp_other/run"):
            with self.subTest(relative=relative):
                self.output = self.root / relative
                with self.assertRaisesRegex(ValueError, "tmp_round2"):
                    self.execute()
                self.assertFalse(self.output.exists())

    def test_project_output_requires_all_result_files_to_be_git_ignored(self):
        for patterns in ("", "tmp_round2/*\n!tmp_round2/evaluation/\n"
                         "tmp_round2/evaluation/*\n!tmp_round2/evaluation/summary.json\n"):
            with self.subTest(patterns=patterns):
                (self.root / ".gitignore").write_text(patterns, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Git-ignored"):
                    self.execute()
                self.assertFalse(self.output.exists())

    def test_new_external_output_is_allowed(self):
        with tempfile.TemporaryDirectory() as outside:
            self.output = Path(outside) / "round2-results"
            summary = self.execute()
            self.assertEqual(summary["claims"], 48)
            self.assertTrue((self.output / "summary.json").is_file())

    def test_structure_baseline_mismatch_is_written_to_difference_report(self):
        original = self.mock_scores
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                self.output = self.root / "tmp_round2" / str(unavailable)
                def disagree(path, target, *args):
                    values, baseline, details = original(path, target, *args)
                    if path.stem == "lwl2_1" and target == "lwl2":
                        return None if unavailable else values, {"decision": "REJECT"}, details
                    return values, baseline, details
                with patch.object(self, "mock_scores", side_effect=disagree):
                    summary = self.execute()
                self.assertEqual(summary["claims"], 48)
                self.assertEqual(summary["boundary_disagreements"], 1)
                self.assertEqual(summary["baseline_mismatches"], 1)
                self.assertEqual(summary["scoring_errors"], 0)
                self.assertFalse(summary["gain_counts_usable"])
                import csv
                with (self.output / "boundary_disagreements.csv").open(encoding="utf-8", newline="") as handle:
                    differences = list(csv.DictReader(handle))
                self.assertEqual(len(differences), 1)
                self.assertEqual(differences[0]["file_path"], "lwl2_1.wav")
                self.assertEqual(differences[0]["target"], "lwl2")
                self.assertEqual(differences[0]["baseline"], "lwl2")
                self.assertEqual(differences[0]["structure_baseline"], "REJECT")
                self.assertIn("structure.score", differences[0]["baseline_mismatch_reason"])
                self.assertEqual(differences[0]["status"], "baseline_mismatch")

    def test_one_failed_score_retains_all_claims_and_blocks_benefit(self):
        original = self.mock_scores
        def fail_one(path, target, *args):
            if path.stem == "lwl2_1" and target == "lwl2":
                raise RuntimeError("injected auxiliary failure")
            return original(path, target, *args)
        with patch.object(self, "mock_scores", side_effect=fail_one):
            summary = self.execute()
        self.assertEqual(summary["claims"], 48)
        self.assertEqual(summary["scoring_errors"], 1)
        self.assertEqual(summary["metrics"]["two_layer"]["categories"]["self_correct"]["count"], 12)
        self.assertEqual(summary["metrics"]["two_layer"]["categories"]["self_correct"]["unscored"], 1)
        self.assertFalse(summary["conclusion_eligible"])

    def test_input_and_pitch_failures_keep_the_complete_batch(self):
        original_scores, original_app = self.mock_scores, self.mock_app
        def input_failure(path, model):
            if path.stem == "lwl2_1" and model.speaker_id == "lwl2":
                return {"decision": "REJECT", "rejection_stage": "input", "reject_reason": "synthetic no speech"}
            return original_app(path, model)
        def pitch_failure(path, target, *args):
            values, baseline, details = original_scores(path, target, *args)
            if path.stem == "lwl2_3" and target == "lwl2":
                values["pitch_relative"] = float("inf")
            return values, baseline, details
        with patch.object(self, "mock_app", side_effect=input_failure), \
                patch.object(self, "mock_scores", side_effect=pitch_failure):
            summary = self.execute()
        self.assertEqual(summary["claims"], 48)
        self.assertEqual(summary["category_counts"], dict.fromkeys(runner.KINDS, 12))
        stats = summary["metrics"]["pitch_relative_delta_fixed_1p1"]["speakers"]["lwl2"]
        self.assertEqual(stats["correct_count"], 6)
        self.assertEqual(stats["false_reject"], 2)
        self.assertEqual(stats["correct_accepted"], 4)

    def test_real_distance_audit_matches_app_on_development_wav_without_training(self):
        context = self.load()
        model = context["models"]["lwl2"]
        path = runner.ROOT / "data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav"
        with patch.object(runner.srdp, "lbg_codebook", side_effect=AssertionError("No training")):
            raw = runner.raw_scores(path, model)
            app = runner.gui.predict_with_session_model_baseline(path, model)
        self.assertEqual(app["voice_score"], f'{raw["raw_voice_score"]:.6f}')
        self.assertEqual(app["passphrase_score"], f'{raw["raw_phrase_score"]:.6f}')
        self.assertEqual(app["voice_passed"], str(raw["raw_voice_score"] <= 1.))
        self.assertEqual(app["passphrase_passed"], str(raw["raw_phrase_score"] <= 1.))


if __name__ == "__main__":
    unittest.main()
