"""Phrase-only ablations with frozen speaker gates; never deploys candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import gui
import srdp


VARIANTS = ("personal_mfcc", "personal_cmn", "shared_mfcc", "shared_cmn")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phrase_matrix(features, model, centered):
    matrix = features[:, srdp.select_dims(model.feature_names, "mfcc")].copy()
    if centered:
        matrix -= matrix.mean(axis=0)
    return matrix


def calibration_references(owners, held, shared):
    return [i for i, owner in enumerate(owners)
        if (owner != owners[held] if shared else i != held)]


@dataclass
class PhraseCandidate:
    templates: list[np.ndarray]
    threshold: float
    centered: bool
    aggregate: str
    window_ratio: float
    warp_penalty: float
    max_run: int
    distance_cache: dict = field(default_factory=dict, repr=False)

    def distance(self, matrix):
        key = (matrix.shape, matrix.tobytes())
        if key not in self.distance_cache:
            self.distance_cache[key] = srdp.passphrase_template_distance(matrix, self.templates, self.aggregate,
                window_ratio=self.window_ratio, warp_penalty=self.warp_penalty, max_run=self.max_run)
        return self.distance_cache[key]

    def save(self, directory, model, policy):
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "phrase_bank.npz", **{
            f"template_{i}": matrix for i, matrix in enumerate(self.templates)})
        # This private copy supplies the frozen extraction and voice parameters for replay.
        gui.save_user_model(model, directory / "frozen_voice.npz")
        payload = {**policy, "threshold": self.threshold, "centered": self.centered,
            "aggregate": self.aggregate, "window_ratio": self.window_ratio,
            "warp_penalty": self.warp_penalty, "max_run": self.max_run,
            "template_count": len(self.templates), "voice_fingerprint": gui.session_model_fingerprint(model),
            "bank_sha256": sha256(directory / "phrase_bank.npz"),
            "deployment": "experimental_only; use replay CLI, not App model loader"}
        (directory / "policy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calibrate(matrices, owners, model, shared, *, formal=False):
    distances, rows = [], []
    for held, matrix in enumerate(matrices):
        indices = calibration_references(owners, held, shared)
        if not indices:
            raise ValueError("Calibration has no held-out references")
        distance = srdp.passphrase_template_distance(matrix, [matrices[i] for i in indices],
            model.passphrase_aggregate, window_ratio=model.dtw_window_ratio,
            warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
        distances.append(distance)
        rows.append({"held_index": held, "held_owner": owners[held], "reference_indices": json.dumps(indices),
            "reference_owners": json.dumps(sorted({owners[i] for i in indices})), "distance": distance})
    if shared or formal or model.calibration_mode == "leave_one_recording_out":
        threshold = srdp.calibration_threshold(distances, std_factor=2.0 if shared else 1.0)
    else:
        values = np.asarray(distances)
        threshold = float(max(values.mean() + 2 * values.std(), values.max() + 1e-6))
    return threshold, rows


def evaluate(candidate, model, result, features):
    if result.get("rejection_stage") == "input":
        return {"decision": "REJECT", "voice_score": "", "phrase_distance": "", "phrase_score": "",
            "phrase_passed": "", "reason": result["reject_reason"]}
    matrix = phrase_matrix(features, model, candidate.centered)
    distance = candidate.distance(matrix)
    score = srdp.safe_ratio(distance, candidate.threshold)
    voice = float(result["voice_score"])
    # Preserve both the saved decision rule and its speaker safety boundary.
    accepted, _ = srdp.verification_decision(voice, score / model.passphrase_accept_score,
        model.decision_mode, model.fusion_voice_weight, model.fusion_threshold,
        voice_guard=model.voice_guard, passphrase_guard=model.passphrase_guard)
    accepted = accepted and result.get("voice_passed") != "False"
    return {"decision": model.speaker_id if accepted else "REJECT", "voice_score": result["voice_score"],
        "phrase_distance": distance, "phrase_score": score,
        "phrase_passed": str(score <= model.passphrase_accept_score), "reason": ""}


def replay(directory, wav):
    policy = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    model = gui.load_user_model(directory / "frozen_voice.npz")
    if gui.session_model_fingerprint(model) != policy["voice_fingerprint"] or sha256(directory / "phrase_bank.npz") != policy["bank_sha256"]:
        raise ValueError("Candidate does not match policy fingerprints")
    with np.load(directory / "phrase_bank.npz", allow_pickle=False) as bank:
        candidate = PhraseCandidate([bank[f"template_{i}"].copy() for i in range(policy["template_count"])],
            **{key: policy[key] for key in ("threshold", "centered", "aggregate", "window_ratio", "warp_penalty", "max_run")})
    result = gui.predict_with_session_model(wav, model)
    features = None if result.get("rejection_stage") == "input" else srdp.extract_one_audio_features(wav, model).features
    return evaluate(candidate, model, result, features)


def summarize(trials):
    summaries = []
    for scope, variant, kind in sorted({(r["scope"], r["variant"], r["kind"]) for r in trials}):
        rows = [r for r in trials if (r["scope"], r["variant"], r["kind"]) == (scope, variant, kind)]
        known = [r for r in rows if r["expected_decision"]]
        summaries.append({"scope": scope, "variant": variant, "kind": kind, "count": len(rows),
            "labeled_count": len(known), "accepted": sum(r["decision"] != "REJECT" for r in rows),
            "phrase_passed": sum(r["phrase_passed"] == "True" for r in rows),
            "false_accept": sum(r["expected_decision"] == "REJECT" and r["decision"] != "REJECT" for r in known),
            "false_reject": sum(r["expected_decision"] != "REJECT" and r["decision"] == "REJECT" for r in known)})
    return summaries


def formal_profile(base, speaker):
    return gui.SessionEnrollmentModel(speaker_id=speaker, codebook=base.codebooks[speaker],
        voice_threshold=base.thresholds[speaker], passphrase_templates=base.passphrase_train_by_speaker[speaker],
        passphrase_threshold=base.passphrase_thresholds[speaker], feature_set=base.feature_set,
        passphrase_feature_set=base.passphrase_feature_set, feature_names=base.feature_names,
        zscore_mean=base.zscore_mean, zscore_std=base.zscore_std, quality_baseline=base.quality_baseline,
        passphrase_aggregate=base.passphrase_aggregate, dtw_window_ratio=base.dtw_window_ratio,
        dtw_warp_penalty=base.dtw_warp_penalty, passphrase_accept_score=1., voice_metric=base.voice_metric,
        background_codebooks={s: b for s, b in base.codebooks.items() if s != speaker},
        decision_mode=base.decision_mode, fusion_voice_weight=base.fusion_voice_weight,
        fusion_threshold=base.fusion_threshold, calibration_mode=base.calibration_mode,
        vad_mode=base.vad_mode, cepstral_normalization=base.cepstral_normalization,
        dtw_max_run=base.dtw_max_run, voice_guard=base.voice_guard, passphrase_guard=base.passphrase_guard)


def run(args):
    output = srdp.project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    base = srdp.load_model()
    metadata_path = srdp.DEMO_AUDIO_DIR / "metadata_srdp.csv"
    metadata = srdp.read_metadata(metadata_path)
    training = [r for r in metadata if r["split"] == "train"]
    if any(r["passphrase"] != "芝麻开门" for r in training):
        raise ValueError("Shared bank requires explicitly labeled correct training phrases")
    labels = json.loads(srdp.project_path(args.live_labels).read_text(encoding="utf-8")) if args.live_labels else {}
    (output / "live_labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    protected = list(gui.APP_DATA_DIR.rglob("*")) + list(srdp.MODEL_DIR.glob("*.npz")) + [metadata_path, srdp.FEATURE_DIR / "srdp_features.npz"]
    protected += list(srdp.RESULTS_DIR.rglob("*"))
    protected += [srdp.project_path(r["file_path"]) for r in metadata] + list(gui.TEMP_AUDIO_DIR.glob("*.wav"))
    before = {str(p): sha256(p) for p in protected if p.is_file()}
    (output / "input_hashes.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    cache, shared_candidates = {}, {}

    def features(path, model):
        key = (str(path), model.zscore_mean.tobytes(), model.zscore_std.tobytes(), model.vad_mode, model.cepstral_normalization)
        if key not in cache:
            cache[key] = srdp.extract_one_audio_features(path, model).features
        return cache[key]

    profiles = [("formal", formal_profile(base, s)) for s in sorted(base.codebooks)]
    profiles += [("live", gui.load_user_model(p)) for p in sorted(gui.USER_MODEL_DIR.glob("*.npz"))]
    history = srdp.read_csv_rows(gui.HISTORY_PATH)
    trials, changes, audits, calibration = [], [], [], []
    for scope, model in profiles:
        fingerprint = gui.session_model_fingerprint(model)
        if model.passphrase_feature_set != "mfcc" or model.decision_mode != "two_layer":
            raise ValueError("This controlled experiment requires the MFCC/two-layer baseline")
        bank_features = [features(srdp.project_path(r["file_path"]), model) for r in training]
        if scope == "live":
            reports = [json.loads(p.read_text(encoding="utf-8")) for p in (gui.APP_DATA_DIR / "diagnostics/enrollments").glob("*.json")]
            report = next(r for r in reports if r["model_fingerprint"] == fingerprint)
            samples = [r for r in report["samples"] if r["accepted"]]
            paths = [srdp.project_path(r["path"]) for r in samples]
            if any(sha256(p) != r["audio_sha256"] for p, r in zip(paths, samples)):
                raise ValueError("Enrollment audio changed")
            own_features = [features(p, model) for p in paths]
            queries = []
            for row in history:
                if row["model_fingerprint"] != fingerprint:
                    continue
                label = labels.get(row["time_text"], {})
                relation, content = label.get("identity_relation", "unknown"), label.get("content", "unknown")
                kind = ("self_correct" if relation == "self" else "other_correct" if relation == "other" else "identity_unknown_correct") if content == "correct" else "self_wrong" if relation == "self" and content == "wrong" else "unknown"
                expected = model.speaker_id if kind == "self_correct" else "REJECT" if kind in {"other_correct", "self_wrong"} else ""
                queries.append({"file_path": row["file_path"], "kind": kind, "expected_decision": expected, "time": row["time_text"], "history": row})
            audits.append({"speaker": model.speaker_id, "fingerprint": fingerprint, "registration": report,
                "verification_count": len(queries), "identity_label_basis": "user: all outcomes correct; relationship inferred, rejected speaker names unknown"})
        else:
            own_features = [m for r, m in zip(training, bank_features) if r["speaker_id"] == model.speaker_id]
            queries = []
            for row in metadata:
                if row["split"] == "train":
                    continue
                # The corpus labels wrong phrase identity via Sxx_wrong; wording itself is unspecified.
                owner = row["speaker_id"].removesuffix("_wrong")
                correct = row["passphrase"] == "芝麻开门"
                kind = ("self_correct" if owner == model.speaker_id else "other_correct" if owner in base.codebooks else "stranger_correct") if correct else ("self_wrong" if owner == model.speaker_id else "other_wrong")
                queries.append({"file_path": row["file_path"], "kind": kind,
                    "expected_decision": model.speaker_id if kind == "self_correct" else "REJECT", "time": ""})
        for matrix, template in zip(own_features, model.passphrase_templates):
            np.testing.assert_allclose(phrase_matrix(matrix, model, False), template, atol=2e-6, rtol=1e-6)
        if len(own_features) != len(model.passphrase_templates):
            raise ValueError("Enrollment/template count mismatch")
        candidates = {}
        for variant in VARIANTS:
            shared, centered = variant.startswith("shared"), variant.endswith("cmn")
            matrices = [phrase_matrix(m, model, centered) for m in (bank_features if shared else own_features)]
            owners = [r["speaker_id"] for r in training] if shared else [model.speaker_id] * len(matrices)
            shared_key = (variant, tuple((m.shape, m.tobytes()) for m in matrices), model.passphrase_aggregate,
                model.dtw_window_ratio, model.dtw_warp_penalty, model.dtw_max_run)
            if shared and shared_key in shared_candidates:
                candidate, rows = shared_candidates[shared_key]
                threshold = candidate.threshold
            elif variant == "personal_mfcc":
                threshold, rows = model.passphrase_threshold, []
            else:
                threshold, rows = calibrate(matrices, owners, model, shared, formal=scope == "formal")
            calibration += [{"scope": scope, "target": model.speaker_id, "variant": variant, "threshold": threshold, **r} for r in rows]
            if not shared or shared_key not in shared_candidates:
                candidate = PhraseCandidate(matrices, threshold, centered, model.passphrase_aggregate,
                    model.dtw_window_ratio, model.dtw_warp_penalty, model.dtw_max_run)
            if shared:
                shared_candidates[shared_key] = candidate, rows
            directory = output / "candidates" / scope / model.speaker_id / variant
            candidate.save(directory, model, {"variant": variant, "template_owners": owners,
                "template_paths": [r["file_path"] for r in training] if shared else [srdp.display_path(p) for p in paths] if scope == "live" else [r["file_path"] for r in training if r["speaker_id"] == model.speaker_id]})
            candidates[variant] = candidate
        for query in queries:
            path = srdp.project_path(query["file_path"])
            original = gui.predict_with_session_model(path, model)
            matrix = None if original.get("rejection_stage") == "input" else features(path, model)
            if "history" in query:
                row = query["history"]
                if sha256(path) != row["audio_sha256"] or original["decision"] != row["decision"]:
                    raise ValueError("History audio or decision changed")
                for key in ("voice_score", "passphrase_score"):
                    if abs(float(original[key]) - float(row[key])) > 2e-6:
                        raise ValueError("History score differs from replay")
            baseline = None
            for variant, candidate in candidates.items():
                start = time.perf_counter()
                result = evaluate(candidate, model, original, matrix)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if variant == "personal_mfcc":
                    baseline = result
                    if result["decision"] != original["decision"] or (result["phrase_score"] != "" and abs(result["phrase_score"] - float(original["passphrase_score"])) > 2e-6):
                        raise ValueError("Personal control does not reproduce frozen model")
                trial = {"scope": scope, "target": model.speaker_id, "model_fingerprint": fingerprint,
                    "variant": variant, "kind": query["kind"], "expected_decision": query["expected_decision"],
                    "time": query["time"], "file_path": query["file_path"], "audio_sha256": sha256(path),
                    **result, "phrase_threshold": candidate.threshold, "phrase_limit": model.passphrase_accept_score,
                    "phrase_ms": elapsed_ms}
                trials.append(trial)
                if variant != "personal_mfcc" and (result["decision"] != baseline["decision"] or result["phrase_passed"] != baseline["phrase_passed"]):
                    changes.append({**trial, "before_decision": baseline["decision"], "before_phrase_score": baseline["phrase_score"],
                        "before_phrase_passed": baseline["phrase_passed"], "final_change": "unchanged" if result["decision"] == baseline["decision"] else "new_accept" if result["decision"] != "REJECT" else "new_reject",
                        "phrase_change": "unchanged" if result["phrase_passed"] == baseline["phrase_passed"] else "new_pass" if result["phrase_passed"] == "True" else "new_fail"})
        print(f"{scope}/{model.speaker_id}: {len(queries)} queries x {len(candidates)} variants", flush=True)
    for filename, rows in (("trials.csv", trials), ("changes.csv", changes), ("calibration.csv", calibration), ("summary.csv", summarize(trials))):
        if rows:
            srdp.write_rows(output / filename, rows, list(rows[0]))
    (output / "live_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {"variants": VARIANTS, "shared_threshold": "mean + 2 std of training-only leave-one-speaker-out template distances; fixed before evaluation",
        "personal_cmn_threshold": "formal: baseline mean+1std; live: saved App calibration rule (legacy max(mean+2std,max+epsilon) or calibrated mean+1std); enrollment only",
        "frozen": "speaker codebook, threshold, decision rule, crop, pretrained z-score, 12 MFCC dimensions, DTW path/aggregation",
        "normalization": "CMN on phrase matrices only, after frozen z-score; speaker features unchanged",
        "calibration_limit": "Shared calibration excludes held speaker templates, but frozen pretrained z-score used all 36 training recordings. Not fully nested speaker cross-validation.",
        "data_limit": "63 formal development recordings previously reused; wrong phrase exact words unknown, no confirmed near-error claim; live has no wrong phrase. Repeated claims are not independent WAVs.",
        "timing": "phrase scoring only, includes cache hits, excludes extraction, disk IO and first JIT; not end-to-end latency or benchmark",
        "deployment": "none; all candidates including failures retained"}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    unchanged = all(p.is_file() and sha256(p) == digest for name, digest in before.items() for p in [Path(name)])
    (output / "preservation.json").write_text(json.dumps({"unchanged": unchanged, "protected_files": len(before)}, indent=2), encoding="utf-8")
    if not unchanged:
        raise ValueError("Protected inputs changed during experiment")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=f"tmp/registration_diagnosis/20261031/phrase_content_{time.time_ns()}")
    parser.add_argument("--live-labels", help="JSON keyed by full history timestamp; unknown if absent")
    parser.add_argument("--replay-candidate", type=Path)
    parser.add_argument("--wav", type=Path)
    args = parser.parse_args()
    if args.replay_candidate:
        if not args.wav:
            parser.error("--replay-candidate requires --wav")
        print(json.dumps(replay(args.replay_candidate, srdp.project_path(args.wav)), ensure_ascii=False))
    else:
        run(args)
