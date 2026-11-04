"""Offline threshold-only A/B development; never installs a policy in the App.

Run from the project root with --output-dir tmp_acceptance/<new-directory>.
Optionally pass --live-root %LOCALAPPDATA%/SRDPVoiceLock for labeled-history
regression. Corpus words/speakers follow metadata; ASR never supplies truth labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import asr_phrase
import gui
import srdp
from acceptance_baseline import kind_for, protected_hashes
from dtw_calibration import BLENDS, blended_threshold, coverage_check, pairwise_distances, select_blend
from phrase_content_experiments import formal_profile


ROOT = srdp.PROJECT_ROOT
KINDS = ("self_correct", "self_wrong", "other_correct", "other_wrong")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(rows, fields):
    return {kind: {"count": sum(r["kind"] == kind for r in rows),
        **{field: sum(bool(r[field]) for r in rows if r["kind"] == kind) for field in fields}}
        for kind in KINDS if any(r["kind"] == kind for r in rows)}


def apply(row, blend):
    threshold = blended_threshold(row["baseline_threshold"], row["pair_q75"], blend)
    phrase = row["dtw_distance"] <= threshold
    two = row["voice_passed"] and phrase
    return dict(threshold=threshold, dtw_score=row["dtw_distance"]/threshold,
                dtw_passed=phrase, two_layer=two, three_layer=two and row["asr_passed"])


def compare(rows, blend):
    result = []
    for row in rows:
        a, b = apply(row, 0.), apply(row, blend)
        result.append({**row, "blend": blend, "b_threshold": b["threshold"],
            "a_dtw": a["dtw_passed"], "b_dtw": b["dtw_passed"],
            "a_two": a["two_layer"], "b_two": b["two_layer"],
            "a_three": a["three_layer"], "b_three": b["three_layer"],
            "b_dtw_score": b["dtw_score"],
            "no_dtw_three": row["voice_passed"] and row["asr_passed"]})
    return result


def metrics(rows):
    fields = ("a_dtw", "b_dtw", "a_two", "b_two", "a_three", "b_three", "no_dtw_three")
    return dict(by_kind=summarize(rows, fields),
        new_negative_accepts_two=sum(r["b_two"] and not r["a_two"] and r["kind"] in KINDS[1:] for r in rows),
        new_negative_accepts_three=sum(r["b_three"] and not r["a_three"] and r["kind"] in KINDS[1:] for r in rows),
        recovered_genuine_two=sum(r["b_two"] and not r["a_two"] and r["kind"] == "self_correct" for r in rows),
        recovered_genuine_three=sum(r["b_three"] and not r["a_three"] and r["kind"] == "self_correct" for r in rows),
        dtw_added_rejections_after_voice_asr=sum(r["no_dtw_three"] and not r["b_three"] for r in rows))


def live_inputs(root):
    times = {"2026-11-02 14:52:08", "2026-11-02 14:52:12", "2026-11-02 14:52:17"}
    history = [r for r in srdp.read_csv_rows(root / "app_data/history.csv") if r["time_text"] in times]
    if {r["time_text"] for r in history} != times or len(history) != 3:
        raise ValueError("The three user-labeled Nov 2 history rows are missing or duplicated")
    fingerprints = {r["model_fingerprint"] for r in history}
    if len(fingerprints) != 1:
        raise ValueError("Expected one live model snapshot")
    fp = next(iter(fingerprints))
    model_path = root / f"app_data/diagnostics/models/{fp}.npz"
    reports = [(p, json.loads(p.read_text(encoding="utf-8")))
               for p in (root / "app_data/diagnostics/enrollments").glob("*.json")]
    report_path, report = next((p, r) for p, r in reports if r["model_fingerprint"] == fp)
    known = {Path(r["file_path"]): r for r in history}
    unknown = [p for stamp in ("144931", "145018")
               for p in (root / "tmp/gui_recordings").glob(f"temp_verify_20261102_{stamp}_*.wav")]
    paths = sorted(set(known) | set(unknown))
    protected = {str(p): digest(p) for p in [model_path, report_path, *paths,
                 *(Path(s["path"]) for s in report["samples"])]}
    for path, row in known.items():
        assert digest(path) == row["audio_sha256"]
    samples = [dict(file_path=str(p), kind="self_correct" if p in known else "unknown",
        actual_speaker=report["speaker_id"] if p in known else "unknown",
        passphrase="芝麻开门" if p in known else "unknown",
        label_source="user confirmed the three correct utterances" if p in known else "user forgot; ASR is not truth") for p in paths]
    return dict(history=history, model_path=model_path, fingerprint=fp, enrollment=report,
                protected=protected, samples=samples)


def run(output, live_root=None):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "tmp_acceptance"):
        raise ValueError("Use a new directory under tmp_acceptance; do not overwrite evidence")
    output.mkdir(parents=True)
    protected = protected_hashes()
    protected.update({str(p): digest(p) for p in (ROOT / "tmp/gui_recordings").glob("*.wav")})
    protected.update({str(p): digest(p) for p in (ROOT / "code/python").glob("*.py")})
    live = live_inputs(live_root) if live_root else None
    if live:
        protected.update(live["protected"])
        dump(output / "live_manifest.json", dict(history=live["history"], enrollment=live["enrollment"], samples=live["samples"]))
    dump(output / "input_hashes.json", protected)
    base = srdp.load_model()
    metadata = srdp.read_metadata(ROOT / "data/demo_audio/metadata_srdp.csv")
    training = [r for r in metadata if r["split"] == "train"]
    queries = [r for r in metadata if r["split"] != "train"]
    raw = {r["file_path"]: srdp.extract_one(ROOT / r["file_path"]).features for r in training}
    snr = {r["file_path"]: srdp.estimate_snr_db(ROOT / r["file_path"])[0] for r in training}
    corpus_manifest = [{**r, "sha256": digest(ROOT / r["file_path"])} for r in metadata]
    dump(output / "corpus_manifest.json", corpus_manifest)
    asr_cache = {}
    profiles = []

    def asr(path):
        sha = digest(path)
        if sha not in asr_cache:
            text = asr_phrase.recognize_fixed_phrase(path)
            asr_cache[sha] = dict(file_path=str(path), audio_sha256=sha, text=text,
                                 passed=asr_phrase.normalize_text(text) == "芝麻开门")
        return asr_cache[sha]

    def score_profile(model, samples, scope):
        if model.passphrase_aggregate != "min" or model.decision_mode != "two_layer" or model.passphrase_accept_score != 1.:
            raise ValueError("This experiment requires the existing min-DTW / two-layer policy")
        pairs = pairwise_distances(model.passphrase_templates, model)
        stats = coverage_check(pairs, model.calibration_mode)
        folder = output / "models" / scope / model.speaker_id
        folder.mkdir(parents=True)
        gui.save_user_model(model, folder / "original.npz")
        dump(folder / "pair_statistics.json", {**stats, "pairs": pairs.tolist(),
             "saved_baseline_threshold": model.passphrase_threshold,
             "template_count": len(model.passphrase_templates)})
        result = []
        for sample in samples:
            path = Path(sample["file_path"])
            if not path.is_absolute():
                path = ROOT / path
            original = gui.predict_with_session_model_baseline(path, model)
            if original.get("rejection_stage") == "input":
                raise ValueError(f"Input rejected; do not silently drop the claim: {path}")
            features = srdp.extract_one_audio_features(path, model).features
            phrase = features[:, srdp.select_dims(model.feature_names, model.passphrase_feature_set)]
            distance = srdp.passphrase_template_distance(phrase, model.passphrase_templates, "min",
                window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
            assert f"{distance:.6f}" == original["passphrase_distance"]
            voice = original["voice_passed"] == "True"
            assert (voice and distance <= model.passphrase_threshold) == (original["decision"] != "REJECT")
            text = asr(path)
            result.append(dict(scope=scope, target=model.speaker_id, file_path=str(path),
                audio_sha256=text["audio_sha256"], actual_speaker=sample.get("actual_speaker", "unknown"),
                phrase_label=sample.get("passphrase", "unknown"), label_source=sample.get("label_source", "corpus metadata convention"),
                kind=sample["kind"], genuine=sample["kind"] == "self_correct",
                model_fingerprint=gui.session_model_fingerprint(model), template_count=len(model.passphrase_templates),
                baseline_threshold=model.passphrase_threshold, pair_q75=stats["pair_q75"],
                voice_passed=voice, voice_distance=original["voice_distance"], voice_score=original["voice_score"],
                dtw_distance=distance, asr_text=text["text"], asr_passed=text["passed"]))
        return result

    development = []
    for speaker in sorted(base.codebooks):
        others = [r for r in training if r["speaker_id"] != speaker]
        mean, std = srdp.zscore_fit([raw[r["file_path"]] for r in others])
        quality = srdp.fit_quality_baseline(others, [raw[r["file_path"]].mean(axis=0) for r in others],
                                          [snr[r["file_path"]] for r in others])
        setup = replace(base, zscore_mean=mean, zscore_std=std, quality_baseline=quality,
            codebooks={}, thresholds={}, passphrase_thresholds={}, passphrase_train_by_speaker={})
        admissions, selected, features = [], [], []
        for row in sorted((r for r in training if r["speaker_id"] == speaker), key=lambda r: r["file_path"]):
            take, matrix = gui.assess_enrollment_take(ROOT / row["file_path"], setup)
            admissions.append(dict(file_path=row["file_path"], accepted=take.accepted, reason=take.reason))
            if take.accepted and len(selected) < gui.ENROLLMENT_REQUIRED:
                selected.append(row["file_path"])
                features.append(matrix)
        profile = dict(target=speaker, enrollment=selected, admissions=admissions,
                       normalization_training=[r["file_path"] for r in others])
        if len(selected) != gui.ENROLLMENT_REQUIRED:
            profiles.append({**profile, "status": "cannot_register"})
            continue
        model = gui.build_session_enrollment_model(speaker, setup, features)
        profiles.append({**profile, "status": "scored", "fingerprint": gui.session_model_fingerprint(model)})
        samples = [{**r, "actual_speaker": r["speaker_id"].removesuffix("_wrong"), "kind": kind_for(r, speaker)} for r in queries]
        development.extend(score_profile(model, samples, "four_take_development"))
        print(f"Four-take development profile scored: {speaker}", flush=True)
    usable = {p["target"] for p in profiles if p["status"] == "scored"}
    fit_rows = [r for r in development if r["actual_speaker"] in usable]
    policy, grid = select_blend(fit_rows)
    policy.update(formula="T_A + blend * max(0, Q75(unordered template pairs) - T_A)",
        candidates=list(BLENDS), development_speakers=sorted(usable), development_claims=len(fit_rows),
        fit_audio=sorted({r["audio_sha256"] for r in fit_rows}),
        exclusion="No live model, live WAV, ASR text, S04 or U01 claim outcome participates in selection",
        evidence="Previously used corpus development; not an independent acceptance test",
        deployment="Offline candidate only; original App thresholds remain unchanged")
    dump(output / "policy.json", policy)
    srdp.write_rows(output / "calibration_grid.csv", grid, list(grid[0]))
    dump(output / "registration_profiles.json", profiles)
    policy_sha = digest(output / "policy.json")
    print(f"Policy fixed before live and 252-claim regression: blend={policy['blend']}", flush=True)

    folds = []
    for held in sorted(usable):
        fitting = [r for r in fit_rows if r["target"] != held and r["actual_speaker"] != held]
        fold_policy, fold_grid = select_blend(fitting)
        held_rows = [r for r in development if r["target"] == held]
        folds.append(dict(held_speaker=held, fit_claims=len(fitting),
            fit_targets=sorted({r["target"] for r in fitting}),
            fit_actual_speakers=sorted({r["actual_speaker"] for r in fitting}),
            fit_audio=sorted({r["audio_sha256"] for r in fitting}),
            blend=fold_policy["blend"], grid=fold_grid,
            metrics=metrics(compare(held_rows, fold_policy["blend"]))))
    dump(output / "speaker_exclusion_folds.json", folds)

    original_claims = []
    for speaker in sorted(base.codebooks):
        model = formal_profile(base, speaker)
        samples = [{**r, "actual_speaker": r["speaker_id"].removesuffix("_wrong"), "kind": kind_for(r, speaker)} for r in queries]
        original_claims.extend(score_profile(model, samples, "saved_nine_template_252"))
    assert len(original_claims) == 252
    original = metrics(compare(original_claims, policy["blend"]))
    assert {k: (v["count"],v["a_two"]) for k,v in original["by_kind"].items()} == {
        "self_correct": (24,17), "self_wrong": (36,1), "other_correct": (84,2), "other_wrong": (108,0)}

    live_rows = []
    if live:
        model = gui.load_user_model(live["model_path"])
        assert gui.session_model_fingerprint(model) == live["fingerprint"]
        samples = [s for s in live["enrollment"]["samples"] if s["accepted"]]
        assert len(samples) == len(model.passphrase_templates)
        for sample, template in zip(samples, model.passphrase_templates):
            path = Path(sample["path"])
            assert digest(path) == sample["audio_sha256"]
            features = srdp.extract_one_audio_features(path, model).features
            np.testing.assert_array_equal(features[:,srdp.select_dims(model.feature_names,model.passphrase_feature_set)],template)
        live_rows = score_profile(model, live["samples"], "live_regression_not_selection")
    dump(output / "candidate_audits.json", [dict(blend=blend,
        four_take=metrics(compare(development, blend)), saved_252=metrics(compare(original_claims, blend)),
        live=metrics(compare(live_rows, blend)),
        live_scores=[dict(file_path=r["file_path"], kind=r["kind"], asr_text=r["asr_text"],
                         **apply(r,blend)) for r in live_rows]) for blend in BLENDS])
    compared = compare(development + original_claims + live_rows, policy["blend"])
    srdp.write_rows(output / "claims.csv", compared, list(compared[0]))
    dump(output / "asr_outputs.json", list(asr_cache.values()))
    assert policy_sha == digest(output / "policy.json")
    for name, value in protected.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        assert digest(path) == value, f"Protected input changed: {path}"
    dev_metrics = metrics(compare(development, policy["blend"]))
    safe = original["new_negative_accepts_two"] == dev_metrics["new_negative_accepts_two"] == 0
    improved = original["recovered_genuine_two"] + dev_metrics["recovered_genuine_two"] > 0
    summary = dict(policy=policy, policy_sha256=policy_sha,
        four_take_development=dev_metrics, saved_252_regression=original,
        live_regression=metrics(compare(live_rows, policy["blend"])),
        protected_files_unchanged=len(protected),
        retrospective_gate_passed=bool(safe and improved),
        deployment="Not deployed; independent new-speaker/recording acceptance is still missing",
        limitations=["Four-take development has only three registrable speakers; S04 cannot enroll under current quality checks",
            "The 252-claim historical regression uses nine templates, not four",
            "Speaker-exclusion folds exclude claim labels; preprocessing still follows the historical other-speaker training protocol",
            "Corpus wrong words are not transcribed and wrong-speaker identities follow metadata convention",
            "Current live correct utterances are known regressions, and the other two live contents/identities remain unknown",
            "No fresh microphone, new user, new device or replay-attack test was performed"])
    dump(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-root", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    run(srdp.project_path(args.output_dir), args.live_root.resolve() if args.live_root else None)
