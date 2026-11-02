"""Read-only corpus regression, not a fresh/blind acceptance test.

Run from the project root: code/python/venv/Scripts/python.exe
code/python/acceptance_baseline.py --output tmp_acceptance/baseline.json
Only the requested JSON is written; existing outputs cannot be overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import gui
import srdp
from phrase_content_experiments import formal_profile


ROOT = srdp.PROJECT_ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for name in ("app_data", "data", "models", "results")
            for p in (ROOT / name).rglob("*") if p.is_file()}


def summarize(trials):
    result = {}
    for kind in sorted({r["kind"] for r in trials}):
        group = [r for r in trials if r["kind"] == kind]
        accepted = sum(r["accepted"] for r in group)
        result[kind] = dict(count=len(group), accepted=accepted,
                            acceptance_rate=accepted / len(group))
        field = "false_reject_rate" if kind == "self_correct" else "false_accept_rate"
        result[kind][field] = (len(group) - accepted if kind == "self_correct" else accepted) / len(group)
    return result


def kind_for(row, target):
    # Corpus convention only: the actual wrong words are not transcribed.
    own = row["speaker_id"].removesuffix("_wrong") == target
    correct = row["passphrase"] == "芝麻开门"
    return ("self_" if own else "other_") + ("correct" if correct else "wrong")


def replay_profile(model, queries):
    trials = []
    for row in queries:
        result = gui.predict_with_session_model_baseline(ROOT / row["file_path"], model)
        trials.append({"file_path": row["file_path"], "target": model.speaker_id,
            "kind": kind_for(row, model.speaker_id), "speaker_label": row["speaker_id"],
            "environment": row["environment"], "phrase_label": row["passphrase"],
            "accepted": result["decision"] != "REJECT",
            **{k: result.get(k, "") for k in ("decision", "voice_score", "passphrase_score",
                "voice_distance", "passphrase_distance", "rejection_stage")}})
    return trials


def dominance(trials):
    pairs = []
    for positive in (r for r in trials if r["kind"] == "self_correct" and r["voice_score"]):
        for negative in trials:
            if negative["target"] != positive["target"] or negative["kind"] == "self_correct" or not negative["voice_score"]:
                continue
            if all(float(negative[k]) + 1e-6 < float(positive[k]) for k in ("voice_score", "passphrase_score")):
                pairs.append({"target": positive["target"], "positive": positive["file_path"],
                              "negative": negative["file_path"], "negative_kind": negative["kind"]})
    return pairs


def run(output):
    if output.exists() or not output.resolve().is_relative_to(ROOT / "tmp_acceptance"):
        raise ValueError("Use a new output under tmp_acceptance/; never overwrite evidence")
    before = protected_hashes()
    sources = {p.name: digest(p) for p in (ROOT / "code/python").glob("*.py")}
    base = srdp.load_model()
    rows = srdp.read_metadata(ROOT / "data/demo_audio/metadata_srdp.csv")
    manifest, raw, pcm_hashes = [], [], []
    for row in rows:
        path = ROOT / row["file_path"]
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        pcm = hashlib.sha256(str(rate).encode() + audio.tobytes()).hexdigest()
        pcm_hashes.append(pcm)
        manifest.append({**row, "sha256": digest(path), "decoded_pcm_sha256": pcm})
        raw.append(srdp.extract_one(path).features)
    if len(set(pcm_hashes)) != len(pcm_hashes):
        raise ValueError("Duplicate decoded recordings require a grouped split")
    names, cached_rows, cached = srdp.load_feature_archive(ROOT / "data/features/srdp_features.npz")
    assert [r["file_path"] for r in rows] == [r["file_path"] for r in cached_rows]
    for matrix, archive in zip(raw, cached):
        np.testing.assert_array_equal(srdp.zscore_apply(matrix, base.zscore_mean, base.zscore_std), archive)
    train_indices = [i for i, r in enumerate(rows) if r["split"] == "train"]
    mean, std = srdp.zscore_fit([raw[i] for i in train_indices])
    np.testing.assert_array_equal(mean, base.zscore_mean)
    np.testing.assert_array_equal(std, base.zscore_std)
    queries = [r for r in rows if r["split"] != "train"]
    batch = [srdp.predict_one(ROOT / r["file_path"], base, metadata=r) for r in queries]
    old_results = {r["file_path"]: r for r in srdp.read_csv_rows(ROOT / "results/tables/lbg_vq_recognition_results.csv")}
    saved_claims, app_claims, folds = [], [], []
    for speaker in sorted(base.codebooks):
        saved_claims.extend(replay_profile(formal_profile(base, speaker), queries))
        training = [i for i in train_indices if rows[i]["speaker_id"] != speaker]
        candidates = sorted((i for i in train_indices if rows[i]["speaker_id"] == speaker),
                            key=lambda i: rows[i]["file_path"])
        mean, std = srdp.zscore_fit([raw[i] for i in training])
        quality = srdp.fit_quality_baseline([rows[i] for i in training],
            [raw[i].mean(axis=0) for i in training],
            [srdp.estimate_snr_db(ROOT / rows[i]["file_path"])[0] for i in training])
        # No target-speaker training statistics, books, or thresholds survive in this fold.
        setup = replace(base, zscore_mean=mean, zscore_std=std, quality_baseline=quality,
                        codebooks={}, thresholds={}, passphrase_thresholds={}, passphrase_train_by_speaker={})
        admissions, selected, features = [], [], []
        for index in candidates:
            take, matrix = gui.assess_enrollment_take(ROOT / rows[index]["file_path"], setup)
            admissions.append(dict(file_path=rows[index]["file_path"], accepted=take.accepted, reason=take.reason))
            if take.accepted and len(selected) < gui.ENROLLMENT_REQUIRED:
                selected.append(index)
                features.append(matrix)
        pools = [set(manifest[i]["decoded_pcm_sha256"] for i in group)
                 for group in (training, selected, [i for i, r in enumerate(rows) if r["split"] != "train"])]
        assert all(not pools[i] & pools[j] for i in range(3) for j in range(i))
        fold = dict(target=speaker, training=[rows[i]["file_path"] for i in training],
                    enrollment=[rows[i]["file_path"] for i in selected], admissions=admissions,
                    unused=[rows[i]["file_path"] for i in candidates if i not in selected])
        if len(selected) < gui.ENROLLMENT_REQUIRED:
            fold["status"] = "cannot_register"
        else:
            model = gui.build_session_enrollment_model(speaker, setup, features)
            claims = replay_profile(model, queries)
            app_claims.extend(claims)
            fold.update(status="scored", model_fingerprint=gui.session_model_fingerprint(model),
                voice_threshold=model.voice_threshold, phrase_threshold=model.passphrase_threshold,
                consistency_advisory=gui.enrollment_consistency_check(features, setup),
                metrics=summarize(claims))
        folds.append(fold)
        print(f"{speaker}: {fold['status']}, qualified={sum(r['accepted'] for r in admissions)}", flush=True)
    assert before == protected_hashes(), "Protected project files changed"
    assert sources == {p.name: digest(p) for p in (ROOT / "code/python").glob("*.py")}
    result = dict(protocol="corpus_regression_v1", evidence="previously_used_data_not_blind_test",
        runtime=dict(python=platform.python_version(), numpy=np.__version__, librosa=librosa.__version__),
        source_sha256=sources, protected_files_unchanged=len(before), recordings=manifest,
        rules=dict(official="Saved four-speaker model; original non-train 63 WAVs, no fitting",
            claimed_identity="Same 63 WAVs against each saved identity; 252 claims, not 252 independent WAVs",
            new_registration="Per target: other 27 training WAVs fit normalization/quality; first four qualified target train WAVs in lexical path order enroll and LOO-calibrate; all 63 non-train WAVs test; no parameter search",
            advisory="Consistency flags are advisory; retained and reported, no silent sample removal",
            acceptance="Future blind comparison must improve correct acceptance without increasing any negative-category false accepts, and have zero observed false accepts in the pilot",
            limitations=["Wrong words and recording sessions/devices are unspecified in the 99-row corpus",
                "Sxx_wrong identity follows metadata convention, not independent speaker verification",
                "Cross-speaker folds share test WAVs; report counts, not population confidence",
                "Registration and LOO calibration share four WAVs exactly as App does; no separate negative calibration",
                "Previously inspected data can support regression only, not final acceptance"]),
        archive_exactly_matches_raw=True,
        official_identification=dict(count=len(batch), correct=sum(r["decision"] == r["expected_decision"] for r in batch),
            by_split={split: dict(count=sum(r["split"] == split for r in batch),
                correct=sum(r["split"] == split and r["decision"] == r["expected_decision"] for r in batch))
                for split in sorted({r["split"] for r in batch})},
            historical_decision_mismatches=sum(r["decision"] != old_results[r["file_path"]]["decision"] for r in batch)),
        saved_identity=dict(metrics=summarize(saved_claims), trials=saved_claims),
        app_registration=dict(registered=sum(f["status"] == "scored" for f in folds), attempted=len(folds),
            metrics=summarize(app_claims), folds=folds, trials=app_claims, score_dominance=dominance(app_claims)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"official": result["official_identification"], "saved_identity": summarize(saved_claims),
        "app_registration": summarize(app_claims), "dominance_pairs": len(dominance(app_claims)),
        "protected_files_unchanged": len(before)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(srdp.project_path(args.output))
