"""Evaluate new round-two WAVs without fitting or changing the frozen scorer.

JSON manifest: {"recordings": [{"file_path": "recordings/lwl2_01.wav",
"speaker": "lwl2", "sequence": 1, "phrase": "芝麻开门", "confirmed": true,
"sha256": "<64 lowercase hex characters>"}, ...]}. Paths are project-relative
or absolute; labels describe the actual speaker/phrase, confirmed by a human.
Run: python code/python/frozen_round2_eval.py --manifest round2.json
--bundle <frozen-directory> --protocol <next_round_protocol.json>
--output-dir <new-directory>. Use --development-test for synthetic/Mock evidence.
Project-local outputs must be Git-ignored under tmp_round2/; external paths are allowed.
The eight frozen comparators include two_layer, so nine distinct policies run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import gui
import srdp
import live_structure_experiments as structure


ROOT = srdp.PROJECT_ROOT
SPEAKERS = ("lwl2", "lyh2")
CORRECT_PHRASE = "芝麻开门"
KINDS = ("self_correct", "self_wrong", "other_correct", "other_wrong")
ALIAS = "pitch_relative_best_single_1p1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path, root):
    return (root / path).resolve()


def validate_recordings(manifest, holdout, excluded, root):
    rows = manifest.get("recordings", [])
    if len(rows) != holdout["physical_recordings"]:
        raise ValueError("Expected exactly 24 recordings")
    seen_paths, seen_hashes, seen_labels = set(), set(), set()
    previous_paths = {resolve(name, root) for name in excluded}
    previous_hashes = set(excluded.values())
    result = []
    for row in rows:
        required = ("file_path", "speaker", "sequence", "phrase", "confirmed", "sha256")
        if any(key not in row for key in required):
            raise ValueError("Recording has missing labels or SHA-256")
        if row["confirmed"] is not True:
            raise ValueError("Recording must be manually confirmed")
        speaker, sequence = row["speaker"], row["sequence"]
        if speaker not in holdout["speakers"] or type(sequence) is not int or not 1 <= sequence <= 12:
            raise ValueError("Speaker/sequence does not match protocol")
        key = speaker, sequence
        if key in seen_labels or row["phrase"] != holdout["phrases"][sequence - 1]:
            raise ValueError("Speaker/sequence/actual phrase does not match protocol")
        path = resolve(row["file_path"], root)
        if path.suffix.lower() != ".wav" or not path.is_file():
            raise ValueError(f"WAV missing: {path}")
        actual = digest(path)
        if path in seen_paths or actual in seen_hashes:
            raise ValueError(f"duplicate WAV path/content: {path}")
        if actual != row["sha256"]:
            raise ValueError(f"WAV SHA-256 mismatch: {path}")
        if path in previous_paths or actual in previous_hashes:
            raise ValueError(f"WAV reuses previous enrollment/calibration: {path}")
        seen_paths.add(path)
        seen_hashes.add(actual)
        seen_labels.add(key)
        result.append({**row, "path": path})
    if seen_labels != {(speaker, i) for speaker in SPEAKERS for i in range(1, 13)}:
        raise ValueError("Each speaker must have sequences 1 through 12")
    return result


def verify_files(hashes):
    for path, expected in hashes.items():
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen/input SHA-256 mismatch: {path}")


def load_freeze(bundle, protocol_path, root=ROOT):
    bundle, protocol_path = bundle.resolve(), protocol_path.resolve()
    receipt = read_json(bundle / "freeze_receipt.json")
    protected = {bundle / "frozen_candidate.json": receipt["candidate_sha256"],
                 bundle / "bundle_hashes.json": receipt["bundle_hashes_sha256"]}
    verify_files(protected)
    candidate = read_json(bundle / "frozen_candidate.json")
    if candidate["bundle_hash_manifest_sha256"] != receipt["bundle_hashes_sha256"]:
        raise ValueError("Candidate/receipt bundle binding differs")
    protected.update({protocol_path: candidate["independent_protocol_sha256"],
                      bundle / "labels.json": candidate["labels_sha256"]})
    manifest = read_json(bundle / "bundle_hashes.json")
    source_names = {"gui.py", "srdp.py", "speech_frames.py", "dtw_paths.py",
                    "live_fusion_experiments.py", "live_structure_experiments.py"}
    required = {"policies.json", "local_thresholds.json", "source_hashes.json",
                "protocol.json", "pitch_protocol.json", "relative_pitch_protocol.json", "environment.json"}
    required |= {f"source/{name}" for name in source_names}
    required |= {f"models/{speaker}{suffix}.npz" for speaker in SPEAKERS
                 for suffix in ("", "_side", "_enrollment")}
    if not required <= manifest.keys():
        raise ValueError("Frozen bundle hash manifest is incomplete")
    for name, value in manifest.items():
        path = resolve(name, bundle)
        if not path.is_relative_to(bundle):
            raise ValueError("Frozen hash path escapes bundle")
        protected[path] = value
    verify_files(protected)
    sources = read_json(bundle / "source_hashes.json")
    if sources.keys() != source_names or any(sources[n] != manifest[f"source/{n}"] for n in sources):
        raise ValueError("Frozen source binding is incomplete or inconsistent")
    protected.update({Path(__file__).parent / name: value for name, value in sources.items()})
    verify_files(protected)
    protocol, labels = read_json(protocol_path), read_json(bundle / "labels.json")
    holdout = protocol["holdout"]
    if (tuple(holdout["speakers"]) != SPEAKERS or holdout["per_speaker"] != 12
            or len(holdout["phrases"]) != 12 or holdout["physical_recordings"] != 24
            or holdout["paired_account_claims"] != 48
            or any(holdout[k] != 12 for k in ("genuine_correct", "own_wrong", "other_correct", "other_wrong"))):
        raise ValueError("Not the frozen two-person round-two protocol")
    models, sides, thresholds = structure.load_bundle(bundle)
    if set(models) != set(SPEAKERS) or set(labels["models"]) != set(SPEAKERS):
        raise ValueError("Frozen model accounts do not match protocol")
    excluded = {}
    for speaker, model in models.items():
        registration = labels["models"][speaker]
        current_path = resolve(registration["model_path"], root)
        current = gui.load_user_model(current_path)
        fingerprint = gui.session_model_fingerprint(model)
        expected = protocol["model_freeze"][speaker]
        if not (fingerprint == gui.session_model_fingerprint(current) == registration["fingerprint"]
                == candidate["model_fingerprints"][speaker] == expected["fingerprint"]):
            raise ValueError(f"Actual/frozen model fingerprint differs: {speaker}")
        if (model.voice_threshold != expected["voice_threshold"]
                or model.passphrase_threshold != expected["phrase_threshold"]
                or model.decision_mode != "two_layer"):
            raise ValueError(f"Original model thresholds/mode differ: {speaker}")
        protected[current_path] = digest(current_path)
        excluded.update({item["path"]: item["sha256"] for item in registration["enrollment"]})
    excluded.update({item["file_path"]: item["sha256"] for item in labels["recordings"]})
    protected.update({resolve(name, root): value for name, value in excluded.items()})
    verify_files(protected)
    policies = read_json(bundle / "policies.json")
    selected = candidate["selected_variant"]
    if candidate["policy"] != policies[selected]:
        raise ValueError("Selected frozen policy differs from candidate")
    comparators = candidate["comparators"]
    if len(comparators) != 8 or len(set(comparators)) != 8 or not {"two_layer", ALIAS} <= set(comparators):
        raise ValueError("Expected eight frozen comparators including original and alias")
    variants = list(dict.fromkeys(["two_layer", selected, *comparators]))
    if any(name not in policies for name in variants):
        raise ValueError("Frozen comparator policy missing")
    protected[bundle / "freeze_receipt.json"] = digest(bundle / "freeze_receipt.json")
    return {"candidate": candidate, "protocol": protocol, "policies": policies,
            "models": models, "sides": sides, "thresholds": thresholds,
            "variants": variants, "excluded": excluded, "protected": protected}


def raw_scores(path, model):
    """Audit with shared distance functions; never feed these values to frozen policies."""
    features = srdp.extract_one_audio_features(path, model).features
    voice = features[:, srdp.select_dims(model.feature_names, model.feature_set)]
    distance = srdp.quantization_distance(voice, model.codebook)
    if model.voice_metric == "cohort_ratio":
        distance = srdp.speaker_distance_scores({"": distance, **{
            name: srdp.quantization_distance(voice, book)
            for name, book in model.background_codebooks.items()}}, "cohort_ratio")[""]
    phrase = srdp.passphrase_template_distance(
        features[:, srdp.select_dims(model.feature_names, model.passphrase_feature_set)],
        model.passphrase_templates, model.passphrase_aggregate,
        window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
    return {"raw_voice_distance": distance, "raw_phrase_distance": phrase,
            "raw_voice_score": srdp.safe_ratio(distance, model.voice_threshold),
            "raw_phrase_score": srdp.safe_ratio(phrase, model.passphrase_threshold) / model.passphrase_accept_score}


def trial_kind(speaker, target, phrase):
    return ("self_" if speaker == target else "other_") + ("correct" if phrase == CORRECT_PHRASE else "wrong")


def rejection_reasons(values, policy):
    reasons = []
    for field, limit in (("voice_mfcc", 1.), ("phrase_mfcc", policy["phrase_guard"])):
        if values[field] > limit:
            reasons.append(f"{field}>{limit}")
    if policy.get("pitch_gate") and values[policy.get("pitch_stream", "pitch_identity")] > 1.:
        reasons.append("pitch gate failed/unavailable")
    if policy.get("cohort_gate") and values["identity_ratio"] > 1.:
        reasons.append("identity_ratio>1")
    return "; ".join(reasons) or "frozen stream/weighted threshold exceeded"


def score_claim(sample, target, context):
    model = context["models"][target]
    row = {k: sample[k] for k in ("file_path", "speaker", "sequence", "phrase", "sha256")}
    row.update(target=target, kind=trial_kind(sample["speaker"], target, sample["phrase"]),
        model_fingerprint=context["candidate"]["model_fingerprints"][target],
        voice_threshold=model.voice_threshold, phrase_threshold=model.passphrase_threshold,
        app_phrase_limit=model.passphrase_accept_score, scores={}, baseline=None,
        structure_baseline=None, baseline_mismatch_reason="",
        decisions=dict.fromkeys(context["variants"]), reasons={}, boundary_issues=[],
        baseline_consistent=False, status="ok", error="")
    try:
        baseline = gui.predict_with_session_model_baseline(sample["path"], model)
        row["baseline"], row["app_reason"] = baseline["decision"], baseline["reject_reason"]
        if baseline.get("rejection_stage") == "input":
            row.update(status="input_rejected", baseline_consistent=True,
                decisions=dict.fromkeys(context["variants"], "REJECT"),
                reasons=dict.fromkeys(context["variants"], baseline["reject_reason"]))
            return row
        row.update(raw_scores(sample["path"], model))
        values, replayed, _ = structure.score(sample["path"], target, context["models"],
                                              context["sides"], context["thresholds"])
        row["structure_baseline"] = replayed["decision"]
        if row["structure_baseline"] != row["baseline"]:
            row["status"] = "baseline_mismatch"
            row["baseline_mismatch_reason"] = (
                f"structure.score baseline={row['structure_baseline']} differs from independent App={row['baseline']}")
            row["boundary_issues"].append(row["baseline_mismatch_reason"])
        if values is None:
            if row["baseline_mismatch_reason"]:
                return row
            raise ValueError("Frozen score did not reproduce original App input")
        row["scores"] = values
        if any(not math.isfinite(values[k]) for k in structure.fusion.STREAM_NAMES):
            raise ValueError("Nonfinite original/Delta/spectral score")
        if math.isnan(values["pitch_relative"]):
            raise ValueError("Invalid relative pitch score")
        if not math.isfinite(values["pitch_relative"]) and row["status"] == "ok":
            row["status"] = "pitch_unavailable"
        for name in context["variants"]:
            policy = context["policies"][name]
            accepted = bool(structure.apply([values], policy)[0])
            row["decisions"][name] = target if accepted else "REJECT"
            row["reasons"][name] = "" if accepted else rejection_reasons(values, policy)
        two_layer_consistent = row["decisions"]["two_layer"] == row["baseline"]
        row["baseline_consistent"] = two_layer_consistent and not row["baseline_mismatch_reason"]
        if not two_layer_consistent:
            row["boundary_issues"].append("Original App/frozen two_layer decisions differ")
        gates = [("voice_mfcc", "raw_voice_score", 1.)]
        gates += [("phrase_mfcc", "raw_phrase_score", limit) for limit in
                  sorted({context["policies"][name]["phrase_guard"] for name in context["variants"]})]
        for frozen_field, raw_field, limit in gates:
            if (values[frozen_field] <= limit) != (row[raw_field] <= limit):
                row["boundary_issues"].append(f"{frozen_field} raw/display gate differs at {limit}")
    except Exception as error:
        row.update(status="baseline_mismatch" if row["baseline_mismatch_reason"] else "scoring_error",
                   error=f"{type(error).__name__}: {error}",
                   decisions=dict.fromkeys(context["variants"]), baseline_consistent=False)
    return row


def summarize(rows, context, development_test=False):
    metrics = {}
    for name in context["variants"]:
        categories, speakers = {}, {}
        for kind in KINDS:
            subset = [r for r in rows if r["kind"] == kind]
            accepted = sum(r["decisions"][name] not in (None, "REJECT") for r in subset)
            categories[kind] = {"count": len(subset), "accepted": accepted,
                "unscored": sum(r["decisions"][name] is None for r in subset),
                "false_accept": accepted if kind != "self_correct" else 0}
        for speaker in SPEAKERS:
            subset = [r for r in rows if r["speaker"] == speaker and r["kind"] == "self_correct"]
            speakers[speaker] = {"correct_count": len(subset),
                "correct_accepted": sum(r["decisions"][name] == speaker for r in subset),
                "false_reject": sum(r["decisions"][name] == "REJECT" for r in subset),
                "correct_unscored": sum(r["decisions"][name] is None for r in subset),
                "negative_false_accepts": {kind: sum(r["decisions"][name] not in (None, "REJECT")
                    for r in rows if r["speaker"] == speaker and r["kind"] == kind) for kind in KINDS[1:]},
                "target_negative_false_accepts": {kind: sum(r["decisions"][name] not in (None, "REJECT")
                    for r in rows if r["target"] == speaker and r["kind"] == kind) for kind in KINDS[1:]}}
        metrics[name] = {"categories": categories, "speakers": speakers}
    selected = context["candidate"]["selected_variant"]
    keys = ("phrase_guard", "cohort_gate", "pitch_gate", "pitch_stream", "streams", "threshold")
    alias_rule_equal = all(context["policies"][selected].get(k) == context["policies"][ALIAS].get(k) for k in keys)
    alias_decisions_equal = all(r["decisions"][selected] == r["decisions"][ALIAS] for r in rows)
    counts = {kind: sum(r["kind"] == kind for r in rows) for kind in KINDS}
    complete = len(rows) == 48 and all(count == 12 for count in counts.values())
    eligible = complete and alias_rule_equal and alias_decisions_equal and all(
        r["status"] != "scoring_error" and r["baseline_consistent"] and not r["boundary_issues"] for r in rows)
    gains, regressions = 0, 0
    for row in rows:
        expected = row["target"] if row["kind"] == "self_correct" else "REJECT"
        before, after = row["baseline"], row["decisions"][selected]
        gains += before != expected and after == expected
        regressions += before == expected and after is not None and after != expected
    per_person = all(metrics[selected]["speakers"][s]["correct_accepted"] >= sum(
        r["baseline"] == s for r in rows if r["speaker"] == s and r["kind"] == "self_correct") for s in SPEAKERS)
    safe = all(metrics[selected]["categories"][kind]["false_accept"] == 0 for kind in KINDS[1:])
    return {"evidence": "development_test" if development_test else "independent_round_two",
        "claims": len(rows), "category_counts": counts, "metrics": metrics,
        "conclusion_eligible": eligible and not development_test,
        "observed_corrected_claims": gains, "observed_regressed_claims": regressions,
        "gain_counts_usable": eligible and not development_test,
        "benefit_conditions": {"each_person_not_worse": per_person, "zero_negative_accepts": safe,
                               "strict_improvement_without_regression": gains > 0 and regressions == 0,
                               "beats_matched_best_single": False},
        "original_method_benefit_pass": eligible and not development_test and per_person and safe and gains > 0 and regressions == 0,
        "pilot_success": False,
        "fusion_gain_proven": False, "alias_rule_equal": alias_rule_equal,
        "alias_decisions_equal": alias_decisions_equal,
        "interpretation": "Candidate equals pitch_relative_best_single_1p1; no extra fusion gain can be claimed.",
        "scoring_errors": sum(r["status"] == "scoring_error" for r in rows),
        "baseline_mismatches": sum(r["status"] == "baseline_mismatch" for r in rows),
        "boundary_disagreements": sum(bool(r["boundary_issues"]) for r in rows),
        "protocol_success_rules": context["protocol"]["pilot_success_rules"]}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(manifest_path, bundle, protocol_path, output, *, development_test=False, root=ROOT):
    root = root.resolve()
    output = output.resolve()
    if output.exists() or output.is_relative_to(bundle.resolve()) or any(output.is_relative_to((root / name).resolve())
                              for name in ("app_data", "tmp", "data", "models", "code")):
        raise ValueError("Output must be a new directory outside protected input/source directories")
    if output.is_relative_to(root):
        if not output.is_relative_to(root / "tmp_round2"):
            raise ValueError("Project-local output must be under tmp_round2/")
        for name in ("trials.csv", "boundary_disagreements.csv", "summary.json"):
            ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(output / name)],
                                     cwd=root, capture_output=True)
            if ignored.returncode != 0:
                raise ValueError(f"Project-local output must be Git-ignored: {output / name}")
    context = load_freeze(bundle, protocol_path, root)
    context["protected"][manifest_path.resolve()] = digest(manifest_path)
    recordings = validate_recordings(read_json(manifest_path), context["protocol"]["holdout"],
                                     context["excluded"], root)
    context["protected"].update({r["path"]: r["sha256"] for r in recordings})
    rows = [score_claim(sample, target, context) for sample in recordings for target in SPEAKERS]
    verify_files(context["protected"])
    summary = summarize(rows, context, development_test)
    summary["integrity"] = {"all_matched": True, "files_checked": len(context["protected"]),
        "candidate_sha256": digest(bundle / "frozen_candidate.json"),
        "protocol_sha256": digest(protocol_path), "manifest_sha256": digest(manifest_path),
        "protected_sha256": {str(path): value for path, value in context["protected"].items()}}
    summary["limitations"] = ["Human labels/confirmation cannot be verified from the WAV automatically.",
        "SHA/path disjointness detects exact reuse, not re-encoded or edited copies.",
        "Small two-person prototype check; not population FAR or cross-device evidence.",
        "Frozen six-digit score semantics remain unchanged; a precision fix requires a separately reviewed refreeze."]
    selected = context["candidate"]["selected_variant"]
    trials, disagreements = [], []
    for row in rows:
        common = {k: v for k, v in row.items() if k not in ("scores", "decisions", "reasons", "boundary_issues")}
        common.update(scores=json.dumps(json_safe(row["scores"]), ensure_ascii=False, allow_nan=False),
            boundary_issues="; ".join(row["boundary_issues"]),
            candidate_change="unscored" if row["decisions"][selected] is None else
                "unchanged" if row["decisions"][selected] == row["baseline"] else
                "new_reject" if row["decisions"][selected] == "REJECT" else "new_accept",
            change_usable=summary["gain_counts_usable"],
            delta_threshold=context["sides"][row["target"]]["delta"][1],
            pitch_target_center=context["thresholds"][row["target"]]["pitch"]["center"])
        if row["boundary_issues"]:
            disagreements.append(common.copy())
        for variant in context["variants"]:
            trials.append({**common, "variant": variant, "decision": row["decisions"][variant],
                "reject_reason": row["reasons"].get(variant, row["error"]),
                "policy": json.dumps(context["policies"][variant], ensure_ascii=False)})
    fields = list(dict.fromkeys(key for row in trials for key in row))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "trials.csv", trials, fields)
    write_csv(output / "boundary_disagreements.csv", disagreements,
              [key for key in fields if key not in ("variant", "decision", "reject_reason", "policy")])
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--development-test", action="store_true")
    args = parser.parse_args()
    summary = run(args.manifest, args.bundle, args.protocol, args.output_dir,
                  development_test=args.development_test)
    print(json.dumps({key: summary[key] for key in ("evidence", "claims", "conclusion_eligible",
        "original_method_benefit_pass", "pilot_success", "fusion_gain_proven")}, ensure_ascii=False))
    return 0 if args.development_test or summary["conclusion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
