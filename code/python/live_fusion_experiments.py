"""Training-only score fusion for explicitly labeled live recordings, without deployment."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import gui
import srdp


STREAM_NAMES = ("voice_mfcc", "phrase_mfcc", "phrase_delta", "phrase_spectral")
VARIANTS = ("two_layer", "guard_only_1p1", "weighted_strict", "weighted_1p1", "best_single_1p1", "feature_fusion_1p1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def weight_grid(variant):
    if variant.startswith("weighted"):
        return [(w, 1 - w, 0., 0.) for w in (.3, .5, .7)]
    if variant.startswith("best_single"):
        return [tuple(float(i == j) for i in range(4)) for j in range(4)]
    return [tuple(v / 4 for v in values) for values in itertools.product(range(5), repeat=4)
        if sum(values) == 4 and values[0] >= 1 and values[1] >= 1 and sum(values[2:]) >= 1]


def apply_policy(scores, policy):
    x = np.asarray(scores, dtype=float)
    gates = (x[:, 0] <= 1.) & (x[:, 1] <= policy["phrase_guard"])
    if policy["variant"] in {"two_layer", "guard_only_1p1"}:
        return gates, np.full(len(x), np.nan)
    values = x @ np.array(policy["weights"])
    return gates & (values <= policy["threshold"]), values


def fit_policy(scores, genuine, variant):
    x, y = np.asarray(scores, dtype=float), np.asarray(genuine, dtype=bool)
    if not y.any() or y.all() or not np.isfinite(x).all():
        raise ValueError("Fusion fitting requires finite labeled genuine and negative trials")
    guard = 1. if variant in {"two_layer", "weighted_strict"} else 1.1
    if variant in {"two_layer", "guard_only_1p1"}:
        return {"variant": variant, "phrase_guard": guard, "weights": [], "threshold": 1.}
    best = None
    for weights in weight_grid(variant):
        values = x @ np.array(weights)
        boundaries = sorted({0., 1., *values[(x[:, 0] <= 1.) & (x[:, 1] <= guard) & (values <= 1.)].tolist()})
        thresholds = [0., 1.] + [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
        for threshold in thresholds:
            policy = {"variant": variant, "phrase_guard": guard, "weights": list(weights), "threshold": threshold}
            accepted, _ = apply_policy(x, policy)
            fa, fr = int(np.sum(accepted & ~y)), int(np.sum(~accepted & y))
            rank = fa, fr, sum(w > 0 for w in weights), abs(weights[0] - .5), -threshold, weights
            if best is None or rank < best[0]:
                best = rank, {**policy, "fit_false_accept": fa, "fit_false_reject": fr}
    return best[1]


def grouped_predictions(scores, genuine, groups, variant):
    predictions, fits = np.zeros(len(groups), dtype=bool), []
    for group in sorted(set(groups)):
        train = np.array([g != group for g in groups])
        policy = fit_policy(np.asarray(scores)[train], np.asarray(genuine)[train], variant)
        predictions[~train] = apply_policy(np.asarray(scores)[~train], policy)[0]
        fits.append({"held_audio": group, "fit_audio": sorted({g for g in groups if g != group}), **policy})
    return predictions, fits


def build_side_streams(model, enrollment):
    result = {}
    for name, dims in (("delta", slice(12, 24)), ("spectral", slice(24, 31))):
        matrices = [x[:, dims] for x in enrollment]
        distances = [srdp.passphrase_template_distance(x, matrices, model.passphrase_aggregate,
            exclude_index=i, window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty,
            max_run=model.dtw_max_run) for i, x in enumerate(matrices)]
        values = np.array(distances)
        threshold = srdp.calibration_threshold(distances, 1.) if model.calibration_mode == "leave_one_recording_out" else float(max(values.mean() + 2 * values.std(), values.max() + 1e-6))
        result[name] = matrices, threshold
    return result


def score_audio(path, model, side):
    result = gui.predict_with_session_model_baseline(path, model)
    if result.get("rejection_stage") == "input":
        return None, result
    features = srdp.extract_one_audio_features(path, model).features
    scores = [float(result["voice_score"]), float(result["passphrase_score"]) / model.passphrase_accept_score]
    for name, dims in (("delta", slice(12, 24)), ("spectral", slice(24, 31))):
        templates, threshold = side[name]
        distance = srdp.passphrase_template_distance(features[:, dims], templates, model.passphrase_aggregate,
            window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
        scores.append(srdp.safe_ratio(distance, threshold))
    # The displayed six-digit ratio must not round an original voice rejection into a pass.
    if result["voice_passed"] == "False" and scores[0] <= 1.:
        scores[0] = float(np.nextafter(1., np.inf))
    return scores, result


def load_bundle(directory, target):
    model = gui.load_user_model(directory / "models" / f"{target}.npz")
    with np.load(directory / "models" / f"{target}_side.npz", allow_pickle=False) as archive:
        side = {name: ([archive[f"{name}_{i}"].copy() for i in range(int(archive["count"]))], float(archive[f"{name}_threshold"])) for name in ("delta", "spectral")}
    return model, side


def replay(directory, path, target):
    integrity = json.loads((directory / "bundle_hashes.json").read_text(encoding="utf-8"))
    if any(digest(directory / name) != value for name, value in integrity.items()):
        raise ValueError("Frozen fusion parameters changed")
    model, side = load_bundle(directory, target)
    scores, baseline = score_audio(path, model, side)
    policies = json.loads((directory / "policies.json").read_text(encoding="utf-8"))
    return {"target": target, "baseline": baseline["decision"], "scores": scores,
        "decisions": {name: target if scores is not None and apply_policy([scores], policy)[0][0] else "REJECT" for name, policy in policies.items()}}


def run(manifest_path, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_json(output / "labels.json", manifest)
    protected = {str(manifest_path.resolve()): digest(manifest_path)}
    snapshots = output / "models"
    snapshots.mkdir()
    models, sides = {}, {}
    for target, registration in manifest["models"].items():
        path = srdp.project_path(registration["model_path"])
        model = gui.load_user_model(path)
        if gui.session_model_fingerprint(model) != registration["fingerprint"]:
            raise ValueError("Current model changed from collection manifest")
        if model.decision_mode != "two_layer" or model.passphrase_feature_set != "mfcc":
            raise ValueError("This experiment requires the saved MFCC/two-layer control")
        protected[str(path.resolve())] = digest(path)
        enrollment = []
        for sample in registration["enrollment"]:
            path = srdp.project_path(sample["path"])
            if digest(path) != sample["sha256"]:
                raise ValueError("Enrollment WAV changed")
            protected[str(path.resolve())] = digest(path)
            enrollment.append(srdp.extract_one_audio_features(path, model).features)
        models[target], sides[target] = model, build_side_streams(model, enrollment)
        gui.save_user_model(model, snapshots / f"{target}.npz")
        arrays = {"count": len(enrollment)}
        for name, (matrices, threshold) in sides[target].items():
            arrays[f"{name}_threshold"] = threshold
            arrays.update({f"{name}_{i}": matrix for i, matrix in enumerate(matrices)})
        np.savez_compressed(snapshots / f"{target}_side.npz", **arrays)
    rows = []
    for sample in manifest["recordings"]:
        path = srdp.project_path(sample["file_path"])
        if digest(path) != sample["sha256"]:
            raise ValueError("Query WAV changed")
        protected[str(path.resolve())] = digest(path)
        for target, model in models.items():
            scores, baseline = score_audio(path, model, sides[target])
            known = sample["phrase"] != "unknown"
            genuine = known and sample["speaker"] == target and sample["phrase"] == "芝麻开门"
            kind = "unknown" if not known else "self_correct" if genuine else "self_wrong" if sample["speaker"] == target else "other_correct" if sample["phrase"] == "芝麻开门" else "other_wrong"
            row = {"file_path": sample["file_path"], "audio_sha256": sample["sha256"], "speaker": sample["speaker"],
                "sequence": sample["sequence"], "phrase": sample["phrase"], "target": target,
                "kind": kind, "genuine": genuine, "labeled": known,
                "model_fingerprint": gui.session_model_fingerprint(model), "baseline": baseline["decision"],
                "input_rejected": scores is None, **dict(zip(STREAM_NAMES, scores or [float("inf")] * 4))}
            if target == sample["speaker"]:
                if baseline["decision"] != sample["original_decision"]:
                    raise ValueError("Own-account replay does not match history")
                for field, value in (("voice_mfcc", sample["original_voice_score"]), ("phrase_mfcc", sample["original_phrase_score"])):
                    if abs(row[field] - float(value)) > 2e-6:
                        raise ValueError("Own-account score differs from history")
            rows.append(row)
    labeled = [r for r in rows if r["labeled"]]
    if any(r["input_rejected"] for r in labeled):
        raise ValueError("A labeled calibration recording failed the input check; do not silently fit infinities")
    x = np.array([[r[k] for k in STREAM_NAMES] for r in labeled])
    y = np.array([r["genuine"] for r in labeled])
    groups = [r["audio_sha256"] for r in labeled]
    summaries, speaker_summaries, changes, predictions, folds, policies = [], [], [], [], [], {}
    all_scores = np.array([[r[k] for k in STREAM_NAMES] for r in rows])
    for variant in VARIANTS:
        policy = fit_policy(x, y, variant)
        policies[variant] = policy
        cv, fits = grouped_predictions(x, y, groups, variant)
        folds += [{"variant": variant, **fit} for fit in fits]
        fitted, _ = apply_policy(all_scores, policy)
        for stage, subset, decisions in (("calibration_fit", rows, fitted), ("grouped_loo", labeled, cv)):
            for kind in sorted({r["kind"] for r in subset}):
                indices = [i for i, r in enumerate(subset) if r["kind"] == kind]
                summaries.append({"stage": stage, "variant": variant, "kind": kind, "count": len(indices),
                    "accepted": sum(bool(decisions[i]) for i in indices), "labeled": kind != "unknown"})
            for speaker, kind in sorted({(r["speaker"], r["kind"]) for r in subset}):
                indices = [i for i, r in enumerate(subset) if (r["speaker"], r["kind"]) == (speaker, kind)]
                speaker_summaries.append({"stage": stage, "variant": variant, "speaker": speaker,
                    "kind": kind, "count": len(indices), "accepted": sum(bool(decisions[i]) for i in indices),
                    "labeled": kind != "unknown"})
            for row, accepted in zip(subset, decisions):
                decision = row["target"] if accepted and not row["input_rejected"] else "REJECT"
                after = {**row, "stage": stage, "variant": variant, "decision": decision}
                predictions.append(after)
                if decision != row["baseline"]:
                    changes.append({**after, "change": "new_accept" if accepted else "new_reject"})
    for filename, records in (("scores.csv", rows), ("summary.csv", summaries), ("speaker_metrics.csv", speaker_summaries), ("predictions.csv", predictions), ("changes.csv", changes)):
        if records:
            srdp.write_rows(output / filename, records, list(records[0]))
    write_json(output / "policies.json", policies)
    write_json(output / "folds.json", folds)
    write_json(output / "protocol.json", {"streams": STREAM_NAMES, "variants": VARIANTS,
        "fitting": "Only explicitly labeled first-round calibration queries. Both target claims of one WAV are excluded together in grouped leave-one-WAV-out.",
        "bounds": "Original MFCC-VQ voice ratio <=1 always; phrase MFCC cap is 1 (strict) or predeclared 1.1; auxiliary thresholds use enrollment recordings only.",
        "selection": "Small predeclared nonnegative weight grid; minimize calibration false accepts, then false rejects. No fitting on unknown lwl2 content.",
        "controls": "Guard-only control separates cap widening from weighted fusion; best-single control checks whether multi-feature fusion outperforms one auxiliary stream.",
        "labeled_speakers": sorted({r["speaker"] for r in labeled}),
        "labeled_wavs": len(set(groups)), "unknown_claims": len(rows) - len(labeled),
        "limitations": "No second-round holdout results. LOO remains internal calibration evidence, not independent final testing.",
        "deployment": "None. Frozen candidates only, not valid replacements for App user models."})
    write_json(output / "input_hashes.json", protected)
    write_json(output / "bundle_hashes.json", {str(p.relative_to(output)).replace("\\", "/"): digest(p) for p in [output / "policies.json", *snapshots.glob("*.npz")]})
    if any(digest(Path(name)) != value for name, value in protected.items()):
        raise ValueError("Live inputs changed during experiment")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--target", choices=("lwl2", "lyh2"))
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(srdp.project_path(args.replay), srdp.project_path(args.wav), args.target), ensure_ascii=False))
    else:
        run(srdp.project_path(args.manifest), srdp.project_path(args.output_dir))
