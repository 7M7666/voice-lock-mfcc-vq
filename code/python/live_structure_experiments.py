"""Exploratory live cohort/local-DTW ablations; frozen App models are read-only."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import gui
import srdp
import live_fusion_experiments as fusion
from dtw_paths import alignment, path_statistics
from speech_frames import energy_activity, mask_for_region


LOCAL_DIMS = {"local_mfcc": slice(0, 12), "local_delta": slice(12, 24)}
SINGLES = (*fusion.STREAM_NAMES, "identity_ratio", *LOCAL_DIMS)
NEW_VARIANTS = (
    "cohort_only_strict", "cohort_only_1p1",
    "cohort_local_mfcc_fixed_1p1", "cohort_local_delta_fixed_1p1",
    "cohort_local_fusion_fixed_1p1", "best_structural_single_1p1",
    "cohort_best_single_1p1", "cohort_local_fusion_fit_1p1",
)
VARIANTS = (*fusion.VARIANTS, *NEW_VARIANTS)
PITCH_VARIANTS = ("pitch_only_strict", "pitch_only_1p1", "pitch_local_mfcc_fixed_1p1",
    "pitch_local_delta_fixed_1p1", "pitch_local_fusion_fixed_1p1", "pitch_best_single_1p1",
    "pitch_local_fusion_fit_1p1", "best_pitch_single_1p1")
RELATIVE_VARIANTS = ("pitch_relative_only_strict", "pitch_relative_only_1p1",
    "pitch_relative_local_mfcc_fixed_1p1", "pitch_relative_local_delta_fixed_1p1",
    "pitch_relative_local_fusion_fixed_1p1", "pitch_relative_best_single_1p1",
    "pitch_relative_local_fusion_fit_1p1", "best_relative_single_1p1")
DELTA_CONTROLS = ("delta_only_fixed_1p1", "pitch_delta_fixed_1p1", "pitch_relative_delta_fixed_1p1")


class PitchUnavailable(ValueError):
    pass


def pitch_descriptor(path, model):
    raw, sr = srdp.load_audio(path)
    _, start, end = srdp.speech_region(raw, sr, model.vad_mode)
    region = raw[start:end].astype(np.float64)
    region -= region.mean()
    f0 = srdp.librosa.yin(region, fmin=65., fmax=400., sr=sr, frame_length=1024,
        hop_length=srdp.HOP_LENGTH)
    frames = srdp.librosa.util.frame(np.pad(region, (512, 512)), frame_length=1024,
        hop_length=srdp.HOP_LENGTH).T
    active, _, _, _ = energy_activity(raw, sr)
    active = mask_for_region(active, start / sr, len(frames), sr)
    voiced = []
    for hz, frame, is_active in zip(f0, frames, active):
        if not is_active:
            continue
        lag = int(round(sr / hz))
        a, b = frame[:-lag], frame[lag:]
        periodicity = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        if periodicity >= .6:
            voiced.append(float(hz))
    if len(voiced) < 8:
        raise PitchUnavailable("Insufficient periodic active frames for experimental pitch")
    logs = np.log2(voiced)
    return {"log2_median_hz": float(np.median(logs)), "median_hz": float(2 ** np.median(logs)),
        "log2_iqr": float(np.percentile(logs, 75) - np.percentile(logs, 25)),
        "voiced_frames": len(voiced), "region_frames": len(frames)}


def local_costs(sample, template, path):
    """Equal template-frame weights prevent repeated cheap frames diluting a bin."""
    if not len(path) or len(template) < 4:
        raise ValueError("Local scoring requires a complete alignment and four template frames")
    costs = np.linalg.norm(sample[path[:, 0]] - template[path[:, 1]], axis=1)
    counts = np.bincount(path[:, 1], minlength=len(template))
    if np.any(counts == 0):
        raise ValueError("Alignment does not cover the entire template")
    per_frame = np.bincount(path[:, 1], weights=costs, minlength=len(template)) / counts
    return np.array([part.mean() for part in np.array_split(per_frame, 4)])


def local_distance(sample, templates, model, *, exclude=None):
    distances, details = [], []
    for index, template in enumerate(templates):
        if index == exclude:
            continue
        mean, path = alignment(sample, template, window_ratio=model.dtw_window_ratio,
            warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
        bins = local_costs(sample, template, path)
        distance = float(bins.max())
        distances.append(distance)
        details.append({"template": index, "mean_cost": mean, "worst_bin_cost": distance,
            **{f"bin_{i + 1}": float(value) for i, value in enumerate(bins)},
            **path_statistics(path, len(sample), len(template))})
    return srdp.aggregate_distances(distances, model.passphrase_aggregate), details


def build_local_thresholds(enrollment, model):
    thresholds, audit = {}, []
    for name, dims in LOCAL_DIMS.items():
        matrices = [matrix[:, dims] for matrix in enrollment]
        values = [local_distance(matrix, matrices, model, exclude=i)[0] for i, matrix in enumerate(matrices)]
        if model.calibration_mode == "leave_one_recording_out":
            threshold = srdp.calibration_threshold(values, 1.)
        else:
            threshold = max(np.mean(values) + 2 * np.std(values), max(values) + 1e-6)
        thresholds[name] = float(threshold)
        audit += [{"stream": name, "held_enrollment": i, "distance": value,
            "threshold": float(threshold)} for i, value in enumerate(values)]
    return thresholds, audit


def check_coordinates(models):
    first = next(iter(models.values()))
    for model in models.values():
        if (model.feature_names != first.feature_names or model.feature_set != first.feature_set
                or model.vad_mode != first.vad_mode or model.cepstral_normalization != first.cepstral_normalization
                or not np.array_equal(model.zscore_mean, first.zscore_mean)
                or not np.array_equal(model.zscore_std, first.zscore_std)):
            raise ValueError("Live cohort codebooks require identical feature coordinates/preprocessing")


def score(path, target, models, sides, thresholds):
    model = models[target]
    original, baseline = fusion.score_audio(path, model, sides[target])
    if original is None:
        return None, baseline, []
    extracted = srdp.extract_one_audio_features(path, model)
    matrix = extracted.features
    dims = srdp.select_dims(model.feature_names, model.feature_set)
    distances = {owner: srdp.quantization_distance(matrix[:, dims], profile.codebook)
        for owner, profile in models.items()}
    rival = min((owner for owner in models if owner != target), key=distances.get)
    values = dict(zip(fusion.STREAM_NAMES, original))
    values["identity_ratio"] = srdp.safe_ratio(distances[target], distances[rival])
    values["target_vq_distance"], values["rival_vq_distance"] = distances[target], distances[rival]
    values["rival"] = rival
    if "pitch" in thresholds[target]:
        pitch = thresholds[target]["pitch"]
        try:
            descriptor = pitch_descriptor(path, model)
        except PitchUnavailable:
            values.update(pitch_identity=float("inf"), pitch_relative=float("inf"), pitch_median_hz="", pitch_voiced_frames=0)
        else:
            deviation = abs(descriptor["log2_median_hz"] - pitch["center"])
            values["pitch_identity"] = deviation / pitch["threshold"]
            rival_deviation = min(abs(descriptor["log2_median_hz"] - item["pitch"]["center"])
                for owner, item in thresholds.items() if owner != target)
            values["pitch_relative"] = deviation / max(rival_deviation, 1e-6)
            values["pitch_median_hz"] = descriptor["median_hz"]
            values["pitch_voiced_frames"] = descriptor["voiced_frames"]
    details = []
    templates = sides[target]["enrollment"]
    for name, dims in LOCAL_DIMS.items():
        distance, pairs = local_distance(matrix[:, dims], [x[:, dims] for x in templates], model)
        values[name] = srdp.safe_ratio(distance, thresholds[target][name])
        details += [{"stream": name, **pair} for pair in pairs]
    raw, sr = srdp.load_audio(path)
    _, context, _, _ = energy_activity(raw, sr)
    active = mask_for_region(context, extracted.voiced_start_sec, len(matrix), sr)
    nearest = min((p for p in details if p["stream"] == "local_mfcc"), key=lambda p: p["mean_cost"])
    _, path_indices = alignment(matrix[:, :12], templates[nearest["template"]][:, :12],
        window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
    values["inactive_query_fraction"] = float(np.mean(~active))
    values["inactive_path_fraction"] = float(np.mean(~active[path_indices[:, 0]]))
    return values, baseline, details


def apply(rows, policy):
    if policy["variant"] in fusion.VARIANTS:
        return fusion.apply_policy([[r[k] for k in fusion.STREAM_NAMES] for r in rows], policy)[0]
    accepted = np.array([not r.get("input_rejected", False) and r["voice_mfcc"] <= 1.
        and r["phrase_mfcc"] <= policy["phrase_guard"] for r in rows])
    if policy["cohort_gate"]:
        accepted &= np.array([r["identity_ratio"] <= 1. for r in rows])
    if policy.get("pitch_gate"):
        accepted &= np.array([r[policy.get("pitch_stream", "pitch_identity")] <= 1. for r in rows])
    if policy["streams"]:
        values = np.array([[r[k] for k in policy["streams"]] for r in rows]).mean(axis=1)
        accepted &= values <= policy["threshold"]
    return accepted


def fit(rows, variant):
    if variant in fusion.VARIANTS:
        return fusion.fit_policy([[r[k] for k in fusion.STREAM_NAMES] for r in rows],
            [r["genuine"] for r in rows], variant)
    policy = {"variant": variant, "phrase_guard": 1. if variant.endswith("strict") else 1.1,
        "cohort_gate": variant.startswith("cohort"), "pitch_gate": variant.startswith("pitch"),
        "pitch_stream": "pitch_relative" if "relative" in variant else "pitch_identity",
        "streams": [], "threshold": 1.}
    if variant in DELTA_CONTROLS:
        return {**policy, "streams": ["phrase_delta"]}
    if "only" in variant:
        return policy
    streams = [["local_mfcc"], ["local_delta"]] if "best_single" in variant else [["local_mfcc", "local_delta"]]
    if "best_single" in variant or variant.startswith(("best_structural", "best_pitch", "best_relative")):
        singles = SINGLES
        if variant.startswith("best_pitch"):
            singles = (*singles, "pitch_identity")
        elif variant.startswith("best_relative"):
            singles = (*singles, "pitch_identity", "pitch_relative")
        streams = [[name] for name in singles if not (policy["cohort_gate"] and name == "identity_ratio")
            and not (policy["pitch_gate"] and name.startswith("pitch_"))]
    elif "fusion" not in variant:
        streams = [["local_mfcc" if "local_mfcc" in variant else "local_delta"]]
    if "fixed" in variant:
        return {**policy, "streams": streams[0]}
    y = np.array([r["genuine"] for r in rows], dtype=bool)
    if not y.any() or y.all():
        raise ValueError("Fitting requires positive and negative labels")
    best = None
    for names in streams:
        values = np.array([[r[k] for k in names] for r in rows]).mean(axis=1)
        eligible = apply(rows, policy)
        boundaries = sorted({0., 1., *values[eligible & (values <= 1.)].tolist()})
        for threshold in [0., 1., *[(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]]:
            candidate = {**policy, "streams": names, "threshold": threshold}
            decisions = apply(rows, candidate)
            fa, fr = int(np.sum(decisions & ~y)), int(np.sum(~decisions & y))
            rank = fa, fr, len(names), -threshold, names
            if best is None or rank < best[0]:
                best = rank, {**candidate, "fit_false_accept": fa, "fit_false_reject": fr}
    return best[1]


def grouped_predictions(rows, variant):
    predictions = np.zeros(len(rows), dtype=bool)
    folds = []
    for held in sorted({r["audio_sha256"] for r in rows}):
        train = [r for r in rows if r["audio_sha256"] != held]
        policy = fit(train, variant)
        indices = [i for i, r in enumerate(rows) if r["audio_sha256"] == held]
        predictions[indices] = apply([rows[i] for i in indices], policy)
        folds.append({"held_audio": held, "fit_audio": sorted({r["audio_sha256"] for r in train}), **policy})
    return predictions, folds


def load_bundle(directory):
    integrity = json.loads((directory / "bundle_hashes.json").read_text(encoding="utf-8"))
    if any(fusion.digest(directory / name) != value for name, value in integrity.items()):
        raise ValueError("Frozen structural bundle changed")
    sources = directory / "source_hashes.json"
    if sources.exists():
        current = json.loads(sources.read_text(encoding="utf-8"))
        if any(fusion.digest(Path(__file__).parent / name) != value for name, value in current.items()):
            raise ValueError("Scoring source changed from frozen structural bundle")
    thresholds = json.loads((directory / "local_thresholds.json").read_text(encoding="utf-8"))
    models, sides = {}, {}
    for target in thresholds:
        models[target], sides[target] = fusion.load_bundle(directory, target)
        with np.load(directory / "models" / f"{target}_enrollment.npz", allow_pickle=False) as archive:
            sides[target]["enrollment"] = [archive[f"matrix_{i}"].copy() for i in range(int(archive["count"]))]
    check_coordinates(models)
    return models, sides, thresholds


def replay(directory, path, target):
    models, sides, thresholds = load_bundle(directory)
    values, baseline, _ = score(path, target, models, sides, thresholds)
    policies = json.loads((directory / "policies.json").read_text(encoding="utf-8"))
    return {"scores": values, "baseline": baseline["decision"],
        "decisions": {name: target if values is not None and apply([values], policy)[0] else "REJECT"
            for name, policy in policies.items()}}


def write_table(output, name, rows):
    if rows:
        srdp.write_rows(output / name, rows, list(rows[0]))


def score_audit(rows):
    from scipy.stats import spearmanr
    streams = [*SINGLES, *(["pitch_identity", "pitch_relative"] if "pitch_identity" in rows[0] else [])]
    overlap, distributions, correlations = [], [], []
    for row in rows:
        if not row["genuine"]:
            continue
        # Dominance is meaningful only within the SAME claimed account.
        negatives = [n for n in rows if not n["genuine"] and n["target"] == row["target"]
            and n["voice_mfcc"] <= 1. and n["phrase_mfcc"] <= 1.1
            and all(n[k] <= row[k] for k in fusion.STREAM_NAMES)]
        overlap.append({"speaker": row["speaker"], "sequence": row["sequence"], "target": row["target"],
            "scores": {k: row[k] for k in streams}, "dominating_negatives": [{
                "speaker": n["speaker"], "sequence": n["sequence"], "target": n["target"], "kind": n["kind"],
                "scores": {k: n[k] for k in streams},
                "new_streams_breaking_dominance": [k for k in streams[4:] if n[k] > row[k]]} for n in negatives]})
    for target in sorted({r["target"] for r in rows}):
        subset = [r for r in rows if r["target"] == target]
        for kind in sorted({r["kind"] for r in subset}):
            part = [r for r in subset if r["kind"] == kind]
            for name in streams:
                values = np.array([r[name] for r in part])
                distributions.append({"target": target, "kind": kind, "stream": name, "count": len(values),
                    "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
                    "at_most_one": int(np.sum(values <= 1.))})
        for i, a in enumerate(streams):
            for b in streams[i + 1:]:
                if len(subset) < 3:
                    continue
                correlations.append({"target": target, "stream_a": a, "stream_b": b,
                    "spearman": float(spearmanr([r[a] for r in subset], [r[b] for r in subset]).statistic)})
    return overlap, distributions, correlations


def run(manifest_path, reference, output, *, with_pitch=False, with_relative=False):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protected = {str(manifest_path): fusion.digest(manifest_path)}
    for base in (srdp.PROJECT_ROOT / "app_data", srdp.PROJECT_ROOT / "tmp/gui_recordings", reference):
        protected.update({str(p): fusion.digest(p) for p in base.rglob("*") if p.is_file()})
    variants = (*VARIANTS, *PITCH_VARIANTS) if with_pitch else VARIANTS
    if with_relative:
        if not with_pitch:
            raise ValueError("Relative pitch requires --with-pitch")
        variants = (*variants, *RELATIVE_VARIANTS, *DELTA_CONTROLS)
    fusion.write_json(output / "protocol.json", {"design": "Exploratory development on previously inspected calibration; not final acceptance.",
        "variants": variants, "local_bins": 4, "local_score": "Worst of four equal template-time bins; mean cost per template frame first, then original multi-template aggregation.",
        "identity": "Target/rival raw VQ distance in identical coordinates. Fixed cohort gate <=1; original absolute voice gate also remains <=1.",
        "selection": "Only first-round known calibration labels; minimize FA, then FR; fixed equal local weights, fitted thresholds <=1. Group both claims of a WAV together.",
        "bounds": "No changed trimming, VQ, original thresholds, z-score, templates or App. Local thresholds: original enrollment-only LOO rule.",
        "limitations": "Two known speakers only; local bins are not syllables; no ASR/content identity separation claim. Family selection uses this development set; LOO is not independent validation.",
        "deployment": "None; holdout remains uncollected; no actual user-model replacement."})
    if with_pitch:
        fusion.write_json(output / "pitch_protocol.json", {"evidence": "Enrollment-only pitch probe after negative cohort/local development; no pitch-query results inspected before defining this family.",
            "extraction": "Raw centered cropped region, YIN 65-400Hz, 1024-sample frames, existing hop; original energy activity AND lag-normalized correlation >=0.6; at least 8 frames.",
            "descriptor": "Median log2(F0); absolute deviation from mean of four enrollment descriptors.",
            "threshold": "Enrollment leave-one-recording-out deviations: original mean+std calibration rule, floor 0.1 octave; no calibration-query fitting for pitch gate.",
            "missing_pitch": "Pitch-dependent decisions fail closed; other controls still score normally. Calibration stops rather than silently excluding queries.",
            "scope": "Pitch helps test these two different habitual ranges; not a unique identity, no similar-pitch or unknown-speaker guarantee."})
    if with_relative:
        fusion.write_json(output / "relative_pitch_protocol.json", {"design": "Additional exploratory family after absolute-pitch gate rejected a genuine high-F0 query; not predeclared before the absolute-pitch experiment.",
            "rule": "Absolute log2-F0 deviation from target enrollment mean / nearest other enrollment mean deviation <=1. No fitted query threshold or changed F0 extractor.",
            "controls": "Identical original voice and phrase guards, original/absolute-pitch gate, best single including both pitch scores; fixed equal local fusion and fitted local threshold.",
            "limitation": "Enrollment-dependent closed-cohort comparison, not unknown-speaker authentication. Original absolute voice gate remains necessary."})
    fusion.write_json(output / "labels.json", manifest)
    snapshots = output / "models"
    snapshots.mkdir()
    models, sides, thresholds, enrollment_audit = {}, {}, {}, []
    enrollment_hashes = set()
    for target, registration in manifest["models"].items():
        model, side = fusion.load_bundle(reference, target)
        actual = gui.load_user_model(srdp.project_path(registration["model_path"]))
        if gui.session_model_fingerprint(model) != registration["fingerprint"] or gui.session_model_fingerprint(actual) != registration["fingerprint"]:
            raise ValueError("Frozen model fingerprint changed")
        enrollment = []
        for item in registration["enrollment"]:
            path = srdp.project_path(item["path"])
            if fusion.digest(path) != item["sha256"]:
                raise ValueError("Enrollment hash changed")
            enrollment_hashes.add(item["sha256"])
            enrollment.append(srdp.extract_one_audio_features(path, model).features)
        local_thresholds, audit = build_local_thresholds(enrollment, model)
        if with_pitch:
            descriptors = [pitch_descriptor(srdp.project_path(item["path"]), model) for item in registration["enrollment"]]
            centers = np.array([d["log2_median_hz"] for d in descriptors])
            deviations = [abs(value - np.delete(centers, i).mean()) for i, value in enumerate(centers)]
            scale = srdp.calibration_threshold(deviations, 1.) if model.calibration_mode == "leave_one_recording_out" else max(np.mean(deviations) + 2 * np.std(deviations), max(deviations) + 1e-6)
            local_thresholds["pitch"] = {"center": float(centers.mean()), "threshold": float(max(.1, scale)), "enrollment": descriptors}
        thresholds[target], models[target] = local_thresholds, model
        sides[target] = {**side, "enrollment": enrollment}
        enrollment_audit += [{"target": target, **r} for r in audit]
        for name in (f"{target}.npz", f"{target}_side.npz"):
            shutil.copyfile(reference / "models" / name, snapshots / name)
        np.savez_compressed(snapshots / f"{target}_enrollment.npz", count=len(enrollment),
            **{f"matrix_{i}": matrix for i, matrix in enumerate(enrollment)})
    check_coordinates(models)
    reference_scores = {(r["audio_sha256"], r["target"]): r for r in srdp.read_csv_rows(reference / "scores.csv")}
    rows, pairs, seen = [], [], set()
    for sample in manifest["recordings"]:
        if sample["phrase"] == "unknown":
            continue
        path = srdp.project_path(sample["file_path"])
        key = sample["sha256"]
        if fusion.digest(path) != key or key in seen or key in enrollment_hashes:
            raise ValueError("Query hash changed, duplicated, or overlaps enrollment")
        seen.add(key)
        for target in models:
            values, baseline, details = score(path, target, models, sides, thresholds)
            if values is None:
                raise ValueError("Calibration query failed speech check")
            if with_pitch and not np.isfinite(values["pitch_identity"]):
                raise ValueError("Calibration query has insufficient pitch; do not silently exclude it")
            original = reference_scores[key, target]
            if baseline["decision"] != original["baseline"] or any(abs(values[k] - float(original[k])) > 2e-6 for k in fusion.STREAM_NAMES):
                raise ValueError("Original four streams did not reproduce")
            metadata = {k: original[k] for k in ("file_path", "audio_sha256", "speaker", "sequence", "phrase", "target", "kind", "baseline", "model_fingerprint")}
            row = {**metadata, "genuine": original["genuine"] == "True", "input_rejected": False, **values}
            rows.append(row)
            pairs += [{**metadata, **r} for r in details]
    policies, predictions, changes, folds, summaries, metrics = {}, [], [], [], [], []
    for variant in variants:
        policy = fit(rows, variant)
        policies[variant] = policy
        cv, fits = grouped_predictions(rows, variant)
        folds += [{"variant": variant, **p} for p in fits]
        for stage, decisions in (("calibration_fit", apply(rows, policy)), ("grouped_loo", cv)):
            for kind in sorted({r["kind"] for r in rows}):
                indices = [i for i, r in enumerate(rows) if r["kind"] == kind]
                summaries.append({"stage": stage, "variant": variant, "kind": kind, "count": len(indices), "accepted": int(sum(decisions[indices]))})
                for field in ("speaker", "target"):
                    for person in models:
                        subset = [i for i in indices if rows[i][field] == person]
                        metrics.append({"stage": stage, "variant": variant, "group_by": field, "person": person,
                            "kind": kind, "count": len(subset), "accepted": int(sum(decisions[subset]))})
            for row, accepted in zip(rows, decisions):
                after = {**row, "stage": stage, "variant": variant, "decision": row["target"] if accepted else "REJECT"}
                predictions.append(after)
                if after["decision"] != row["baseline"]:
                    changes.append({**after, "change": "new_accept" if accepted else "new_reject"})
    for name, records in (("scores.csv", rows), ("local_alignment_audit.csv", pairs), ("enrollment_calibration.csv", enrollment_audit),
            ("summary.csv", summaries), ("speaker_metrics.csv", metrics), ("predictions.csv", predictions), ("changes.csv", changes)):
        write_table(output, name, records)
    overlap, distributions, correlations = score_audit(rows)
    fusion.write_json(output / "score_overlap_audit.json", overlap)
    write_table(output, "score_distributions.csv", distributions)
    write_table(output, "score_correlations.csv", correlations)
    fusion.write_json(output / "local_thresholds.json", thresholds)
    fusion.write_json(output / "policies.json", policies)
    fusion.write_json(output / "folds.json", folds)
    fusion.write_json(output / "input_hashes.json", protected)
    bundle = [output / name for name in ("policies.json", "local_thresholds.json", "protocol.json")]
    if with_pitch:
        bundle.append(output / "pitch_protocol.json")
    if with_relative:
        bundle.append(output / "relative_pitch_protocol.json")
    source_dir = output / "source"
    source_dir.mkdir()
    source_files = [Path(__file__).parent / name for name in ("live_structure_experiments.py", "live_fusion_experiments.py",
        "gui.py", "srdp.py", "speech_frames.py", "dtw_paths.py")]
    for path in source_files:
        shutil.copyfile(path, source_dir / path.name)
    fusion.write_json(output / "source_hashes.json", {p.name: fusion.digest(p) for p in source_files})
    fusion.write_json(output / "environment.json", {"python": sys.version,
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "librosa", "numba", "soundfile", "PySide6")}})
    bundle += [output / "source_hashes.json", output / "environment.json", *source_dir.glob("*.py")]
    fusion.write_json(output / "bundle_hashes.json", {p.relative_to(output).as_posix(): fusion.digest(p) for p in [*bundle, *snapshots.glob("*.npz")]})
    if any(fusion.digest(Path(name)) != value for name, value in protected.items()):
        raise ValueError("Protected inputs changed")
    print(json.dumps({"wavs": len(seen), "claims": len(rows), "variants": len(variants), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--target", choices=("lwl2", "lyh2"))
    parser.add_argument("--with-pitch", action="store_true")
    parser.add_argument("--with-relative", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(srdp.project_path(args.replay), srdp.project_path(args.wav), args.target), ensure_ascii=False))
    else:
        run(srdp.project_path(args.manifest), srdp.project_path(args.reference), srdp.project_path(args.output_dir),
            with_pitch=args.with_pitch, with_relative=args.with_relative)
