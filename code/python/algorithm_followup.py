"""Reproducible development comparisons; never selects or replaces the default model."""
from __future__ import annotations

import json
import hashlib
import platform
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

import srdp
from dtw_paths import alignment, path_statistics


def save_table(path, rows):
    if rows:
        srdp.write_rows(path, rows, list(rows[0]))


def choose_templates(matrices, mode, max_run=0):
    if mode == "all":
        return matrices
    if mode == "first":
        return matrices[:1]
    distances = np.zeros((len(matrices), len(matrices)))
    for i in range(len(matrices)):
        for j in range(i):
            distances[i, j] = distances[j, i] = alignment(matrices[i], matrices[j], max_run=max_run)[0]
    return [matrices[int(np.argmin(distances.sum(axis=1)))]]


def audit_paths(output):
    names, rows, features = srdp.load_feature_archive(srdp.FEATURE_DIR / "srdp_features.npz")
    model = srdp.load_model("results/calibration_experiments/loo_voice/model.npz")
    train = {speaker: [] for speaker in model.codebooks}
    for row, matrix in zip(rows, features):
        if row["split"] == "train":
            train[row["speaker_id"]].append((row, matrix))
    pairs, nearest, plot_cases = [], [], []
    for row, matrix in zip(rows, features):
        if row["split"] == "train":
            speaker = row["speaker_id"]
        else:
            scores = {speaker: srdp.quantization_distance(matrix[:, :12], book) for speaker, book in model.codebooks.items()}
            speaker = min(scores, key=scores.get)
        choices = []
        for reference, template in train[speaker]:
            if reference["file_path"] == row["file_path"]:
                continue
            distance, path = alignment(matrix, template)
            record = {
                "file_path": row["file_path"], "split": row["split"], "speaker": speaker,
                "template_path": reference["file_path"], "distance": distance,
                "phrase_score": distance / model.passphrase_thresholds[speaker],
                **path_statistics(path, len(matrix), len(template)),
            }
            pairs.append(record)
            choices.append((distance, record, template, path))
        distance, record, template, path = min(choices, key=lambda item: item[0])
        nearest.append(record)
        if "S02_wrong/corridor/S02_wrong_003" in row["file_path"] or "S02/corridor/S02_corridor_004" in row["file_path"]:
            plot_cases.append((row, matrix, template, path, record))
    save_table(output / "alignment_pairs.csv", pairs)
    save_table(output / "nearest_paths.csv", nearest)
    summary = []
    for split in sorted({row["split"] for row in nearest}):
        group = [row for row in nearest if row["split"] == split]
        for key in ("max_axis_run", "length_ratio", "diagonal_fraction", "phrase_score"):
            values = [row[key] for row in group]
            summary.append({"split": split, "metric": key, "count": len(group), "min": min(values), "median": float(np.median(values)), "max": max(values)})
    save_table(output / "path_summary.csv", summary)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(plot_cases), figsize=(11, 4.5), squeeze=False)
    for ax, (row, matrix, template, path, record) in zip(axes[0], plot_cases):
        local = np.stack([np.linalg.norm(template - frame, axis=1) for frame in matrix])
        ax.imshow(local, origin="lower", aspect="auto", cmap="viridis")
        ax.plot(path[:, 1], path[:, 0], color="white", linewidth=1)
        ax.set(xlabel="Reference frame", ylabel="Query frame", title=f"{Path(row['file_path']).stem}\nscore={record['phrase_score']:.3f}, max run={record['max_axis_run']}")
    fig.tight_layout()
    fig.savefig(output / "alignment_paths.png", dpi=160)
    plt.close(fig)
    print("DTW path audit saved for all 99 recordings", flush=True)


def fit_variant(rows, raw, *, voice_set="mfcc", phrase_set="all", max_run=0, template_mode="all", vad_mode="trim", normalization="none"):
    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    if any(raw[i] is None for i in train_indices):
        raise ValueError("VAD rejected a training recording; do not silently drop training data")
    speakers = sorted({rows[i]["speaker_id"] for i in train_indices})
    dims = srdp.select_dims(srdp.FEATURE_NAMES, voice_set)
    phrase_dims = srdp.select_dims(srdp.FEATURE_NAMES, phrase_set)
    mean, std = srdp.zscore_fit([raw[i] for i in train_indices])
    features = [srdp.zscore_apply(x, mean, std) if x is not None else None for x in raw]
    grouped = {speaker: [features[i] for i in train_indices if rows[i]["speaker_id"] == speaker] for speaker in speakers}
    trials = []
    for held in train_indices:
        fold_mean, fold_std = srdp.zscore_fit([raw[i] for i in train_indices if i != held])
        sample = srdp.zscore_apply(raw[held], fold_mean, fold_std)
        for speaker in speakers:
            others = [srdp.zscore_apply(raw[i], fold_mean, fold_std) for i in train_indices if i != held and rows[i]["speaker_id"] == speaker]
            book = srdp.lbg_codebook(np.vstack([x[:, dims] for x in others]), 16)
            templates = choose_templates([x[:, phrase_dims] for x in others], template_mode, max_run)
            distance = srdp.passphrase_template_distance(sample[:, phrase_dims], templates, "min", max_run=max_run)
            trials.append({"file_path": rows[held]["file_path"], "claim_speaker": speaker, "true_speaker": rows[held]["speaker_id"], "genuine": rows[held]["speaker_id"] == speaker, "voice_distance": srdp.quantization_distance(sample[:, dims], book), "passphrase_distance": distance})
    voice_thresholds, phrase_thresholds = {}, {}
    for speaker in speakers:
        own = [row for row in trials if row["genuine"] and row["claim_speaker"] == speaker]
        voice_thresholds[speaker] = srdp.calibration_threshold([row["voice_distance"] for row in own], 2.0)
        phrase_thresholds[speaker] = srdp.calibration_threshold([row["passphrase_distance"] for row in own], 1.0)
    for row in trials:
        row["voice_score"] = row["voice_distance"] / voice_thresholds[row["claim_speaker"]]
        row["passphrase_score"] = row["passphrase_distance"] / phrase_thresholds[row["claim_speaker"]]
    model = replace(
        srdp.load_model(), zscore_mean=mean, zscore_std=std,
        codebooks={speaker: srdp.lbg_codebook(np.vstack([x[:, dims] for x in grouped[speaker]]), 16) for speaker in speakers},
        thresholds=voice_thresholds, passphrase_thresholds=phrase_thresholds,
        passphrase_train_by_speaker={speaker: choose_templates([x[:, phrase_dims] for x in grouped[speaker]], template_mode, max_run) for speaker in speakers},
        feature_set=voice_set, passphrase_feature_set=phrase_set, passphrase_aggregate="min",
        dtw_window_ratio=1.0, dtw_warp_penalty=0.0, dtw_max_run=max_run,
        decision_mode="two_layer", fusion_threshold=1.0, fusion_voice_weight=0.5,
        voice_metric="absolute", quality_weight=0.0, calibration_mode="leave_one_recording_out",
        vad_mode=vad_mode, cepstral_normalization=normalization,
    )
    return model, features, trials


def evaluate(output, name, rows, features, model, trials):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    results, durations = [], []
    for row, matrix in zip(rows, features):
        if row["split"] == "train":
            continue
        started = time.perf_counter()
        if matrix is None:
            result = srdp.no_speech_result(row, "未检测到足够的有效语音")
        else:
            result = srdp.predict_feature_matrix(matrix, model, row, score_all_branches=True)
        durations.append(1000 * (time.perf_counter() - started))
        results.append(result)
    srdp.save_vq_results(directory / "recognition_results.csv", results)
    srdp.save_calibrated_model(directory / "model.npz", model)
    save_table(directory / "calibration_trials.csv", trials)
    summary = {
        "variant": name, **srdp.summarize_decisions(results, [row["decision"] for row in results]),
        "mean_scoring_ms": float(np.mean(durations)), "p95_scoring_ms": float(np.percentile(durations, 95)),
        "voice_feature_set": model.feature_set, "phrase_feature_set": model.passphrase_feature_set,
        "vad_mode": model.vad_mode, "cepstral_normalization": model.cepstral_normalization,
        "dtw_max_run": model.dtw_max_run, "model_path": srdp.display_path(directory / "model.npz"),
    }
    breakdown = []
    for field in ("true_speaker", "environment"):
        for value in sorted({row[field] for row in results}):
            group = [row for row in results if row[field] == value]
            breakdown.append({"group_by": field, "group": value, "count": len(group), **srdp.summarize_decisions(group, [row["decision"] for row in group])})
    save_table(directory / "group_metrics.csv", breakdown)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, results


def fit_bounded_fusion(trials, voice_guard, phrase_guard):
    genuine = np.array([row["genuine"] for row in trials])
    voice = np.array([row["voice_score"] for row in trials])
    phrase = np.array([row["passphrase_score"] for row in trials])
    gates = (voice <= voice_guard) & (phrase <= phrase_guard)
    best = None
    for weight in (.3, .5, .7):
        scores = weight * voice + (1 - weight) * phrase
        boundaries = sorted({0.0, 1.0, *scores[gates].tolist()})
        for threshold in [0., 1.] + [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]:
            if threshold > 1.0:
                continue
            accepted = gates & (scores <= threshold)
            fa, fr = int(np.sum(accepted & ~genuine)), int(np.sum(~accepted & genuine))
            rank = fa, fr, abs(weight - .5), -threshold
            if best is None or rank < best[0]:
                best = rank, {"weight": weight, "threshold": threshold, "calibration_false_accept": fa, "calibration_false_reject": fr, "voice_guard": voice_guard, "passphrase_guard": phrase_guard}
    return best[1]


def fusion_comparison(output, rows, features, model, trials):
    summaries, changes, policies = [], [], []
    summary, baseline = evaluate(output, "fusion_two_layer", rows, features, model, trials)
    summaries.append(summary)
    for name, vg, pg in (("strict", 1., 1.), ("voice_complement", 1.1, 1.), ("both_complement", 1.1, 1.1)):
        policy = fit_bounded_fusion(trials, vg, pg)
        policies.append(dict(variant=name, **policy))
        candidate = replace(model, decision_mode="bounded_fusion", voice_guard=vg, passphrase_guard=pg, fusion_voice_weight=policy["weight"], fusion_threshold=policy["threshold"])
        summary, results = evaluate(output, f"fusion_{name}", rows, features, candidate, trials)
        summaries.append(summary)
        for before, after in zip(baseline, results):
            if before["decision"] != after["decision"]:
                changes.append({"variant": name, "file_path": after["file_path"], "expected_decision": after["expected_decision"], "before": before["decision"], "after": after["decision"], "voice_score": after["voice_score"], "passphrase_score": after["passphrase_score"], "fusion_score": after["fusion_score"], "threshold": policy["threshold"], "effect": "corrected" if after["decision"] == after["expected_decision"] else "regressed"})
    save_table(output / "fusion_comparison.csv", summaries)
    save_table(output / "fusion_changes.csv", changes)
    save_table(output / "fusion_policies.csv", policies)


def stress_audio(output, models, rows):
    import librosa
    trials = []
    rng = np.random.default_rng(20261029)
    with tempfile.TemporaryDirectory(prefix="srdp_audio_stress_") as directory:
        path = Path(directory) / "input.wav"
        cases = []
        for row in rows:
            if row["split"] != "test":
                continue
            audio, sr = srdp.load_audio(srdp.project_path(row["file_path"]))
            for gain in (1., .1, .01):
                cases.append((f"speech_gain_{gain:g}", row["file_path"], audio * gain, row["expected_decision"]))
        length = 3 * srdp.TARGET_SR
        cases.append(("silence", "synthetic", np.zeros(length), "REJECT"))
        cases.append(("dc", "synthetic", np.full(length, .05), "REJECT"))
        for seed in range(5):
            white = rng.normal(0, .03, length)
            colored = librosa.effects.preemphasis(white, coef=-.9)
            cases.extend((("white_noise", f"synthetic_{seed}", white, "REJECT"), ("colored_noise", f"synthetic_{seed}", colored, "REJECT")))
        cases.append(("tone_440Hz", "synthetic", .03 * np.sin(2 * np.pi * 440 * np.arange(length) / srdp.TARGET_SR), "REJECT"))
        for kind, source, audio, expected in cases:
            sf.write(path, audio, srdp.TARGET_SR, subtype="FLOAT")
            for name, model in models.items():
                started = time.perf_counter()
                result = srdp.predict_one(path, model, metadata={"expected_decision": expected})
                trials.append({"variant": name, "kind": kind, "source": source, "expected_decision": expected, "decision": result["decision"], "correct": result["correct"], "frontend_rejected": not bool(result["voice_score"]), "reject_reason": result["reject_reason"], "latency_ms": 1000 * (time.perf_counter() - started)})
    save_table(output / "audio_stress_trials.csv", trials)
    summary = []
    for name in models:
        for kind in sorted({row["kind"] for row in trials}):
            group = [row for row in trials if row["variant"] == name and row["kind"] == kind]
            summary.append({"variant": name, "kind": kind, "count": len(group), "correct": sum(row["correct"] == "True" for row in group), "accepted": sum(row["decision"] != "REJECT" for row in group), "false_reject": sum(row["expected_decision"] != "REJECT" and row["decision"] == "REJECT" for row in group), "false_accept": sum(row["expected_decision"] == "REJECT" and row["decision"] != "REJECT" for row in group), "wrong_speaker": sum(row["expected_decision"] != "REJECT" and row["decision"] not in ("REJECT", row["expected_decision"]) for row in group), "frontend_rejected": sum(row["frontend_rejected"] for row in group), "mean_end_to_end_ms": float(np.mean([row["latency_ms"] for row in group]))})
    save_table(output / "audio_stress_summary.csv", summary)


def run_experiments(args):
    output = srdp.project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "librosa": srdp.librosa.__version__, "numba": __import__("numba").__version__},
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("srdp.py", "dtw_paths.py", "algorithm_followup.py")},
        "evaluation_status": "development comparisons on historically reused 63 recordings, not independent validation",
        "calibration": "36 training recordings only; leave one complete recording out; refit global z-score every fold; voice mean+2std; phrase mean+1std",
        "fusion_negatives": "108 other-registered-user claims, all saying correct phrase; no genuine-user wrong-phrase calibration recordings",
        "fusion_guards": "predeclared (1,1), (1.1,1), (1.1,1.1); fit weights/thresholds only on training calibration scores; no default deployment",
        "preprocessing": "frame energy + spectral flatness VAD, not WebRTC; 25ms frames/10ms hop, flatness<0.35, RMS>=max(1e-5,0.1*peak), >=0.20s voiced, 60ms outer padding, internal pauses preserved",
        "normalization": "utterance CMN/CMVN on 12 static MFCCs, recompute deltas, retain other 7 features, then training-only z-score",
        "timing": "warm DTW JIT before measurement; scoring excludes extraction; audio_stress measures WAV loading + extraction + scoring",
        "limitations": "synthetic noise/gain are stress tests, not new people/devices; calibration fitting statistics are not held-out performance; no real microphone acceptance claim",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    alignment(np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32))
    if args.stage in {"all", "audit"}:
        audit_paths(output)
    if args.stage == "audit":
        return
    rows = srdp.read_metadata(srdp.DEMO_AUDIO_DIR / "metadata_srdp.csv")
    raw = [srdp.extract_one(srdp.project_path(row["file_path"])).features for row in rows]
    model, features, trials = fit_variant(rows, raw)
    summary, _ = evaluate(output, "baseline", rows, features, model, trials)
    comparisons = [summary]
    if args.stage in {"all", "dtw"}:
        for name, settings in (
            ("dtw_run2", {"max_run": 2}), ("dtw_run3", {"max_run": 3}), ("dtw_run5", {"max_run": 5}),
            ("dtw_medoid", {"template_mode": "medoid"}), ("dtw_single_first", {"template_mode": "first"}),
            ("dtw_mfcc_delta", {"phrase_set": "mfcc_delta"}),
        ):
            candidate, matrices, calibration = fit_variant(rows, raw, **settings)
            result, _ = evaluate(output, name, rows, matrices, candidate, calibration)
            comparisons.append(result)
            save_table(output / "comparison.csv", comparisons)
    if args.stage in {"all", "preprocessing"}:
        stress_models = {"trim": model}
        genuine = [row["voice_score"] for row in trials if row["genuine"]]
        impostor = [row["voice_score"] for row in trials if not row["genuine"]]
        separation = [{"variant": "baseline", "genuine_median": float(np.median(genuine)), "impostor_median": float(np.median(impostor)), "impostor_below_voice_gate": sum(value <= 1 for value in impostor), "genuine_above_voice_gate": sum(value > 1 for value in genuine)}]
        for name, vad, normalization, voice in (
            ("frame_vad", "frame", "none", "mfcc"), ("cmn", "trim", "cmn", "mfcc"),
            ("cmvn", "trim", "cmvn", "mfcc"), ("voice_mfcc_delta", "trim", "none", "mfcc_delta"),
            ("voice_all", "trim", "none", "all"),
        ):
            matrices = []
            for row in rows:
                try:
                    matrices.append(srdp.extract_one(srdp.project_path(row["file_path"]), vad_mode=vad, cepstral_normalization=normalization).features)
                except srdp.NoSpeechError:
                    matrices.append(None)
            candidate, normalized, calibration = fit_variant(rows, matrices, vad_mode=vad, normalization=normalization, voice_set=voice)
            result, _ = evaluate(output, name, rows, normalized, candidate, calibration)
            comparisons.append(result)
            genuine = [row["voice_score"] for row in calibration if row["genuine"]]
            impostor = [row["voice_score"] for row in calibration if not row["genuine"]]
            separation.append({"variant": name, "genuine_median": float(np.median(genuine)), "impostor_median": float(np.median(impostor)), "impostor_below_voice_gate": sum(value <= 1 for value in impostor), "genuine_above_voice_gate": sum(value > 1 for value in genuine)})
            if name == "frame_vad":
                stress_models["frame"] = candidate
            save_table(output / "comparison.csv", comparisons)
        save_table(output / "speaker_separation.csv", separation)
        stress_audio(output, stress_models, rows)
    if args.stage in {"all", "fusion"}:
        fusion_comparison(output, rows, features, model, trials)
    save_table(output / "comparison.csv", comparisons)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/followup_experiments")
    parser.add_argument("--stage", default="all", choices=["all", "audit", "dtw", "preprocessing", "fusion"])
    run_experiments(parser.parse_args())
