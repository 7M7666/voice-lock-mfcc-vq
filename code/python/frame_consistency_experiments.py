"""Enrollment/query preprocessing ablations on existing recordings only."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

import gui
import srdp
from dtw_paths import alignment
from speech_frames import extract_branches, energy_activity, mask_for_region


POLICIES = {"reenroll_legacy": (False, False, False), "voice_active": (True, False, False), "phrase_span": (False, True, False),
    "both": (True, True, False), "phrase_sustained": (False, True, True), "both_sustained": (True, True, True)}


class FeatureCache:
    def __init__(self):
        self.values = {}

    def get(self, path, model, policy):
        key = (str(path), model.zscore_mean.tobytes(), model.zscore_std.tobytes(), model.vad_mode, model.cepstral_normalization, policy)
        if key not in self.values:
            try:
                voice_active, phrase_span, sustained = POLICIES[policy]
                self.values[key] = extract_branches(path, model, voice_active=voice_active, phrase_span=phrase_span, sustained=sustained)
            except srdp.NoSpeechError as error:
                self.values[key] = error
        result = self.values[key]
        if isinstance(result, Exception):
            raise result
        return result


def score_branches(features, model):
    voice = features.voice[:, srdp.select_dims(model.feature_names, model.feature_set)]
    phrase = features.phrase[:, srdp.select_dims(model.feature_names, model.passphrase_feature_set)]
    distance = srdp.quantization_distance(voice, model.codebook)
    if model.voice_metric == "cohort_ratio":
        distance = srdp.speaker_distance_scores({"": distance, **{speaker: srdp.quantization_distance(voice, book) for speaker, book in model.background_codebooks.items()}}, "cohort_ratio")[""]
    phrase_distance = srdp.passphrase_template_distance(phrase, model.passphrase_templates, model.passphrase_aggregate, window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
    voice_score = srdp.safe_ratio(distance, model.voice_threshold)
    phrase_score = srdp.safe_ratio(phrase_distance, model.passphrase_threshold)
    accepted, fusion = srdp.verification_decision(voice_score, phrase_score / model.passphrase_accept_score, model.decision_mode, model.fusion_voice_weight, model.fusion_threshold, voice_guard=model.voice_guard, passphrase_guard=model.passphrase_guard)
    return {"decision": model.speaker_id if accepted else "REJECT", "voice_score": f"{voice_score:.6f}", "passphrase_score": f"{phrase_score:.6f}", "fusion_score": f"{fusion:.6f}", "reason": "" if accepted else "score_rejection"}


def find_registration_paths(model, cache):
    dims = srdp.select_dims(model.feature_names, model.passphrase_feature_set)
    matches = [[] for _ in model.passphrase_templates]
    for path in sorted(gui.TEMP_AUDIO_DIR.glob("enroll*.wav")):
        try:
            matrix = cache.get(path, model, "reenroll_legacy").phrase[:, dims]
        except (srdp.NoSpeechError, ValueError):
            continue
        for index, template in enumerate(model.passphrase_templates):
            if matrix.shape == template.shape and np.allclose(matrix, template, atol=1e-6, rtol=1e-6):
                matches[index].append(path)
    paths = []
    for index, candidates in enumerate(matches):
        if not candidates:
            raise ValueError(f"Cannot recover enrollment WAV for {model.speaker_id} template {index}")
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}
        if len(hashes) != 1:
            raise ValueError(f"Ambiguous source WAVs for {model.speaker_id} template {index}")
        paths.append(candidates[0])
    return paths


def training_base(profile):
    return replace(srdp.load_model(), feature_names=profile.feature_names, zscore_mean=profile.zscore_mean, zscore_std=profile.zscore_std,
        quality_baseline=profile.quality_baseline, feature_set=profile.feature_set, passphrase_feature_set=profile.passphrase_feature_set,
        passphrase_aggregate=profile.passphrase_aggregate, dtw_window_ratio=profile.dtw_window_ratio, dtw_warp_penalty=profile.dtw_warp_penalty,
        dtw_max_run=profile.dtw_max_run, voice_metric=profile.voice_metric, vad_mode=profile.vad_mode, cepstral_normalization=profile.cepstral_normalization,
        decision_mode=profile.decision_mode, fusion_voice_weight=profile.fusion_voice_weight, fusion_threshold=profile.fusion_threshold,
        voice_guard=profile.voice_guard, passphrase_guard=profile.passphrase_guard,
        calibration_mode="leave_one_recording_out" if profile.calibration_mode == "leave_one_recording_out" else "legacy")


def train_profile(speaker, paths, base, policy, size, cache):
    matrices = [cache.get(path, base, policy) for path in paths]
    model = gui.build_session_enrollment_model(speaker, base, [item.voice for item in matrices], passphrase_features=[item.phrase for item in matrices], codebook_size=size)
    return model, matrices


def replay_candidate(directory, path):
    configuration = json.loads((directory / "processing_policy.json").read_text(encoding="utf-8"))
    model = gui.load_user_model(directory / "parameters.npz")
    if gui.session_model_fingerprint(model) != configuration["model_fingerprint"]:
        raise ValueError("Candidate parameters do not match their processing policy")
    policy = configuration["policy"]
    try:
        features = FeatureCache().get(path, model, "reenroll_legacy" if policy == "frozen_profile" else policy)
        return score_branches(features, model)
    except srdp.NoSpeechError as error:
        return {"decision": "REJECT", "voice_score": "", "passphrase_score": "", "fusion_score": "", "reason": str(error)}


def attribution(model, source_paths, query, cache, query_kind):
    features = cache.get(query, model, "reenroll_legacy")
    vdims = srdp.select_dims(model.feature_names, model.feature_set)
    pdims = srdp.select_dims(model.feature_names, model.passphrase_feature_set)
    rows = []
    # Attribute both the current crop and a deliberately loose query-only boundary.
    raw, sr = srdp.load_audio(query)
    pre = srdp.pre_emphasize(raw)
    rms = srdp.librosa.feature.rms(y=pre, frame_length=512, hop_length=160)[0]
    indices = np.flatnonzero(rms >= .01 * float(rms.max()))
    first = max(0, int(indices[0] * 160) - 800)
    last = min(len(raw), int(indices[-1] * 160 + 512) + 800)
    loose = srdp.extract_audio_region(raw, sr, first, last, cepstral_normalization=model.cepstral_normalization)
    _, context, _, _ = energy_activity(raw, sr)
    loose_matrix = srdp.zscore_apply(loose.features, model.zscore_mean, model.zscore_std)
    for crop, matrix, activity in (("legacy", features.legacy.features, features.activity_on_legacy), ("loose_query_only", loose_matrix, mask_for_region(context, first / sr, len(loose_matrix), sr))):
        local_voice = np.linalg.norm(matrix[:, vdims, None] - model.codebook.T[None, :, :], axis=1).min(axis=1)
        for label, mask in (("activity", activity), ("background_candidate", ~activity)):
            rows.append({"target": model.speaker_id, "query": srdp.display_path(query), "query_kind": query_kind, "crop": crop, "branch": "voice", "category": label, "count": int(mask.sum()), "total": len(matrix), "mean_local_cost": float(np.mean(local_voice[mask])) if mask.any() else "", "contribution_to_total_distance": float(local_voice[mask].sum() / len(matrix))})
        choices = [(alignment(matrix[:, pdims], template, window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run), i) for i, template in enumerate(model.passphrase_templates)]
        ((_, path), index) = min(choices, key=lambda item: item[0][0])
        reference = cache.get(source_paths[index], model, "reenroll_legacy")
        costs = np.linalg.norm(matrix[path[:, 0]][:, pdims] - model.passphrase_templates[index][path[:, 1]], axis=1)
        qa = activity[path[:, 0]]
        ra = reference.activity_on_legacy[path[:, 1]]
        for label, mask in (("activity_activity", qa & ra), ("activity_background", qa & ~ra), ("background_activity", ~qa & ra), ("background_background", ~qa & ~ra)):
            rows.append({"target": model.speaker_id, "query": srdp.display_path(query), "query_kind": query_kind, "crop": crop, "branch": "phrase", "category": label, "count": int(mask.sum()), "total": len(path), "mean_local_cost": float(costs[mask].mean()) if mask.any() else "", "contribution_to_total_distance": float(costs[mask].sum() / len(path))})
    return rows


def summarize(rows):
    result = []
    for scope in sorted({row["scope"] for row in rows}):
        for variant in sorted({row["variant"] for row in rows if row["scope"] == scope}):
            for kind in sorted({row["kind"] for row in rows if row["scope"] == scope}):
                group = [row for row in rows if row["scope"] == scope and row["variant"] == variant and row["kind"] == kind]
                result.append({"scope": scope, "variant": variant, "kind": kind, "count": len(group), "registration_failed": sum(row["reason"] == "registration_failed" for row in group), "accepted": sum(row["decision"] != "REJECT" for row in group), "correct": sum(row["decision"] == row["expected_decision"] for row in group), "false_reject": sum(row["expected_decision"] != "REJECT" and row["decision"] == "REJECT" for row in group), "false_accept": sum(row["expected_decision"] == "REJECT" and row["decision"] != "REJECT" for row in group)})
    return result


def decision_changes(rows):
    baseline = {(row["scope"], row["group"], row["file_path"]): row for row in rows if row["variant"] == "reenroll_legacy"}
    changes = []
    for row in rows:
        if row["variant"] in {"frozen_profile", "reenroll_legacy"}:
            continue
        before = baseline[row["scope"], row["group"], row["file_path"]]
        if before["decision"] == row["decision"]:
            continue
        changes.append({**row, "change": "new_accept" if row["decision"] != "REJECT" else "new_reject",
            "before_decision": before["decision"], "before_voice_score": before["voice_score"],
            "before_passphrase_score": before["passphrase_score"], "before_fusion_score": before["fusion_score"], "before_reason": before["reason"]})
    return changes


def run(args):
    output = srdp.project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache()
    manifest = json.loads(srdp.project_path(args.live_manifest).read_text(encoding="utf-8"))
    live = manifest["trials"]
    metadata = srdp.read_metadata(srdp.DEMO_AUDIO_DIR / "metadata_srdp.csv")
    evaluation = [row for row in metadata if row["split"] != "train"]
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    tails = []
    for index, trial in enumerate(live):
        raw, sr = srdp.load_audio(gui.TEMP_AUDIO_DIR / trial["file"])
        path = inputs / f"tail_{index}.wav"
        sf.write(path, np.tile(raw[-8000:], 6), sr, subtype="FLOAT")
        tails.append(path)
    groups = []
    attributions = []
    for user in sorted({trial["user"] for trial in live}):
        profile = gui.load_user_model(gui.user_model_path(user))
        paths = find_registration_paths(profile, cache)
        cases = [("genuine" if trial["user"] == user else "other_live_user", gui.TEMP_AUDIO_DIR / trial["file"], user if trial["user"] == user else "REJECT") for trial in live]
        cases += [("other_dataset_user", srdp.project_path(row["file_path"]), "REJECT") for row in evaluation]
        cases += [("repeated_tail", path, "REJECT") for path in tails]
        groups.append(("live", user, user, paths, training_base(profile), profile.codebook.shape[0], cases, profile))
        for trial in live:
            attributions.extend(attribution(profile, paths, gui.TEMP_AUDIO_DIR / trial["file"], cache, "genuine" if trial["user"] == user else "other_live_user"))
        print(f"Recovered and attributed enrollment for {user}", flush=True)
    srdp.write_rows(output / "frame_attribution.csv", attributions, list(attributions[0]))
    if not args.skip_formal:
        base = srdp.load_model()
        for speaker in sorted({row["speaker_id"] for row in metadata if row["split"] == "train"}):
            for environment in sorted({row["environment"] for row in metadata if row["split"] == "train"}):
                paths = [srdp.project_path(row["file_path"]) for row in metadata if row["split"] == "train" and row["speaker_id"] == speaker and row["environment"] == environment]
                if len(paths) != 3:
                    raise ValueError("Formal registration simulation requires the existing three training WAVs per speaker/environment")
                cases = [("genuine" if row["expected_decision"] == speaker else "wrong_phrase" if row["split"] == "wrong_passphrase" else "impostor" if row["split"] == "impostor" else "other_registered_user", srdp.project_path(row["file_path"]), speaker if row["expected_decision"] == speaker else "REJECT") for row in evaluation]
                groups.append(("formal_simulation", f"{speaker}_{environment}", speaker, paths, base, 16, cases, None))
    results, quality, policies, registrations = [], [], [], []
    for scope, group, speaker, paths, base, size, cases, frozen in groups:
        variants = ["frozen_profile", *POLICIES] if frozen is not None else list(POLICIES)
        for policy in variants:
            error = ""
            matrices = []
            try:
                model, matrices = (frozen, [cache.get(path, base, "reenroll_legacy") for path in paths]) if policy == "frozen_profile" else train_profile(speaker, paths, base, policy, size, cache)
            except (srdp.NoSpeechError, ValueError) as failure:
                model, error = None, str(failure)
            directory = output / "candidates" / group / policy
            if model is not None:
                gui.save_user_model(model, directory / "parameters.npz")
                configuration = {"policy": policy, "voice_active": POLICIES.get(policy, (False, False, False))[0], "phrase_span": POLICIES.get(policy, (False, False, False))[1], "sustained": POLICIES.get(policy, (False, False, False))[2], "model_fingerprint": gui.session_model_fingerprint(model), "deployment_status": "experimental_parameters_require_processing_policy; not an App user profile", "enrollment_paths": [srdp.display_path(path) for path in paths]}
                (directory / "processing_policy.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
                policies.append({"scope": scope, "group": group, "variant": policy, "voice_threshold": model.voice_threshold, "phrase_threshold": model.passphrase_threshold, "codebook_size": len(model.codebook), "parameters_path": srdp.display_path(directory / "parameters.npz")})
                vdims = srdp.select_dims(model.feature_names, model.feature_set)
                pdims = srdp.select_dims(model.feature_names, model.passphrase_feature_set)
                held_voice = srdp.leave_one_out_voice_distances([item.voice[:, vdims] for item in matrices], size)
                held_phrase = [srdp.passphrase_template_distance(item.phrase[:, pdims], [other.phrase[:, pdims] for other in matrices], model.passphrase_aggregate, exclude_index=index, window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run) for index, item in enumerate(matrices)]
                for index, item in enumerate(matrices):
                    flags = []
                    if item.activity_seconds < .3:
                        flags.append("short_activity")
                    if item.clip_fraction >= .01:
                        flags.append("clipping")
                    if held_phrase[index] > 3 * max(float(np.median(held_phrase)), 1e-6):
                        flags.append("template_outlier")
                    quality.append({"scope": scope, "group": group, "variant": policy, "enrollment_path": srdp.display_path(paths[index]), "voice_frames": len(item.voice), "phrase_frames": len(item.phrase), "activity_seconds": item.activity_seconds, "activity_contrast": item.activity_contrast, "clip_fraction": item.clip_fraction, "phrase_start": item.phrase_start, "phrase_end": item.phrase_end, "quality_score": item.legacy.quality_score, "loo_voice_score": held_voice[index] / model.voice_threshold, "loo_phrase_score": held_phrase[index] / model.passphrase_threshold, "quality_flags": ";".join(flags)})
            registrations.append({"scope": scope, "group": group, "variant": policy, "registered": model is not None, "error": error})
            for kind, path, expected in cases:
                if model is None:
                    result = {"decision": "REJECT", "voice_score": "", "passphrase_score": "", "fusion_score": "", "reason": "registration_failed"}
                else:
                    try:
                        item = cache.get(path, base, "reenroll_legacy" if policy == "frozen_profile" else policy)
                        result = score_branches(item, model)
                        if policy == "frozen_profile":
                            reference = gui.predict_with_session_model(path, model)
                            assert result["decision"] == reference["decision"]
                            assert abs(float(result["voice_score"]) - float(reference["voice_score"])) <= 2e-6
                            assert abs(float(result["passphrase_score"]) - float(reference["passphrase_score"])) <= 2e-6
                    except srdp.NoSpeechError as failure:
                        result = {"decision": "REJECT", "voice_score": "", "passphrase_score": "", "fusion_score": "", "reason": str(failure)}
                results.append({"scope": scope, "group": group, "variant": policy, "kind": kind, "file_path": srdp.display_path(path), "expected_decision": expected, **result})
            print(f"Evaluated {scope}/{group}/{policy}", flush=True)
        srdp.write_rows(output / "trials.csv", results, list(results[0]))
        summary = summarize(results)
        srdp.write_rows(output / "summary.csv", summary, list(summary[0]))
    for name, rows in (("registration_quality", quality), ("candidate_parameters", policies), ("registrations", registrations)):
        srdp.write_rows(output / f"{name}.csv", rows, list(rows[0]))
    changes = decision_changes(results)
    srdp.write_rows(output / "decision_changes.csv", changes, [*results[0], "change", "before_decision", "before_voice_score", "before_passphrase_score", "before_fusion_score", "before_reason"])
    protocol = {"purpose": "Development ablations, not independent validation; no default deployment", "policies": POLICIES, "activity": "raw RMS 25ms/10ms, threshold 3*max(q20,1e-5), 20ms neighbor context; >=0.20s active before applying filtering; phrase is a continuous active span with 40ms outer padding, retains internal pauses", "normalization": "frozen pretrained global scaler, no test/query refit", "training": "same enrollment WAVs and fixed codebook size for every ablation, whole-recording LOO voice thresholds and original App phrase threshold rule; only registered samples fit thresholds", "formal_simulation": "12 users-by-environments, three existing train WAVs each, all 63 existing evaluation WAVs tested as claims against each registered profile", "limitations": "energy activity labels are heuristic; repeated tails are synthetic; no genuine-user wrong-phrase tests for the two live users; archived GUI profiles may have older voice training, so reenroll_legacy is the ablation control", "sources_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("speech_frames.py", "frame_consistency_experiments.py", "srdp.py", "gui.py")}}
    protocol["sustained_activity"] = "New candidates only: merge activity across gaps <=150ms, discard groups with <60ms accumulated active frames; retain pauses within the final continuous phrase span. Parameters describe an experimental temporal filter, not semantic speech labels."
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} trials to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/registration_diagnosis/20261031/sustained_preprocessing")
    parser.add_argument("--live-manifest", default="tmp/registration_diagnosis/profile_correction.json")
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument("--replay-candidate", default="")
    parser.add_argument("--wav", default="")
    args = parser.parse_args()
    if args.replay_candidate:
        if not args.wav:
            parser.error("--replay-candidate requires --wav")
        print(json.dumps(replay_candidate(srdp.project_path(args.replay_candidate), srdp.project_path(args.wav)), ensure_ascii=False))
    else:
        run(args)
