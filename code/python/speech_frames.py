"""Experimental branch-specific preprocessing; legacy extraction remains the default."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import srdp


@dataclass
class BranchFeatures:
    voice: np.ndarray
    phrase: np.ndarray
    legacy: srdp.AudioInferenceFeatures
    activity_on_legacy: np.ndarray
    phrase_start: float
    phrase_end: float
    activity_seconds: float
    activity_contrast: float
    clip_fraction: float


def energy_activity(y: np.ndarray, sr: int):
    centered = y.astype(np.float64) - float(np.mean(y))
    rms = srdp.librosa.feature.rms(y=centered, frame_length=srdp.WIN_LENGTH, hop_length=srdp.HOP_LENGTH)[0]
    floor, peak = np.percentile(rms, [20, 95])
    threshold = 3.0 * max(float(floor), 1e-5)
    active = rms >= threshold
    seconds = float(np.count_nonzero(active) * srdp.HOP_LENGTH / sr)
    context = active.copy()
    for offset in (1, 2):
        context[offset:] |= active[:-offset]
        context[:-offset] |= active[offset:]
    return active, context, seconds, float(peak / max(float(floor), 1e-5))


def activity_groups(active: np.ndarray, sr: int) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    groups = []
    for start, end in zip(starts, ends):
        if groups and (start - groups[-1][1]) * srdp.HOP_LENGTH / sr <= .15:
            groups[-1] = (groups[-1][0], int(end))
        else:
            groups.append((int(start), int(end)))
    return groups


def sustained_activity(active: np.ndarray, sr: int) -> np.ndarray:
    selected = np.zeros_like(active)
    for start, end in activity_groups(active, sr):
        if np.count_nonzero(active[start:end]) * srdp.HOP_LENGTH / sr >= .06:
            selected[start:end] = active[start:end]
    return selected


def recording_quality(y: np.ndarray, sr: int) -> dict:
    active, _, seconds, contrast = energy_activity(y, sr)
    sustained = sustained_activity(active, sr)
    indices = np.flatnonzero(sustained)
    start = float(indices[0] * srdp.HOP_LENGTH / sr) if indices.size else 0.0
    end = min(len(y) / sr, float((indices[-1] * srdp.HOP_LENGTH + srdp.WIN_LENGTH / 2) / sr)) if indices.size else 0.0
    groups = activity_groups(active, sr)
    short = [(start, end) for start, end in groups if not sustained[start:end].any()]
    flags = []
    sustained_seconds = float(np.count_nonzero(sustained) * srdp.HOP_LENGTH / sr)
    clipped = float(np.mean(np.abs(y) >= .99))
    if sustained_seconds < .20:
        flags.append("short_activity")
    if clipped >= .01:
        flags.append("clipping")
    if short:
        flags.append("isolated_activity")
    if indices.size and (start < .04 or len(y) / sr - end < .04):
        flags.append("recording_edge")
    return {"activity_seconds": seconds, "sustained_seconds": sustained_seconds, "activity_contrast": contrast,
        "clip_fraction": clipped, "activity_start_sec": start, "activity_end_sec": end,
        "isolated_bursts": len(short), "quality_flags": flags}


def mask_for_region(mask: np.ndarray, start_sec: float, frames: int, sr: int) -> np.ndarray:
    indices = np.rint(start_sec * sr / srdp.HOP_LENGTH + np.arange(frames)).astype(int)
    return mask[np.clip(indices, 0, len(mask) - 1)]


def extract_branches(path: Path, model, *, voice_active=False, phrase_span=False, sustained=False) -> BranchFeatures:
    legacy = srdp.extract_one_audio_features(path, model)
    raw, sr = srdp.load_audio(path)
    active, context, seconds, contrast = energy_activity(raw, sr)
    if sustained:
        active = sustained_activity(active, sr)
        context = active.copy()
        for offset in (1, 2):
            context[offset:] |= active[:-offset]
            context[:-offset] |= active[offset:]
        seconds = float(np.count_nonzero(active) * srdp.HOP_LENGTH / sr)
    on_legacy = mask_for_region(context, legacy.voiced_start_sec, len(legacy.features), sr)
    if (voice_active or phrase_span) and seconds < .20:
        raise srdp.NoSpeechError("活动语音不足0.20秒，实验策略不建模或验证")
    voice = legacy.features[on_legacy] if voice_active else legacy.features
    if voice_active and len(voice) == 0:
        raise srdp.NoSpeechError("裁剪区间内没有活动语音帧")
    start, end = legacy.voiced_start_sec, legacy.voiced_end_sec
    phrase = legacy.features
    if phrase_span:
        indices = np.flatnonzero(active)
        pad = 4 * srdp.HOP_LENGTH
        first = max(0, int(indices[0] * srdp.HOP_LENGTH) - pad)
        last = min(len(raw), int(indices[-1] * srdp.HOP_LENGTH + srdp.WIN_LENGTH // 2) + pad)
        extracted = srdp.extract_audio_region(raw, sr, first, last, cepstral_normalization=model.cepstral_normalization)
        phrase = srdp.zscore_apply(extracted.features, model.zscore_mean, model.zscore_std)
        start, end = first / sr, last / sr
    return BranchFeatures(voice, phrase, legacy, on_legacy, start, end, seconds, contrast, float(np.mean(np.abs(raw) >= .99)))
