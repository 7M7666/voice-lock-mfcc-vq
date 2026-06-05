from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import sounddevice as sd
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEMO_AUDIO_DIR = DATA_DIR / "demo_audio"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

TARGET_SR = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
PRE_EMPHASIS = 0.95

FEATURE_NAMES = (
    [f"mfcc_{i}" for i in range(1, 13)]
    + [f"delta_mfcc_{i}" for i in range(1, 13)]
    + [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "zero_crossing_rate",
        "rms_energy",
        "spectral_entropy",
    ]
)

NPG_PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#8491B4", "#F39B7F"]
NPG_RED = NPG_PALETTE[0]
NPG_CYAN = NPG_PALETTE[1]
NPG_GREEN = NPG_PALETTE[2]
NPG_BLUE = NPG_PALETTE[3]
NPG_GREY_BLUE = NPG_PALETTE[4]
NPG_ORANGE = NPG_PALETTE[5]
NPG_GRID = "#E5E7EB"
NPG_TEXT = "#263238"
NPG_CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "npg_confusion",
    ["#FFFFFF", "#DDEBF2", NPG_CYAN, NPG_BLUE],
)
FIGURE_DPI = 300
LOW_QUALITY_THRESHOLD = 0.35

METADATA_FIELDS = [
    "file_path",
    "speaker_id",
    "environment",
    "passphrase",
    "sample_rate_hz",
    "duration_sec",
    "split",
    "expected_decision",
    "notes",
]

OPTIONAL_METADATA_FIELDS = [
    "device",
    "distance",
    "trial_type",
    "is_replay",
]

METADATA_FIELDS += OPTIONAL_METADATA_FIELDS

RESULT_METADATA_FIELDS = [
    "device",
    "recording_distance",
    "trial_type",
    "is_replay",
]


@dataclass
class AudioFeature:
    features: np.ndarray
    duration_sec: float
    sample_rate: int
    voiced_start_sec: float
    voiced_end_sec: float


@dataclass
class AudioInferenceFeatures:
    features: np.ndarray
    duration_sec: float
    sample_rate: int
    voiced_start_sec: float
    voiced_end_sec: float
    estimated_snr_db: float
    quality_score: float
    low_quality: bool


@dataclass
class VoiceLockModel:
    feature_names: list[str]
    zscore_mean: np.ndarray
    zscore_std: np.ndarray
    codebooks: dict[str, np.ndarray]
    thresholds: dict[str, float]
    passphrase_thresholds: dict[str, float]
    passphrase_train_by_speaker: dict[str, list[np.ndarray]]
    feature_set: str
    passphrase_feature_set: str
    passphrase_aggregate: str
    dtw_window_ratio: float
    dtw_warp_penalty: float
    decision_mode: str
    fusion_voice_weight: float
    fusion_threshold: float
    quality_weight: float
    quality_baseline: dict[str, float]
    passphrase_check: bool = True


def ensure_dirs() -> None:
    for path in (DEMO_AUDIO_DIR, FEATURE_DIR, MODEL_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def display_path(path: Path) -> str:
    try:
        return rel(path)
    except ValueError:
        return str(path)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalise_metadata_row(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in METADATA_FIELDS}


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [normalise_metadata_row(row) for row in csv.DictReader(f)]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_npg_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=NPG_TEXT, labelsize=9)
    ax.title.set_color(NPG_TEXT)
    ax.xaxis.label.set_color(NPG_TEXT)
    ax.yaxis.label.set_color(NPG_TEXT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=NPG_GRID, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)


def save_npg_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def append_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(normalise_metadata_row(row) for row in rows)


def load_audio(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"empty audio file: {path}")
    return y.astype(np.float32), sr


def pre_emphasize(y: np.ndarray, coeff: float = PRE_EMPHASIS) -> np.ndarray:
    if y.size < 2:
        return y
    return np.append(y[0], y[1:] - coeff * y[:-1]).astype(np.float32)


def trim_silence(y: np.ndarray, sr: int) -> tuple[np.ndarray, int, int]:
    frame_length = 512
    hop_length = 160
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0 or float(np.max(rms)) <= 1e-10:
        return y, 0, len(y)
    threshold = 0.10 * float(np.max(rms))
    voiced = np.flatnonzero(rms >= threshold)
    if voiced.size == 0:
        return y, 0, len(y)
    pad = int(0.05 * sr)
    start = max(0, int(voiced[0] * hop_length) - pad)
    end = min(len(y), int(voiced[-1] * hop_length + frame_length) + pad)
    return y[start:end], start, end


def spectral_entropy(y: np.ndarray) -> np.ndarray:
    power = np.abs(
        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann",
        )
    ) ** 2
    probs = power / (np.sum(power, axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(probs * np.log2(probs + 1e-12), axis=0)
    return entropy / np.log2(power.shape[0])


def extract_one(audio_path: Path) -> AudioFeature:
    y_raw, sr = load_audio(audio_path)
    y_pre = pre_emphasize(y_raw)
    y_voice, start, end = trim_silence(y_pre, sr)
    if y_voice.size < WIN_LENGTH:
        raise ValueError(f"voiced segment is too short: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=y_voice,
        sr=sr,
        n_mfcc=13,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )[1:13]
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(
        y=y_voice, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_voice, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y_voice, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    flatness = librosa.feature.spectral_flatness(
        y=y_voice, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(
        y_voice, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    rms = librosa.feature.rms(y=y_voice, frame_length=N_FFT, hop_length=HOP_LENGTH)
    entropy = spectral_entropy(y_voice)[None, :]
    min_frames = min(
        mfcc.shape[1],
        delta.shape[1],
        centroid.shape[1],
        bandwidth.shape[1],
        rolloff.shape[1],
        flatness.shape[1],
        zcr.shape[1],
        rms.shape[1],
        entropy.shape[1],
    )
    stacked = np.vstack(
        [
            mfcc[:, :min_frames],
            delta[:, :min_frames],
            centroid[:, :min_frames],
            bandwidth[:, :min_frames],
            rolloff[:, :min_frames],
            flatness[:, :min_frames],
            zcr[:, :min_frames],
            rms[:, :min_frames],
            entropy[:, :min_frames],
        ]
    )
    return AudioFeature(
        features=stacked.T.astype(np.float32),
        duration_sec=float(len(y_raw) / sr),
        sample_rate=sr,
        voiced_start_sec=float(start / sr),
        voiced_end_sec=float(end / sr),
    )


def zscore_fit(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not matrices:
        raise ValueError("no features for z-score fitting")
    stacked = np.vstack(matrices)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def estimate_snr_db(audio_path: Path) -> tuple[float, float]:
    y, sr = load_audio(audio_path)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    if rms.size == 0:
        return 0.0, float(len(y) / sr)
    powers = rms**2
    noise_power = float(np.percentile(powers, 10))
    speech_power = float(np.percentile(powers, 85))
    snr_db = 10.0 * np.log10((speech_power + 1e-12) / (noise_power + 1e-12))
    return float(snr_db), float(len(y) / sr)


def percentile_spread(values: np.ndarray, low: float, high: float) -> float:
    return float(max(np.percentile(values, high) - np.percentile(values, low), 1e-6))


def fit_quality_baseline(rows: list[dict[str, str]], raw_vectors: list[np.ndarray], snr_values: list[float]) -> dict[str, float]:
    train_indices = [idx for idx, row in enumerate(rows) if row.get("split", "").lower() == "train"]
    if not train_indices:
        train_indices = list(range(len(rows)))
    train_snr = np.array([snr_values[idx] for idx in train_indices], dtype=np.float32)
    train_rms = np.array([raw_vectors[idx][FEATURE_NAMES.index("rms_energy")] for idx in train_indices], dtype=np.float32)
    train_flatness = np.array([raw_vectors[idx][FEATURE_NAMES.index("spectral_flatness")] for idx in train_indices], dtype=np.float32)
    return {
        "snr_floor": float(np.percentile(train_snr, 25)),
        "snr_spread": percentile_spread(train_snr, 5, 25),
        "rms_floor": float(np.percentile(train_rms, 10)),
        "rms_spread": percentile_spread(train_rms, 10, 75),
        "flatness_ceiling": float(np.percentile(train_flatness, 90)),
        "flatness_spread": percentile_spread(train_flatness, 50, 90),
    }


def audio_quality_score(raw_vector: np.ndarray, snr_db: float, baseline: dict[str, float]) -> float:
    rms = float(raw_vector[FEATURE_NAMES.index("rms_energy")])
    flatness = float(raw_vector[FEATURE_NAMES.index("spectral_flatness")])
    snr_penalty = max(0.0, (baseline["snr_floor"] - snr_db) / baseline["snr_spread"])
    rms_penalty = max(0.0, (baseline["rms_floor"] - rms) / baseline["rms_spread"])
    flatness_penalty = max(0.0, (flatness - baseline["flatness_ceiling"]) / baseline["flatness_spread"])
    return float(min(2.0, 0.50 * snr_penalty + 0.30 * rms_penalty + 0.20 * flatness_penalty))


def select_dims(feature_names: list[str], feature_set: str) -> list[int]:
    if feature_set == "mfcc":
        prefixes = ("mfcc_",)
    elif feature_set == "mfcc_delta":
        prefixes = ("mfcc_", "delta_mfcc_")
    elif feature_set == "all":
        return list(range(len(feature_names)))
    else:
        raise ValueError(f"unknown feature set: {feature_set}")
    return [idx for idx, name in enumerate(feature_names) if name.startswith(prefixes)]


def lloyd_refine(x: np.ndarray, centers: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    last_distortion = np.inf
    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(len(centers)):
            members = x[labels == idx]
            if members.size:
                new_centers[idx] = members.mean(axis=0)
        distortion = float(np.mean(np.min(distances, axis=1)))
        if abs(last_distortion - distortion) <= tol * max(last_distortion, 1.0):
            return new_centers.astype(np.float32)
        centers = new_centers
        last_distortion = distortion
    return centers.astype(np.float32)


def lbg_codebook(
    x: np.ndarray,
    target_size: int,
    epsilon: float = 0.01,
    max_iter: int = 80,
    tol: float = 1e-5,
) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("empty training matrix")
    if target_size < 1 or target_size & (target_size - 1):
        raise ValueError("strict LBG codebook size must be a power of two, e.g. 4/8/16")

    target_size = min(target_size, x.shape[0])
    centers = x.mean(axis=0, keepdims=True).astype(np.float32)
    global_std = np.maximum(x.std(axis=0), 1e-3).astype(np.float32)

    while centers.shape[0] < target_size:
        perturb = epsilon * np.maximum(np.abs(centers), global_std)
        centers = np.vstack([centers + perturb, centers - perturb])
        centers = lloyd_refine(x, centers, max_iter=max_iter, tol=tol)
    return centers[:target_size].astype(np.float32)


def quantization_distance(x: np.ndarray, codebook: np.ndarray) -> float:
    distances = np.linalg.norm(x[:, None, :] - codebook[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_ratio: float = 1.0,
    warp_penalty: float = 0.0,
) -> float:
    if x.size == 0 or y.size == 0:
        return float("inf")
    n, m = x.shape[0], y.shape[0]
    constrained = 0 < window_ratio < 1.0
    band = max(abs(n - m), int(round(max(n, m) * window_ratio))) if constrained else m
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev_steps = np.zeros(m + 1, dtype=np.int32)
    curr_steps = np.zeros(m + 1, dtype=np.int32)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr.fill(np.inf)
        curr_steps.fill(0)
        frame_costs = np.linalg.norm(y - x[i - 1], axis=1)
        if constrained:
            center = int(round(i * m / n))
            start = max(1, center - band)
            end = min(m, center + band)
        else:
            start, end = 1, m
        for j in range(start, end + 1):
            options = (
                prev[j] + warp_penalty,
                curr[j - 1] + warp_penalty,
                prev[j - 1],
            )
            choice = int(np.argmin(options))
            if choice == 0:
                best_cost, best_steps = options[0], prev_steps[j]
            elif choice == 1:
                best_cost, best_steps = options[1], curr_steps[j - 1]
            else:
                best_cost, best_steps = options[2], prev_steps[j - 1]
            curr[j] = best_cost + float(frame_costs[j - 1])
            curr_steps[j] = best_steps + 1
        prev, curr = curr, prev
        prev_steps, curr_steps = curr_steps, prev_steps
    return float(prev[m] / max(int(prev_steps[m]), 1))


def train_threshold(
    own_distances: list[float],
    impostor_distances: list[float],
    std_factor: float,
    margin: float,
) -> float:
    own = np.array(own_distances, dtype=np.float32)
    base = float(own.mean() + std_factor * own.std()) if own.size > 1 else float(own.mean() * 1.8 + margin)
    if impostor_distances:
        nearest_impostor = float(np.min(impostor_distances))
        base = min(base, float((np.max(own) + nearest_impostor) / 2.0))
    return float(max(base, np.max(own) + 1e-6))


def safe_ratio(value: float, threshold: float) -> float:
    return float(value / threshold) if threshold > 0 else float("inf")


def aggregate_distances(distances: list[float], mode: str) -> float:
    if not distances:
        return float("inf")
    values = sorted(float(value) for value in distances)
    if mode == "min":
        return values[0]
    if mode == "mean3":
        count = min(3, len(values))
        return float(np.mean(values[:count]))
    if mode == "mean5":
        count = min(5, len(values))
        return float(np.mean(values[:count]))
    if mode == "median":
        return float(np.median(values))
    raise ValueError(f"unknown passphrase aggregate mode: {mode}")


def passphrase_template_distance(
    sample: np.ndarray,
    templates: list[np.ndarray],
    aggregate: str,
    *,
    exclude_index: int | None = None,
    window_ratio: float = 1.0,
    warp_penalty: float = 0.0,
) -> float:
    distances = [
        dtw_distance(
            sample,
            template,
            window_ratio=window_ratio,
            warp_penalty=warp_penalty,
        )
        for idx, template in enumerate(templates)
        if idx != exclude_index
    ]
    return aggregate_distances(distances, aggregate)


def train_passphrase_thresholds(
    train_by_speaker: dict[str, list[np.ndarray]],
    mode: str,
    factor: float,
    aggregate: str,
    window_ratio: float,
    warp_penalty: float,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    thresholds, train_distances = {}, {}
    for speaker, templates in train_by_speaker.items():
        if len(templates) < 2:
            thresholds[speaker] = float("inf")
            train_distances[speaker] = []
            continue
        distances = []
        for idx, template in enumerate(templates):
            distances.append(
                passphrase_template_distance(
                    template,
                    templates,
                    aggregate,
                    exclude_index=idx,
                    window_ratio=window_ratio,
                    warp_penalty=warp_penalty,
                )
            )
        values = np.array(distances, dtype=np.float32)
        if mode == "mean_std":
            threshold = float(values.mean() + factor * values.std())
        elif mode == "max_scale":
            threshold = float(values.max() * factor)
        else:
            raise ValueError(f"unknown passphrase threshold mode: {mode}")
        thresholds[speaker] = threshold
        train_distances[speaker] = distances
    return thresholds, train_distances


def load_feature_archive(path: Path):
    data = np.load(path, allow_pickle=False)
    count = len(data["file_path"])
    features = [data[f"features_{idx:04d}"] for idx in range(count)]
    expected_values = (
        data["expected_decision"]
        if "expected_decision" in data.files
        else np.array([""] * count)
    )
    rows = []
    quality_scores = data["quality_score"] if "quality_score" in data.files else np.zeros(count, dtype=np.float32)
    estimated_snr = data["estimated_snr_db"] if "estimated_snr_db" in data.files else np.zeros(count, dtype=np.float32)
    low_quality = data["low_quality"] if "low_quality" in data.files else np.array(["False"] * count)
    for idx in range(count):
        rows.append(
            {
                "file_path": str(data["file_path"][idx]),
                "speaker_id": str(data["speaker_id"][idx]),
                "environment": str(data["environment"][idx]),
                "split": str(data["split"][idx]).lower(),
                "expected_decision": str(expected_values[idx]),
                "estimated_snr_db": f"{float(estimated_snr[idx]):.6f}",
                "quality_score": f"{float(quality_scores[idx]):.6f}",
                "low_quality": str(low_quality[idx]),
                **{
                    field: str(data[field][idx]) if field in data.files else ""
                    for field in OPTIONAL_METADATA_FIELDS
                },
            }
        )
    names = [str(x) for x in data["feature_names"]]
    return names, rows, features


def archive_string(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = data[key]
    if np.ndim(value) == 0:
        return str(value)
    return str(value[0]) if len(value) else default


def archive_float(data, key: str, default: float) -> float:
    if key not in data.files:
        return default
    value = data[key]
    if np.ndim(value) == 0:
        return float(value)
    return float(value[0]) if len(value) else default


def archive_bool(data, key: str, default: bool) -> bool:
    value = archive_string(data, key, str(default))
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_quality_baseline(data) -> dict[str, float]:
    if "quality_baseline" not in data.files or "quality_baseline_keys" not in data.files:
        return {}
    keys = [str(key) for key in data["quality_baseline_keys"]]
    values = [float(value) for value in data["quality_baseline"]]
    return dict(zip(keys, values))


def load_model(
    model_path: str | Path = "models/lbg_vq_codebooks.npz",
    features_path: str | Path = "data/features/srdp_features.npz",
) -> VoiceLockModel:
    model_data = np.load(project_path(model_path), allow_pickle=False)
    feature_data = np.load(project_path(features_path), allow_pickle=False)
    names, rows, features = load_feature_archive(project_path(features_path))
    speakers = [str(speaker) for speaker in model_data["speakers"]]
    feature_set = archive_string(model_data, "feature_set", "mfcc")
    passphrase_feature_set = archive_string(model_data, "passphrase_feature_set", "mfcc")
    passphrase_dims = select_dims(names, passphrase_feature_set)
    passphrase_train_by_speaker = {speaker: [] for speaker in speakers}
    for row, feat in zip(rows, features):
        speaker = row["speaker_id"]
        if row["split"] == "train" and speaker in passphrase_train_by_speaker:
            passphrase_train_by_speaker[speaker].append(feat[:, passphrase_dims])
    return VoiceLockModel(
        feature_names=names,
        zscore_mean=feature_data["zscore_mean"].astype(np.float32),
        zscore_std=feature_data["zscore_std"].astype(np.float32),
        codebooks={speaker: model_data[f"codebook_{speaker}"] for speaker in speakers},
        thresholds={
            speaker: float(value)
            for speaker, value in zip(speakers, model_data["thresholds"])
        },
        passphrase_thresholds={
            speaker: float(value)
            for speaker, value in zip(speakers, model_data["passphrase_thresholds"])
        },
        passphrase_train_by_speaker=passphrase_train_by_speaker,
        feature_set=feature_set,
        passphrase_feature_set=passphrase_feature_set,
        passphrase_aggregate=archive_string(model_data, "passphrase_aggregate", "min"),
        dtw_window_ratio=archive_float(model_data, "dtw_window_ratio", 1.0),
        dtw_warp_penalty=archive_float(model_data, "dtw_warp_penalty", 0.0),
        decision_mode=archive_string(model_data, "decision_mode", "two_layer"),
        fusion_voice_weight=archive_float(model_data, "fusion_voice_weight", 0.5),
        fusion_threshold=archive_float(model_data, "fusion_threshold", 1.0),
        quality_weight=archive_float(model_data, "quality_weight", 0.0),
        quality_baseline=load_quality_baseline(feature_data),
        passphrase_check=archive_bool(model_data, "passphrase_check", True),
    )


def extract_one_audio_features(audio_path: str | Path, model: VoiceLockModel) -> AudioInferenceFeatures:
    audio_path = project_path(audio_path)
    result = extract_one(audio_path)
    snr_db, _ = estimate_snr_db(audio_path)
    raw_vector = result.features.mean(axis=0).astype(np.float32)
    quality_score = (
        audio_quality_score(raw_vector, snr_db, model.quality_baseline)
        if model.quality_baseline
        else 0.0
    )
    return AudioInferenceFeatures(
        features=zscore_apply(result.features, model.zscore_mean, model.zscore_std),
        duration_sec=result.duration_sec,
        sample_rate=result.sample_rate,
        voiced_start_sec=result.voiced_start_sec,
        voiced_end_sec=result.voiced_end_sec,
        estimated_snr_db=snr_db,
        quality_score=quality_score,
        low_quality=quality_score >= LOW_QUALITY_THRESHOLD,
    )


def truthy_text(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def format_reject_reason(row: dict[str, str]) -> str:
    if row.get("decision") != "REJECT":
        return ""
    reasons = []
    if row.get("is_replay", "").strip().lower() == "true" or row.get("split") == "replay":
        reasons.append("回放风险样本")
    if truthy_text(row.get("low_quality", "")):
        reasons.append("录音质量偏低")
    if row.get("voice_accepted") != "True":
        reasons.append("声纹距离超过阈值")
    elif not row.get("passphrase_score"):
        reasons.append("未执行口令验证")
    elif row.get("passphrase_passed") != "True":
        reasons.append("口令DTW距离超过阈值")
    elif row.get("fusion_passed") == "False":
        reasons.append("融合分数超过阈值")
    if not reasons:
        reasons.append("判定未通过")
    return "；".join(dict.fromkeys(reasons))


def predict_feature_matrix(
    features: np.ndarray,
    model: VoiceLockModel,
    metadata: dict[str, str] | None = None,
    *,
    estimated_snr_db: str = "",
    quality_score: float = 0.0,
    low_quality: bool = False,
) -> dict[str, str]:
    row = normalise_metadata_row(metadata or {})
    dims = select_dims(model.feature_names, model.feature_set)
    passphrase_dims = select_dims(model.feature_names, model.passphrase_feature_set)
    speaker_features = features[:, dims]
    passphrase_features = features[:, passphrase_dims]
    distances = {
        speaker: quantization_distance(speaker_features, codebook)
        for speaker, codebook in model.codebooks.items()
    }
    pred = min(distances, key=distances.get)
    distance = distances[pred]
    threshold = model.thresholds[pred]
    voice_score = safe_ratio(distance, threshold)
    voice_accepted = distance <= threshold
    voice_decision = pred if voice_accepted else "REJECT"
    passphrase_distance = ""
    passphrase_threshold = ""
    passphrase_score = ""
    passphrase_passed = ""
    fusion_score = ""
    fusion_threshold = ""
    fusion_passed = ""
    decision = voice_decision

    if model.passphrase_check and voice_accepted and pred in model.passphrase_train_by_speaker:
        passphrase_distance_value = passphrase_template_distance(
            passphrase_features,
            model.passphrase_train_by_speaker[pred],
            model.passphrase_aggregate,
            window_ratio=model.dtw_window_ratio,
            warp_penalty=model.dtw_warp_penalty,
        )
        passphrase_threshold_value = model.passphrase_thresholds[pred]
        passphrase_score_value = safe_ratio(passphrase_distance_value, passphrase_threshold_value)
        passphrase_ok = passphrase_distance_value <= passphrase_threshold_value
        fusion_score_value = (
            model.fusion_voice_weight * voice_score
            + (1.0 - model.fusion_voice_weight) * passphrase_score_value
            + model.quality_weight * quality_score
        )
        fusion_ok = fusion_score_value <= model.fusion_threshold
        passphrase_distance = f"{passphrase_distance_value:.6f}"
        passphrase_threshold = f"{passphrase_threshold_value:.6f}"
        passphrase_score = f"{passphrase_score_value:.6f}"
        passphrase_passed = str(passphrase_ok)
        fusion_score = f"{fusion_score_value:.6f}"
        fusion_threshold = f"{model.fusion_threshold:.6f}"
        fusion_passed = str(fusion_ok)
        if model.decision_mode == "fusion":
            decision = pred if passphrase_ok and fusion_ok else "REJECT"
        else:
            decision = pred if passphrase_ok else "REJECT"

    true_speaker = row.get("speaker_id", "")
    expected = (row.get("expected_decision") or "").strip()
    if not expected and true_speaker:
        expected = true_speaker if true_speaker in model.codebooks else "REJECT"
    if expected and expected not in sorted(model.codebooks) + ["REJECT"]:
        expected = "REJECT"
    voice_correct = str(voice_decision == expected) if expected else ""
    correct = str(decision == expected) if expected else ""
    result = {
        "file_path": row.get("file_path", ""),
        "true_speaker": true_speaker,
        "expected_decision": expected,
        "pred_speaker": pred,
        "voice_decision": voice_decision,
        "voice_accepted": str(voice_accepted),
        "voice_correct": voice_correct,
        "passphrase_distance": passphrase_distance,
        "passphrase_threshold": passphrase_threshold,
        "passphrase_score": passphrase_score,
        "passphrase_passed": passphrase_passed,
        "fusion_score": fusion_score,
        "fusion_threshold": fusion_threshold,
        "fusion_passed": fusion_passed,
        "decision": decision,
        "accepted": str(decision != "REJECT"),
        "correct": correct,
        "environment": row.get("environment", ""),
        "split": row.get("split", ""),
        "device": row.get("device", ""),
        "recording_distance": row.get("distance", ""),
        "trial_type": row.get("trial_type", ""),
        "is_replay": row.get("is_replay", ""),
        "distance": f"{distance:.6f}",
        "threshold": f"{threshold:.6f}",
        "voice_score": f"{voice_score:.6f}",
        "estimated_snr_db": estimated_snr_db,
        "quality_score": f"{quality_score:.6f}",
        "low_quality": str(low_quality),
    }
    result["reject_reason"] = format_reject_reason(result)
    return result


def predict_one(
    wav_path: str | Path,
    model: VoiceLockModel | None = None,
    *,
    model_path: str | Path = "models/lbg_vq_codebooks.npz",
    features_path: str | Path = "data/features/srdp_features.npz",
    metadata: dict[str, str] | None = None,
) -> dict[str, str]:
    model = model or load_model(model_path, features_path)
    extracted = extract_one_audio_features(wav_path, model)
    row = normalise_metadata_row(metadata or {})
    row["file_path"] = row.get("file_path") or display_path(project_path(wav_path))
    return predict_feature_matrix(
        extracted.features,
        model,
        row,
        estimated_snr_db=f"{extracted.estimated_snr_db:.6f}",
        quality_score=extracted.quality_score,
        low_quality=extracted.low_quality,
    )


def cmd_record(args: argparse.Namespace) -> None:
    ensure_dirs()
    out_dir = DEMO_AUDIO_DIR / "srdp_dataset" / args.speaker / args.environment
    rows = []
    for idx in range(1, args.takes + 1):
        audio_path = out_dir / f"{args.speaker}_{args.environment}_{idx:03d}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Recording {args.duration:.1f}s -> {audio_path}")
        input("Press Enter to start...")
        audio = sd.rec(
            int(args.duration * args.sample_rate),
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        sf.write(audio_path, audio, args.sample_rate)
        rows.append(
            {
                "file_path": rel(audio_path),
                "speaker_id": args.speaker,
                "environment": args.environment,
                "passphrase": args.passphrase,
                "sample_rate_hz": str(args.sample_rate),
                "duration_sec": f"{args.duration:.3f}",
                "split": "train" if idx <= args.train_count else "test",
                "expected_decision": args.expected_decision or args.speaker,
                "notes": args.notes,
                "device": args.device,
                "distance": args.distance,
                "trial_type": args.trial_type,
                "is_replay": str(args.is_replay),
            }
        )
    metadata_path = DEMO_AUDIO_DIR / args.metadata
    append_metadata(metadata_path, rows)
    print(f"Metadata appended: {metadata_path}")


def cmd_snr(args: argparse.Namespace) -> None:
    ensure_dirs()
    rows = []
    for row in read_metadata(project_path(args.metadata)):
        audio_path = project_path(row["file_path"])
        if not audio_path.exists():
            print(f"skip missing audio: {audio_path}")
            continue
        snr_db, duration = estimate_snr_db(audio_path)
        rows.append(
            {
                "file_path": row.get("file_path", ""),
                "speaker_id": row.get("speaker_id", ""),
                "environment": row.get("environment", ""),
                "split": row.get("split", ""),
                "device": row.get("device", ""),
                "distance": row.get("distance", ""),
                "trial_type": row.get("trial_type", ""),
                "is_replay": row.get("is_replay", ""),
                "duration_sec": f"{duration:.6f}",
                "estimated_snr_db": f"{snr_db:.3f}",
                "method": "rms_percentile_85_vs_10",
            }
        )
    write_rows(
        project_path(args.output),
        rows,
        [
            "file_path",
            "speaker_id",
            "environment",
            "split",
            *OPTIONAL_METADATA_FIELDS,
            "duration_sec",
            "estimated_snr_db",
            "method",
        ],
    )
    plot_snr_by_environment(rows, project_path(args.figure))
    print(f"Saved SNR estimates: {project_path(args.output)}")


def plot_snr_by_environment(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    envs = sorted({row.get("environment", "unknown") or "unknown" for row in rows})
    values = [
        [float(row["estimated_snr_db"]) for row in rows if (row.get("environment") or "unknown") == env]
        for env in envs
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    box = ax.boxplot(values, tick_labels=envs, showmeans=True, patch_artist=True)
    env_colors = [NPG_CYAN, NPG_GREEN, NPG_BLUE, NPG_GREY_BLUE, NPG_ORANGE]
    for idx, patch in enumerate(box["boxes"]):
        color = env_colors[idx % len(env_colors)]
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(NPG_TEXT)
    for median in box["medians"]:
        median.set_color(NPG_RED)
        median.set_linewidth(1.4)
    ax.set_ylabel("Estimated SNR (dB)")
    ax.set_title("Recording Quality by Environment")
    apply_npg_axis(ax)
    save_npg_figure(fig, path)


def plot_environment_overview(rows: list[dict[str, str]], vectors: list[np.ndarray], path: Path) -> None:
    envs = sorted({row.get("environment", "unknown") or "unknown" for row in rows})
    selected = ["mfcc_1", "spectral_centroid", "spectral_bandwidth", "rms_energy", "spectral_entropy"]
    label_map = {
        "mfcc_1": "MFCC 1",
        "spectral_centroid": "Centroid",
        "spectral_bandwidth": "Bandwidth",
        "rms_energy": "RMS energy",
        "spectral_entropy": "Entropy",
    }
    idxs = [FEATURE_NAMES.index(name) for name in selected]
    values = []
    for env in envs:
        env_vecs = [vec for row, vec in zip(rows, vectors) if (row.get("environment") or "unknown") == env]
        values.append(np.mean(np.vstack(env_vecs), axis=0)[idxs])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(envs))
    width = 0.14
    values_matrix = np.vstack(values)
    for i, name in enumerate(selected):
        ax.bar(
            x + (i - 2) * width,
            values_matrix[:, i],
            width=width,
            label=label_map[name],
            color=NPG_PALETTE[i],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=0)
    ax.set_ylabel("Z-score normalized mean")
    ax.set_title("Environment Feature Overview")
    ax.legend(fontsize=8, frameon=False, ncols=3)
    apply_npg_axis(ax)
    save_npg_figure(fig, path)


def cmd_features(args: argparse.Namespace) -> None:
    ensure_dirs()
    metadata_rows = read_metadata(project_path(args.metadata))
    rows, raw_features, snr_values = [], [], []
    for row in metadata_rows:
        audio_path = project_path(row["file_path"])
        if not audio_path.exists():
            print(f"skip missing audio: {audio_path}")
            continue
        result = extract_one(audio_path)
        snr_db, _ = estimate_snr_db(audio_path)
        out = dict(row)
        out["frames"] = str(result.features.shape[0])
        out["duration_sec"] = f"{result.duration_sec:.6f}"
        out["voiced_start_sec"] = f"{result.voiced_start_sec:.6f}"
        out["voiced_end_sec"] = f"{result.voiced_end_sec:.6f}"
        out["estimated_snr_db"] = f"{snr_db:.6f}"
        rows.append(out)
        raw_features.append(result.features)
        snr_values.append(snr_db)
    if not raw_features:
        raise RuntimeError("no valid audio features extracted")

    raw_vectors = [feat.mean(axis=0).astype(np.float32) for feat in raw_features]
    quality_baseline = fit_quality_baseline(rows, raw_vectors, snr_values)
    quality_scores = [
        audio_quality_score(raw_vec, snr_db, quality_baseline)
        for raw_vec, snr_db in zip(raw_vectors, snr_values)
    ]
    low_quality_flags = [score >= LOW_QUALITY_THRESHOLD for score in quality_scores]

    train_features = [feat for row, feat in zip(rows, raw_features) if row.get("split", "").lower() == "train"]
    mean, std = zscore_fit(train_features or raw_features)
    features = [zscore_apply(feat, mean, std) for feat in raw_features]
    vectors = [feat.mean(axis=0).astype(np.float32) for feat in features]

    archive = {
        "feature_names": np.array(FEATURE_NAMES),
        "zscore_mean": mean,
        "zscore_std": std,
        "file_path": np.array([row.get("file_path", "") for row in rows]),
        "speaker_id": np.array([row.get("speaker_id", "") for row in rows]),
        "environment": np.array([row.get("environment", "") for row in rows]),
        "split": np.array([row.get("split", "") for row in rows]),
        "expected_decision": np.array([row.get("expected_decision", "") for row in rows]),
        **{field: np.array([row.get(field, "") for row in rows]) for field in OPTIONAL_METADATA_FIELDS},
        "utterance_vectors": np.vstack(vectors),
        "estimated_snr_db": np.array(snr_values, dtype=np.float32),
        "quality_score": np.array(quality_scores, dtype=np.float32),
        "low_quality": np.array([str(flag) for flag in low_quality_flags]),
        "quality_baseline": np.array([quality_baseline[key] for key in sorted(quality_baseline)], dtype=np.float32),
        "quality_baseline_keys": np.array(sorted(quality_baseline)),
    }
    for idx, feat in enumerate(features):
        archive[f"features_{idx:04d}"] = feat
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)

    fieldnames = [
        "file_path",
        "speaker_id",
        "environment",
        "split",
        *OPTIONAL_METADATA_FIELDS,
        "expected_decision",
        "frames",
        "duration_sec",
        "voiced_start_sec",
        "voiced_end_sec",
        "estimated_snr_db",
        "quality_score",
        "low_quality",
    ] + [f"mean_{name}" for name in FEATURE_NAMES]
    summary_rows = []
    for row, vec, quality_score, low_quality in zip(rows, vectors, quality_scores, low_quality_flags):
        out = {key: row.get(key, "") for key in fieldnames}
        out["quality_score"] = f"{quality_score:.6f}"
        out["low_quality"] = str(low_quality)
        out.update({f"mean_{name}": f"{value:.6f}" for name, value in zip(FEATURE_NAMES, vec)})
        summary_rows.append(out)
    write_rows(project_path(args.summary), summary_rows, fieldnames)
    plot_environment_overview(rows, vectors, project_path(args.figure))
    print(f"Saved features: {output_path}")
    print(f"Saved summary: {project_path(args.summary)}")


def save_vq_results(path: Path, rows: list[dict[str, str]]) -> None:
    write_rows(
        path,
        rows,
        [
            "file_path",
            "true_speaker",
            "expected_decision",
            "pred_speaker",
            "voice_decision",
            "voice_accepted",
            "voice_correct",
            "passphrase_distance",
            "passphrase_threshold",
            "passphrase_score",
            "passphrase_passed",
            "fusion_score",
            "fusion_threshold",
            "fusion_passed",
            "decision",
            "accepted",
            "correct",
            "environment",
            "split",
            *RESULT_METADATA_FIELDS,
            "distance",
            "threshold",
            "voice_score",
            "estimated_snr_db",
            "quality_score",
            "low_quality",
            "reject_reason",
        ],
    )


def plot_confusion(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=NPG_CONFUSION_CMAP)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("LBG-VQ Recognition / Rejection")
    max_value = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if max_value and matrix[i, j] > max_value * 0.55 else NPG_TEXT
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    apply_npg_axis(ax, grid_axis="")
    save_npg_figure(fig, path)


def save_vq_metrics(path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics = []
    for stage, decision_key, name in (
        ("voice_only", "voice_decision", "LBG-VQ voice only"),
        ("two_layer", "decision", "Voice + passphrase DTW"),
    ):
        for split in ["all"] + sorted({row["split"] for row in rows}):
            subset = rows if split == "all" else [row for row in rows if row["split"] == split]
            if not subset:
                continue
            correct = sum(1 for row in subset if row[decision_key] == row["expected_decision"])
            metrics.append(
                {
                    "stage": stage,
                    "stage_label": name,
                    "split": split,
                    "total": str(len(subset)),
                    "correct": str(correct),
                    "accuracy": f"{correct / len(subset):.6f}",
                }
            )
    write_rows(path, metrics, ["stage", "stage_label", "split", "total", "correct", "accuracy"])
    return metrics


def plot_accuracy_summary(path: Path, metrics: list[dict[str, str]]) -> None:
    splits = ["all", "test", "wrong_passphrase", "impostor"]
    labels = ["Overall", "Registered test", "Wrong passphrase", "Impostor"]
    stages = [("voice_only", "Voice only"), ("two_layer", "Two-layer")]
    colors = {"voice_only": NPG_GREY_BLUE, "two_layer": NPG_GREEN}
    lookup = {(row["stage"], row["split"]): float(row["accuracy"]) for row in metrics}
    x = np.arange(len(splits))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, (stage, label) in enumerate(stages):
        values = [lookup.get((stage, split), 0.0) * 100 for split in splits]
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=label,
            color=colors[stage],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                fontsize=8,
                color=NPG_TEXT,
            )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Voice Lock Accuracy by Evaluation Split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.legend(frameon=False)
    apply_npg_axis(ax)
    save_npg_figure(fig, path)


def plot_passphrase_distances(path: Path, rows: list[dict[str, str]]) -> None:
    groups = [
        ("test", "Registered test"),
        ("wrong_passphrase", "Wrong passphrase"),
        ("impostor", "Impostor"),
    ]
    values = []
    labels = []
    for split, label in groups:
        ratios = [
            float(row["passphrase_distance"]) / float(row["passphrase_threshold"])
            for row in rows
            if row["split"] == split and row["passphrase_distance"] and row["passphrase_threshold"]
        ]
        if ratios:
            values.append(ratios)
            labels.append(label)
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    box = ax.boxplot(values, tick_labels=labels, showmeans=True, patch_artist=True)
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor([NPG_GREEN, NPG_RED, NPG_GREY_BLUE][idx % 3])
        patch.set_alpha(0.38)
        patch.set_edgecolor(NPG_TEXT)
    for median in box["medians"]:
        median.set_color(NPG_BLUE)
        median.set_linewidth(1.4)
    ax.axhline(1.0, color=NPG_RED, linestyle="--", linewidth=1.2, label="Passphrase threshold")
    ax.set_ylabel("DTW distance / threshold")
    ax.set_title("Passphrase Verification Distance")
    ax.legend(frameon=False)
    apply_npg_axis(ax)
    save_npg_figure(fig, path)


def plot_error_breakdown(path: Path, rows: list[dict[str, str]]) -> None:
    categories = {"False accept": 0, "False reject": 0, "Wrong speaker": 0}
    for row in rows:
        expected = row["expected_decision"]
        decision = row["decision"]
        if decision == expected:
            continue
        if expected == "REJECT" and decision != "REJECT":
            categories["False accept"] += 1
        elif expected != "REJECT" and decision == "REJECT":
            categories["False reject"] += 1
        else:
            categories["Wrong speaker"] += 1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(categories)
    values = [categories[label] for label in labels]
    bars = ax.bar(
        labels,
        values,
        color=[NPG_RED, NPG_BLUE, NPG_GREY_BLUE],
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, str(value), ha="center", color=NPG_TEXT)
    ax.set_ylabel("Error count")
    ax.set_title("Remaining Error Types After Two-Layer Verification")
    ax.set_ylim(0, max(values + [1]) + 2)
    apply_npg_axis(ax)
    save_npg_figure(fig, path)


def plot_pipeline_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    boxes = [
        ("Audio", 0.06),
        ("MFCC\nfeatures", 0.24),
        ("LBG-VQ\nvoice ID", 0.42),
        ("MFCC-DTW\npassphrase check", 0.62),
        ("ACCEPT /\nREJECT", 0.84),
    ]
    box_colors = [NPG_CYAN, NPG_BLUE, NPG_GREY_BLUE, NPG_GREEN, NPG_RED]
    for idx, (label, x) in enumerate(boxes):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="white" if idx in (1, 4) else NPG_TEXT,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": box_colors[idx], "edgecolor": "white", "linewidth": 1.0},
        )
    for (_, x1), (_, x2) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(x2 - 0.08, 0.55), xytext=(x1 + 0.08, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": NPG_TEXT})
    ax.text(0.52, 0.18, "Second layer rejects same-speaker wrong passphrases", ha="center", fontsize=10, color=NPG_TEXT)
    save_npg_figure(fig, path)


def cmd_vq(args: argparse.Namespace) -> None:
    ensure_dirs()
    names, rows, features = load_feature_archive(project_path(args.features))
    dims = select_dims(names, args.feature_set)
    passphrase_dims = select_dims(names, args.passphrase_feature_set)
    sliced = [feat[:, dims] for feat in features]
    passphrase_features = [feat[:, passphrase_dims] for feat in features]

    speakers = sorted({row["speaker_id"] for row in rows if row["speaker_id"] and row["split"] == "train"})
    train_by_speaker = {speaker: [] for speaker in speakers}
    passphrase_train_by_speaker = {speaker: [] for speaker in speakers}
    for row, feat, pass_feat in zip(rows, sliced, passphrase_features):
        if row["split"] == "train" and row["speaker_id"] in train_by_speaker:
            train_by_speaker[row["speaker_id"]].append(feat)
            passphrase_train_by_speaker[row["speaker_id"]].append(pass_feat)

    codebooks = {}
    for speaker in speakers:
        if train_by_speaker[speaker]:
            codebooks[speaker] = lbg_codebook(
                np.vstack(train_by_speaker[speaker]),
                target_size=args.codebook_size,
                epsilon=args.epsilon,
            )
    if not codebooks:
        raise RuntimeError("no LBG-VQ codebooks trained; check metadata train split")

    thresholds = {}
    for speaker, codebook in codebooks.items():
        own = [quantization_distance(feat, codebook) for feat in train_by_speaker[speaker]]
        impostor = []
        for other, matrices in train_by_speaker.items():
            if other == speaker:
                continue
            impostor.extend(quantization_distance(feat, codebook) for feat in matrices)
        thresholds[speaker] = train_threshold(own, impostor, args.threshold_std, args.threshold_margin)

    passphrase_thresholds, passphrase_train_distances = train_passphrase_thresholds(
        passphrase_train_by_speaker,
        args.passphrase_threshold_mode,
        args.passphrase_threshold_factor,
        args.passphrase_aggregate,
        args.dtw_window_ratio,
        args.dtw_warp_penalty,
    )
    batch_model = VoiceLockModel(
        feature_names=names,
        zscore_mean=np.zeros(len(names), dtype=np.float32),
        zscore_std=np.ones(len(names), dtype=np.float32),
        codebooks=codebooks,
        thresholds=thresholds,
        passphrase_thresholds=passphrase_thresholds,
        passphrase_train_by_speaker=passphrase_train_by_speaker,
        feature_set=args.feature_set,
        passphrase_feature_set=args.passphrase_feature_set,
        passphrase_aggregate=args.passphrase_aggregate,
        dtw_window_ratio=args.dtw_window_ratio,
        dtw_warp_penalty=args.dtw_warp_penalty,
        decision_mode=args.decision_mode,
        fusion_voice_weight=args.fusion_voice_weight,
        fusion_threshold=args.fusion_threshold,
        quality_weight=args.quality_weight,
        quality_baseline={},
        passphrase_check=args.passphrase_check,
    )

    eval_indices = [idx for idx, row in enumerate(rows) if row["split"] != "train"]
    if not eval_indices:
        eval_indices = list(range(len(rows)))

    labels = sorted(codebooks) + ["REJECT"]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    voice_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    result_rows = []
    correct = 0
    voice_correct = 0
    total = 0
    for idx in eval_indices:
        row = rows[idx]
        quality_score_value = float(row.get("quality_score") or 0.0)
        prediction = predict_feature_matrix(
            features[idx],
            batch_model,
            row,
            estimated_snr_db=row.get("estimated_snr_db", ""),
            quality_score=quality_score_value,
            low_quality=truthy_text(row.get("low_quality", "")),
        )
        expected = prediction["expected_decision"] or "REJECT"
        decision = prediction["decision"]
        voice_decision = prediction["voice_decision"]
        is_correct = prediction["correct"] == "True"
        is_voice_correct = prediction["voice_correct"] == "True"
        total += 1
        correct += int(is_correct)
        voice_correct += int(is_voice_correct)
        matrix[labels.index(expected), labels.index(decision)] += 1
        voice_matrix[labels.index(expected), labels.index(voice_decision)] += 1
        result_rows.append(prediction)

    save_vq_results(project_path(args.results), result_rows)
    plot_confusion(project_path(args.figure), labels, matrix)
    plot_confusion(project_path(args.voice_figure), labels, voice_matrix)
    metrics = save_vq_metrics(project_path(args.metrics), result_rows)
    plot_accuracy_summary(project_path(args.accuracy_figure), metrics)
    plot_passphrase_distances(project_path(args.passphrase_figure), result_rows)
    plot_error_breakdown(project_path(args.error_figure), result_rows)
    plot_pipeline_flow(project_path(args.pipeline_figure))
    model_path = project_path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    max_passphrase_train = max((len(passphrase_train_distances[s]) for s in sorted(codebooks)), default=0)
    passphrase_train_matrix = np.full((len(codebooks), max_passphrase_train), np.nan, dtype=np.float32)
    for row_idx, speaker in enumerate(sorted(codebooks)):
        values = passphrase_train_distances[speaker]
        passphrase_train_matrix[row_idx, : len(values)] = values
    np.savez_compressed(
        model_path,
        speakers=np.array(sorted(codebooks)),
        thresholds=np.array([thresholds[s] for s in sorted(codebooks)], dtype=np.float32),
        passphrase_thresholds=np.array([passphrase_thresholds[s] for s in sorted(codebooks)], dtype=np.float32),
        passphrase_threshold_mode=np.array([args.passphrase_threshold_mode]),
        passphrase_threshold_factor=np.array([args.passphrase_threshold_factor], dtype=np.float32),
        passphrase_feature_set=np.array([args.passphrase_feature_set]),
        passphrase_aggregate=np.array([args.passphrase_aggregate]),
        dtw_window_ratio=np.array([args.dtw_window_ratio], dtype=np.float32),
        dtw_warp_penalty=np.array([args.dtw_warp_penalty], dtype=np.float32),
        passphrase_train_distances=passphrase_train_matrix,
        passphrase_check=np.array([str(args.passphrase_check)]),
        decision_mode=np.array([args.decision_mode]),
        fusion_voice_weight=np.array([args.fusion_voice_weight], dtype=np.float32),
        fusion_threshold=np.array([args.fusion_threshold], dtype=np.float32),
        quality_weight=np.array([args.quality_weight], dtype=np.float32),
        feature_set=np.array([args.feature_set]),
        feature_names=np.array([names[idx] for idx in dims]),
        passphrase_feature_names=np.array([names[idx] for idx in passphrase_dims]),
        **{f"codebook_{speaker}": codebooks[speaker] for speaker in sorted(codebooks)},
    )
    print(f"Evaluated {total} utterances; voice_accuracy={voice_correct / total if total else 0:.3f}; final_accuracy={correct / total if total else 0:.3f}")
    print(f"Saved LBG-VQ results: {project_path(args.results)}")
    print(f"Saved confusion matrix: {project_path(args.figure)}")
    print(f"Saved metrics: {project_path(args.metrics)}")
    print(f"Saved LBG-VQ codebooks: {model_path}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    python = sys.executable
    this_file = Path(__file__).resolve()
    vq_command = [
        python,
        str(this_file),
        "vq",
        "--feature-set",
        args.feature_set,
        "--codebook-size",
        str(args.codebook_size),
        "--passphrase-threshold-mode",
        args.passphrase_threshold_mode,
        "--passphrase-threshold-factor",
        str(args.passphrase_threshold_factor),
        "--passphrase-feature-set",
        args.passphrase_feature_set,
        "--passphrase-aggregate",
        args.passphrase_aggregate,
        "--dtw-window-ratio",
        str(args.dtw_window_ratio),
        "--dtw-warp-penalty",
        str(args.dtw_warp_penalty),
        "--decision-mode",
        args.decision_mode,
        "--fusion-voice-weight",
        str(args.fusion_voice_weight),
        "--fusion-threshold",
        str(args.fusion_threshold),
        "--quality-weight",
        str(args.quality_weight),
    ]
    if not args.passphrase_check:
        vq_command.append("--disable-passphrase-check")
    for command in (
        [python, str(this_file), "snr", "--metadata", args.metadata],
        [python, str(this_file), "features", "--metadata", args.metadata],
        vq_command,
    ):
        print(">", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def metric_accuracy(metrics: list[dict[str, str]], stage: str, split: str) -> str:
    for row in metrics:
        if row["stage"] == stage and row["split"] == split:
            return row["accuracy"]
    return ""


def count_error_types(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    false_reject = 0
    false_accept = 0
    wrong_speaker = 0
    for row in rows:
        expected = row["expected_decision"]
        decision = row["decision"]
        if expected != "REJECT" and decision == "REJECT":
            false_reject += 1
        elif expected == "REJECT" and decision != "REJECT":
            false_accept += 1
        elif expected != "REJECT" and decision not in (expected, "REJECT"):
            wrong_speaker += 1
    return false_reject, false_accept, wrong_speaker


DTW_STRATEGIES = {
    "min": {"aggregate": "min", "window_ratio": 1.0, "warp_penalty": 0.0},
    "mean3": {"aggregate": "mean3", "window_ratio": 1.0, "warp_penalty": 0.0},
    "median": {"aggregate": "median", "window_ratio": 1.0, "warp_penalty": 0.0},
    "constrained10": {"aggregate": "min", "window_ratio": 0.10, "warp_penalty": 0.0},
    "constrained20": {"aggregate": "min", "window_ratio": 0.20, "warp_penalty": 0.0},
    "weighted10": {"aggregate": "min", "window_ratio": 1.0, "warp_penalty": 0.10},
}


def parse_feature_codebook_configs(values: list[str]) -> list[tuple[str, int]]:
    configs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"config must use feature_set:codebook_size, got {value!r}")
        feature_set, codebook_size = value.split(":", 1)
        if feature_set not in {"mfcc", "mfcc_delta", "all"}:
            raise ValueError(f"unknown feature set in config: {feature_set}")
        configs.append((feature_set, int(codebook_size)))
    return configs


def factor_tag(value: float) -> str:
    return str(value).replace(".", "p")


def build_vq_command(
    *,
    python: str,
    this_file: Path,
    features: str,
    feature_set: str,
    codebook_size: int,
    passphrase_feature_set: str,
    passphrase_aggregate: str,
    passphrase_threshold_mode: str,
    passphrase_threshold_factor: float,
    dtw_window_ratio: float,
    dtw_warp_penalty: float,
    decision_mode: str,
    fusion_voice_weight: float,
    fusion_threshold: float,
    quality_weight: float,
    result_path: Path,
    metrics_path: Path,
    experiment_dir: Path,
    model_path: Path,
) -> list[str]:
    return [
        python,
        str(this_file),
        "vq",
        "--features",
        features,
        "--feature-set",
        feature_set,
        "--codebook-size",
        str(codebook_size),
        "--passphrase-threshold-mode",
        passphrase_threshold_mode,
        "--passphrase-threshold-factor",
        str(passphrase_threshold_factor),
        "--passphrase-feature-set",
        passphrase_feature_set,
        "--passphrase-aggregate",
        passphrase_aggregate,
        "--dtw-window-ratio",
        str(dtw_window_ratio),
        "--dtw-warp-penalty",
        str(dtw_warp_penalty),
        "--decision-mode",
        decision_mode,
        "--fusion-voice-weight",
        str(fusion_voice_weight),
        "--fusion-threshold",
        str(fusion_threshold),
        "--quality-weight",
        str(quality_weight),
        "--results",
        rel(result_path),
        "--metrics",
        rel(metrics_path),
        "--figure",
        rel(experiment_dir / "confusion_matrix.png"),
        "--voice-figure",
        rel(experiment_dir / "voice_only_confusion_matrix.png"),
        "--accuracy-figure",
        rel(experiment_dir / "accuracy_by_split.png"),
        "--passphrase-figure",
        rel(experiment_dir / "passphrase_dtw_distance.png"),
        "--error-figure",
        rel(experiment_dir / "error_breakdown.png"),
        "--pipeline-figure",
        rel(experiment_dir / "two_layer_pipeline.png"),
        "--model",
        rel(model_path),
    ]


def summarize_decisions(rows: list[dict[str, str]], decisions: list[str]) -> dict[str, str]:
    correct = [decision == row["expected_decision"] for row, decision in zip(rows, decisions)]

    def accuracy(split: str) -> str:
        idxs = [
            idx
            for idx, row in enumerate(rows)
            if split == "all" or row["split"] == split
        ]
        if not idxs:
            return ""
        return f"{sum(correct[idx] for idx in idxs) / len(idxs):.6f}"

    false_reject = 0
    false_accept = 0
    wrong_speaker = 0
    for row, decision in zip(rows, decisions):
        expected = row["expected_decision"]
        if expected != "REJECT" and decision == "REJECT":
            false_reject += 1
        elif expected == "REJECT" and decision != "REJECT":
            false_accept += 1
        elif expected != "REJECT" and decision not in (expected, "REJECT"):
            wrong_speaker += 1

    return {
        "fusion_all_accuracy": accuracy("all"),
        "registered_test_accuracy": accuracy("test"),
        "wrong_passphrase_reject_rate": accuracy("wrong_passphrase"),
        "impostor_reject_rate": accuracy("impostor"),
        "false_reject": str(false_reject),
        "false_accept": str(false_accept),
        "wrong_speaker": str(wrong_speaker),
    }


def fusion_decisions(rows: list[dict[str, str]], voice_weight: float, threshold: float, quality_weight: float) -> list[str]:
    decisions = []
    for row in rows:
        voice_score = float(row["voice_score"])
        if voice_score > 1.0 or not row["passphrase_score"]:
            decisions.append("REJECT")
            continue
        passphrase_score = float(row["passphrase_score"])
        quality_score = float(row.get("quality_score") or 0.0)
        fusion_score = (
            voice_weight * voice_score
            + (1.0 - voice_weight) * passphrase_score
            + quality_weight * quality_score
        )
        if passphrase_score <= 1.0 and fusion_score <= threshold:
            decisions.append(row["pred_speaker"])
        else:
            decisions.append("REJECT")
    return decisions


RECOMMENDED_CONFIG_FIELDS = [
    "recommendation",
    "selection_rule",
    "feature_set",
    "codebook_size",
    "passphrase_feature_set",
    "passphrase_aggregate",
    "passphrase_threshold_mode",
    "passphrase_threshold_factor",
    "fusion_voice_weight",
    "fusion_threshold",
    "quality_weight",
    "fusion_all_accuracy",
    "registered_test_accuracy",
    "wrong_passphrase_reject_rate",
    "impostor_reject_rate",
    "false_accept",
    "false_reject",
    "wrong_speaker",
    "model_path",
    "results_path",
]


def field_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def field_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, "") or default)
    except ValueError:
        return default


def select_recommended_config(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise RuntimeError("no fusion rows available for recommended config")
    candidates = [row for row in rows if field_int(row, "false_accept") == 0]
    if not candidates:
        candidates = rows
    best = sorted(
        candidates,
        key=lambda row: (
            field_float(row, "fusion_all_accuracy"),
            field_float(row, "registered_test_accuracy"),
            -field_int(row, "false_reject", 9999),
            -field_int(row, "wrong_speaker", 9999),
        ),
        reverse=True,
    )[0]
    result = {field: best.get(field, "") for field in RECOMMENDED_CONFIG_FIELDS}
    result["recommendation"] = "safety_first"
    result["selection_rule"] = (
        "prefer false_accept=0, then max fusion_all_accuracy, "
        "then max registered_test_accuracy, then min false_reject"
    )
    if not result["model_path"]:
        result["model_path"] = infer_model_path(best.get("results_path", ""))
    return result


def infer_model_path(results_path: str) -> str:
    prefix = "results/fusion_experiments/"
    suffix = "/recognition_results.csv"
    if results_path.startswith(prefix) and results_path.endswith(suffix):
        tag = results_path[len(prefix) : -len(suffix)]
        return f"models/fusion_experiments/{tag}_codebooks.npz"
    return ""


def write_recommended_config(path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    recommended = select_recommended_config(rows)
    write_rows(path, [recommended], RECOMMENDED_CONFIG_FIELDS)
    return recommended


def cmd_recommend_config(args: argparse.Namespace) -> None:
    recommended = write_recommended_config(
        project_path(args.output),
        read_csv_rows(project_path(args.matrix)),
    )
    print(f"Saved recommended config: {project_path(args.output)}")
    print(json.dumps(recommended, ensure_ascii=False, indent=2))


def cmd_predict_one(args: argparse.Namespace) -> None:
    metadata = {
        "file_path": display_path(project_path(args.wav)),
        "speaker_id": args.speaker_id,
        "environment": args.environment,
        "split": args.split,
        "expected_decision": args.expected_decision,
        "device": args.device,
        "distance": args.distance,
        "trial_type": args.trial_type,
        "is_replay": str(args.is_replay),
    }
    result = predict_one(
        args.wav,
        model_path=args.model,
        features_path=args.features,
        metadata=metadata,
    )
    if args.output:
        write_rows(project_path(args.output), [result], list(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_smoke_test(args: argparse.Namespace) -> None:
    metadata_path = project_path(args.metadata)
    rows = read_metadata(metadata_path)
    if not rows:
        raise RuntimeError(f"metadata has no rows: {metadata_path}")
    missing = [
        row["file_path"]
        for row in rows
        if not project_path(row["file_path"]).exists()
    ]
    if missing:
        raise RuntimeError(f"missing audio files: {len(missing)}; first={missing[0]}")
    print(f"metadata ok: {len(rows)} rows, 0 missing audio files")

    if not args.skip_pipeline:
        command = [sys.executable, str(Path(__file__).resolve()), "pipeline", "--metadata", args.metadata]
        print(">", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    required = [
        "results/tables/snr_estimates.csv",
        "data/features/srdp_features.npz",
        "results/tables/feature_summary.csv",
        "results/tables/lbg_vq_recognition_results.csv",
        "results/tables/voice_lock_metrics.csv",
        "models/lbg_vq_codebooks.npz",
        "results/tables/fusion_matrix.csv",
    ]
    missing_outputs = [
        path
        for path in required
        if not project_path(path).exists() or project_path(path).stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(f"missing smoke-test outputs: {', '.join(missing_outputs)}")

    recommended = write_recommended_config(
        project_path(args.recommended_summary),
        read_csv_rows(project_path(args.fusion_matrix)),
    )
    if field_int(recommended, "false_accept") != 0:
        raise RuntimeError("recommended config is not safety-first: false_accept is not 0")
    model_path = recommended.get("model_path", "")
    if model_path and not project_path(model_path).exists():
        raise RuntimeError(f"recommended model does not exist: {model_path}")
    print(f"recommended config ok: {project_path(args.recommended_summary)}")
    print("smoke test passed")


def cmd_experiments(args: argparse.Namespace) -> None:
    python = sys.executable
    this_file = Path(__file__).resolve()
    features_path = project_path(args.features)
    if args.refresh_features or not features_path.exists():
        feature_command = [
            python,
            str(this_file),
            "features",
            "--metadata",
            args.metadata,
            "--output",
            args.features,
        ]
        print(">", " ".join(feature_command), flush=True)
        subprocess.run(feature_command, cwd=PROJECT_ROOT, check=True)

    summary_rows = []
    output_dir = project_path(args.output_dir)
    model_dir = project_path(args.model_dir)
    for feature_set in args.feature_sets:
        for codebook_size in args.codebook_sizes:
            tag = f"{feature_set}_cb{codebook_size}"
            experiment_dir = output_dir / tag
            result_path = experiment_dir / "recognition_results.csv"
            metrics_path = experiment_dir / "metrics.csv"
            model_path = model_dir / f"{tag}_codebooks.npz"
            vq_command = [
                python,
                str(this_file),
                "vq",
                "--features",
                args.features,
                "--feature-set",
                feature_set,
                "--codebook-size",
                str(codebook_size),
                "--passphrase-threshold-mode",
                args.passphrase_threshold_mode,
                "--passphrase-threshold-factor",
                str(args.passphrase_threshold_factor),
                "--passphrase-feature-set",
                args.passphrase_feature_set,
                "--passphrase-aggregate",
                args.passphrase_aggregate,
                "--dtw-window-ratio",
                str(args.dtw_window_ratio),
                "--dtw-warp-penalty",
                str(args.dtw_warp_penalty),
                "--decision-mode",
                args.decision_mode,
                "--fusion-voice-weight",
                str(args.fusion_voice_weight),
                "--fusion-threshold",
                str(args.fusion_threshold),
                "--quality-weight",
                str(args.quality_weight),
                "--results",
                rel(result_path),
                "--metrics",
                rel(metrics_path),
                "--figure",
                rel(experiment_dir / "confusion_matrix.png"),
                "--voice-figure",
                rel(experiment_dir / "voice_only_confusion_matrix.png"),
                "--accuracy-figure",
                rel(experiment_dir / "accuracy_by_split.png"),
                "--passphrase-figure",
                rel(experiment_dir / "passphrase_dtw_distance.png"),
                "--error-figure",
                rel(experiment_dir / "error_breakdown.png"),
                "--pipeline-figure",
                rel(experiment_dir / "two_layer_pipeline.png"),
                "--model",
                rel(model_path),
            ]
            print(">", " ".join(vq_command), flush=True)
            subprocess.run(vq_command, cwd=PROJECT_ROOT, check=True)

            metrics = read_csv_rows(metrics_path)
            results = read_csv_rows(result_path)
            false_reject, false_accept, wrong_speaker = count_error_types(results)
            summary_rows.append(
                {
                    "feature_set": feature_set,
                    "codebook_size": str(codebook_size),
                    "passphrase_feature_set": args.passphrase_feature_set,
                    "passphrase_aggregate": args.passphrase_aggregate,
                    "passphrase_threshold_factor": str(args.passphrase_threshold_factor),
                    "dtw_window_ratio": str(args.dtw_window_ratio),
                    "dtw_warp_penalty": str(args.dtw_warp_penalty),
                    "decision_mode": args.decision_mode,
                    "fusion_threshold": str(args.fusion_threshold),
                    "quality_weight": str(args.quality_weight),
                    "voice_all_accuracy": metric_accuracy(metrics, "voice_only", "all"),
                    "two_layer_all_accuracy": metric_accuracy(metrics, "two_layer", "all"),
                    "registered_test_accuracy": metric_accuracy(metrics, "two_layer", "test"),
                    "wrong_passphrase_reject_rate": metric_accuracy(metrics, "two_layer", "wrong_passphrase"),
                    "impostor_reject_rate": metric_accuracy(metrics, "two_layer", "impostor"),
                    "false_reject": str(false_reject),
                    "false_accept": str(false_accept),
                    "wrong_speaker": str(wrong_speaker),
                    "results_path": rel(result_path),
                    "metrics_path": rel(metrics_path),
                }
            )

    write_rows(
        project_path(args.summary),
        summary_rows,
        [
            "feature_set",
            "codebook_size",
            "passphrase_feature_set",
            "passphrase_aggregate",
            "passphrase_threshold_factor",
            "dtw_window_ratio",
            "dtw_warp_penalty",
            "decision_mode",
            "fusion_threshold",
            "quality_weight",
            "voice_all_accuracy",
            "two_layer_all_accuracy",
            "registered_test_accuracy",
            "wrong_passphrase_reject_rate",
            "impostor_reject_rate",
            "false_reject",
            "false_accept",
            "wrong_speaker",
            "results_path",
            "metrics_path",
        ],
    )
    print(f"Saved experiment matrix: {project_path(args.summary)}")


def cmd_dtw_experiments(args: argparse.Namespace) -> None:
    python = sys.executable
    this_file = Path(__file__).resolve()
    summary_rows = []
    output_dir = project_path(args.output_dir)
    model_dir = project_path(args.model_dir)
    configs = parse_feature_codebook_configs(args.configs)

    for feature_set, codebook_size in configs:
        for passphrase_feature_set in args.passphrase_feature_sets:
            for strategy_name in args.strategies:
                strategy = DTW_STRATEGIES[strategy_name]
                tag = f"{feature_set}_cb{codebook_size}_{passphrase_feature_set}_{strategy_name}"
                experiment_dir = output_dir / tag
                result_path = experiment_dir / "recognition_results.csv"
                metrics_path = experiment_dir / "metrics.csv"
                model_path = model_dir / f"{tag}_codebooks.npz"
                vq_command = build_vq_command(
                    python=python,
                    this_file=this_file,
                    features=args.features,
                    feature_set=feature_set,
                    codebook_size=codebook_size,
                    passphrase_feature_set=passphrase_feature_set,
                    passphrase_aggregate=strategy["aggregate"],
                    passphrase_threshold_mode=args.passphrase_threshold_mode,
                    passphrase_threshold_factor=args.passphrase_threshold_factor,
                    dtw_window_ratio=strategy["window_ratio"],
                    dtw_warp_penalty=strategy["warp_penalty"],
                    decision_mode="two_layer",
                    fusion_voice_weight=0.5,
                    fusion_threshold=1.0,
                    quality_weight=0.0,
                    result_path=result_path,
                    metrics_path=metrics_path,
                    experiment_dir=experiment_dir,
                    model_path=model_path,
                )
                print(">", " ".join(vq_command), flush=True)
                subprocess.run(vq_command, cwd=PROJECT_ROOT, check=True)

                metrics = read_csv_rows(metrics_path)
                results = read_csv_rows(result_path)
                false_reject, false_accept, wrong_speaker = count_error_types(results)
                summary_rows.append(
                    {
                        "feature_set": feature_set,
                        "codebook_size": str(codebook_size),
                        "passphrase_feature_set": passphrase_feature_set,
                        "dtw_strategy": strategy_name,
                        "passphrase_aggregate": strategy["aggregate"],
                        "passphrase_threshold_mode": args.passphrase_threshold_mode,
                        "passphrase_threshold_factor": str(args.passphrase_threshold_factor),
                        "dtw_window_ratio": str(strategy["window_ratio"]),
                        "dtw_warp_penalty": str(strategy["warp_penalty"]),
                        "voice_all_accuracy": metric_accuracy(metrics, "voice_only", "all"),
                        "two_layer_all_accuracy": metric_accuracy(metrics, "two_layer", "all"),
                        "registered_test_accuracy": metric_accuracy(metrics, "two_layer", "test"),
                        "wrong_passphrase_reject_rate": metric_accuracy(metrics, "two_layer", "wrong_passphrase"),
                        "impostor_reject_rate": metric_accuracy(metrics, "two_layer", "impostor"),
                        "false_reject": str(false_reject),
                        "false_accept": str(false_accept),
                        "wrong_speaker": str(wrong_speaker),
                        "results_path": rel(result_path),
                        "metrics_path": rel(metrics_path),
                    }
                )

    write_rows(
        project_path(args.summary),
        summary_rows,
        [
            "feature_set",
            "codebook_size",
            "passphrase_feature_set",
            "dtw_strategy",
            "passphrase_aggregate",
            "passphrase_threshold_mode",
            "passphrase_threshold_factor",
            "dtw_window_ratio",
            "dtw_warp_penalty",
            "voice_all_accuracy",
            "two_layer_all_accuracy",
            "registered_test_accuracy",
            "wrong_passphrase_reject_rate",
            "impostor_reject_rate",
            "false_reject",
            "false_accept",
            "wrong_speaker",
            "results_path",
            "metrics_path",
        ],
    )
    print(f"Saved DTW strategy matrix: {project_path(args.summary)}")


def cmd_fusion_experiments(args: argparse.Namespace) -> None:
    python = sys.executable
    this_file = Path(__file__).resolve()
    summary_rows = []
    output_dir = project_path(args.output_dir)
    model_dir = project_path(args.model_dir)
    configs = parse_feature_codebook_configs(args.configs)

    for feature_set, codebook_size in configs:
        for passphrase_feature_set in args.passphrase_feature_sets:
            for factor in args.passphrase_threshold_factors:
                tag = (
                    f"{feature_set}_cb{codebook_size}_{passphrase_feature_set}_"
                    f"factor{factor_tag(factor)}"
                )
                experiment_dir = output_dir / tag
                result_path = experiment_dir / "recognition_results.csv"
                metrics_path = experiment_dir / "metrics.csv"
                model_path = model_dir / f"{tag}_codebooks.npz"
                vq_command = build_vq_command(
                    python=python,
                    this_file=this_file,
                    features=args.features,
                    feature_set=feature_set,
                    codebook_size=codebook_size,
                    passphrase_feature_set=passphrase_feature_set,
                    passphrase_aggregate=args.passphrase_aggregate,
                    passphrase_threshold_mode=args.passphrase_threshold_mode,
                    passphrase_threshold_factor=factor,
                    dtw_window_ratio=args.dtw_window_ratio,
                    dtw_warp_penalty=args.dtw_warp_penalty,
                    decision_mode="two_layer",
                    fusion_voice_weight=0.5,
                    fusion_threshold=1.0,
                    quality_weight=0.0,
                    result_path=result_path,
                    metrics_path=metrics_path,
                    experiment_dir=experiment_dir,
                    model_path=model_path,
                )
                print(">", " ".join(vq_command), flush=True)
                subprocess.run(vq_command, cwd=PROJECT_ROOT, check=True)

                metrics = read_csv_rows(metrics_path)
                results = read_csv_rows(result_path)
                false_reject, false_accept, wrong_speaker = count_error_types(results)
                for voice_weight in args.fusion_voice_weights:
                    for fusion_threshold in args.fusion_thresholds:
                        for quality_weight in args.quality_weights:
                            fusion_metrics = summarize_decisions(
                                results,
                                fusion_decisions(results, voice_weight, fusion_threshold, quality_weight),
                            )
                            summary_rows.append(
                                {
                                    "feature_set": feature_set,
                                    "codebook_size": str(codebook_size),
                                    "passphrase_feature_set": passphrase_feature_set,
                                    "passphrase_aggregate": args.passphrase_aggregate,
                                    "passphrase_threshold_mode": args.passphrase_threshold_mode,
                                    "passphrase_threshold_factor": str(factor),
                                    "dtw_window_ratio": str(args.dtw_window_ratio),
                                    "dtw_warp_penalty": str(args.dtw_warp_penalty),
                                    "fusion_voice_weight": str(voice_weight),
                                    "fusion_threshold": str(fusion_threshold),
                                    "quality_weight": str(quality_weight),
                                    "two_layer_all_accuracy": metric_accuracy(metrics, "two_layer", "all"),
                                    "two_layer_false_reject": str(false_reject),
                                    "two_layer_false_accept": str(false_accept),
                                    "two_layer_wrong_speaker": str(wrong_speaker),
                                    **fusion_metrics,
                                    "model_path": rel(model_path),
                                    "results_path": rel(result_path),
                                }
                            )

    write_rows(
        project_path(args.summary),
        summary_rows,
        [
            "feature_set",
            "codebook_size",
            "passphrase_feature_set",
            "passphrase_aggregate",
            "passphrase_threshold_mode",
            "passphrase_threshold_factor",
            "dtw_window_ratio",
            "dtw_warp_penalty",
            "fusion_voice_weight",
            "fusion_threshold",
            "quality_weight",
            "two_layer_all_accuracy",
            "two_layer_false_reject",
            "two_layer_false_accept",
            "two_layer_wrong_speaker",
            "fusion_all_accuracy",
            "registered_test_accuracy",
            "wrong_passphrase_reject_rate",
            "impostor_reject_rate",
            "false_reject",
            "false_accept",
            "wrong_speaker",
            "model_path",
            "results_path",
        ],
    )
    recommended = write_recommended_config(
        project_path(args.recommended_summary),
        summary_rows,
    )
    print(f"Saved fusion matrix: {project_path(args.summary)}")
    print(f"Saved recommended config: {project_path(args.recommended_summary)}")
    print(
        "Recommended safety-first config: "
        f"{recommended['feature_set']} cb{recommended['codebook_size']}, "
        f"passphrase={recommended['passphrase_feature_set']}, "
        f"fusion_accuracy={recommended['fusion_all_accuracy']}, "
        f"false_accept={recommended['false_accept']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SRDP voice-lock command line.")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="record SRDP dataset samples")
    record.add_argument("--speaker", required=True)
    record.add_argument("--environment", required=True)
    record.add_argument("--passphrase", default="芝麻开门")
    record.add_argument("--takes", type=int, default=5)
    record.add_argument("--duration", type=float, default=2.5)
    record.add_argument("--sample-rate", type=int, default=16000)
    record.add_argument("--train-count", type=int, default=3)
    record.add_argument("--metadata", default="metadata_srdp.csv")
    record.add_argument("--expected-decision", default="")
    record.add_argument("--notes", default="")
    record.add_argument("--device", default="")
    record.add_argument("--distance", default="")
    record.add_argument("--trial-type", default="")
    record.add_argument("--is-replay", action="store_true")
    record.set_defaults(func=cmd_record)

    snr = sub.add_parser("snr", help="estimate approximate SNR values")
    snr.add_argument("--metadata", default="data/demo_audio/metadata_srdp.csv")
    snr.add_argument("--output", default="results/tables/snr_estimates.csv")
    snr.add_argument("--figure", default="results/figures/snr_by_environment.png")
    snr.set_defaults(func=cmd_snr)

    features = sub.add_parser("features", help="extract features and apply z-score")
    features.add_argument("--metadata", default="data/demo_audio/metadata_srdp.csv")
    features.add_argument("--output", default="data/features/srdp_features.npz")
    features.add_argument("--summary", default="results/tables/feature_summary.csv")
    features.add_argument("--figure", default="results/figures/environment_feature_overview.png")
    features.set_defaults(func=cmd_features)

    vq = sub.add_parser("vq", help="run strict LBG-VQ recognition and rejection")
    vq.add_argument("--features", default="data/features/srdp_features.npz")
    vq.add_argument("--feature-set", choices=["mfcc", "mfcc_delta", "all"], default="mfcc")
    vq.add_argument("--codebook-size", type=int, default=16)
    vq.add_argument("--epsilon", type=float, default=0.01)
    vq.add_argument("--threshold-std", type=float, default=2.0)
    vq.add_argument("--threshold-margin", type=float, default=0.5)
    vq.add_argument("--disable-passphrase-check", action="store_false", dest="passphrase_check")
    vq.set_defaults(passphrase_check=True)
    vq.add_argument("--passphrase-threshold-mode", choices=["mean_std", "max_scale"], default="mean_std")
    vq.add_argument("--passphrase-threshold-factor", type=float, default=1.0)
    vq.add_argument("--passphrase-feature-set", choices=["mfcc", "mfcc_delta", "all"], default="mfcc")
    vq.add_argument("--passphrase-aggregate", choices=["min", "mean3", "mean5", "median"], default="min")
    vq.add_argument("--dtw-window-ratio", type=float, default=1.0)
    vq.add_argument("--dtw-warp-penalty", type=float, default=0.0)
    vq.add_argument("--decision-mode", choices=["two_layer", "fusion"], default="two_layer")
    vq.add_argument("--fusion-voice-weight", type=float, default=0.5)
    vq.add_argument("--fusion-threshold", type=float, default=1.0)
    vq.add_argument("--quality-weight", type=float, default=0.0)
    vq.add_argument("--results", default="results/tables/lbg_vq_recognition_results.csv")
    vq.add_argument("--metrics", default="results/tables/voice_lock_metrics.csv")
    vq.add_argument("--figure", default="results/figures/lbg_vq_confusion_matrix.png")
    vq.add_argument("--voice-figure", default="results/figures/lbg_vq_voice_only_confusion_matrix.png")
    vq.add_argument("--accuracy-figure", default="results/figures/accuracy_by_split.png")
    vq.add_argument("--passphrase-figure", default="results/figures/passphrase_dtw_distance.png")
    vq.add_argument("--error-figure", default="results/figures/error_breakdown.png")
    vq.add_argument("--pipeline-figure", default="results/figures/two_layer_pipeline.png")
    vq.add_argument("--model", default="models/lbg_vq_codebooks.npz")
    vq.set_defaults(func=cmd_vq)

    predict = sub.add_parser("predict-one", help="predict one wav with the trained model")
    predict.add_argument("wav")
    predict.add_argument("--model", default="models/lbg_vq_codebooks.npz")
    predict.add_argument("--features", default="data/features/srdp_features.npz")
    predict.add_argument("--speaker-id", default="")
    predict.add_argument("--environment", default="")
    predict.add_argument("--split", default="")
    predict.add_argument("--expected-decision", default="")
    predict.add_argument("--device", default="")
    predict.add_argument("--distance", default="")
    predict.add_argument("--trial-type", default="")
    predict.add_argument("--is-replay", action="store_true")
    predict.add_argument("--output", default="")
    predict.set_defaults(func=cmd_predict_one)

    pipeline = sub.add_parser("pipeline", help="run SNR, features, and LBG-VQ")
    pipeline.add_argument("--metadata", default="data/demo_audio/metadata_srdp.csv")
    pipeline.add_argument("--feature-set", choices=["mfcc", "mfcc_delta", "all"], default="mfcc")
    pipeline.add_argument("--codebook-size", type=int, default=16)
    pipeline.add_argument("--disable-passphrase-check", action="store_false", dest="passphrase_check")
    pipeline.set_defaults(passphrase_check=True)
    pipeline.add_argument("--passphrase-threshold-mode", choices=["mean_std", "max_scale"], default="mean_std")
    pipeline.add_argument("--passphrase-threshold-factor", type=float, default=1.0)
    pipeline.add_argument("--passphrase-feature-set", choices=["mfcc", "mfcc_delta", "all"], default="mfcc")
    pipeline.add_argument("--passphrase-aggregate", choices=["min", "mean3", "mean5", "median"], default="min")
    pipeline.add_argument("--dtw-window-ratio", type=float, default=1.0)
    pipeline.add_argument("--dtw-warp-penalty", type=float, default=0.0)
    pipeline.add_argument("--decision-mode", choices=["two_layer", "fusion"], default="two_layer")
    pipeline.add_argument("--fusion-voice-weight", type=float, default=0.5)
    pipeline.add_argument("--fusion-threshold", type=float, default=1.0)
    pipeline.add_argument("--quality-weight", type=float, default=0.0)
    pipeline.set_defaults(func=cmd_pipeline)

    experiments = sub.add_parser("experiments", help="run feature/codebook comparison matrix")
    experiments.add_argument("--metadata", default="data/demo_audio/metadata_srdp.csv")
    experiments.add_argument("--features", default="data/features/srdp_features.npz")
    experiments.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["mfcc", "mfcc_delta", "all"],
        default=["mfcc", "mfcc_delta", "all"],
    )
    experiments.add_argument("--codebook-sizes", nargs="+", type=int, default=[8, 16, 32])
    experiments.add_argument("--passphrase-threshold-mode", choices=["mean_std", "max_scale"], default="mean_std")
    experiments.add_argument("--passphrase-threshold-factor", type=float, default=1.0)
    experiments.add_argument("--passphrase-feature-set", choices=["mfcc", "mfcc_delta", "all"], default="mfcc")
    experiments.add_argument("--passphrase-aggregate", choices=["min", "mean3", "mean5", "median"], default="min")
    experiments.add_argument("--dtw-window-ratio", type=float, default=1.0)
    experiments.add_argument("--dtw-warp-penalty", type=float, default=0.0)
    experiments.add_argument("--decision-mode", choices=["two_layer", "fusion"], default="two_layer")
    experiments.add_argument("--fusion-voice-weight", type=float, default=0.5)
    experiments.add_argument("--fusion-threshold", type=float, default=1.0)
    experiments.add_argument("--quality-weight", type=float, default=0.0)
    experiments.add_argument("--refresh-features", action="store_true")
    experiments.add_argument("--output-dir", default="results/experiments")
    experiments.add_argument("--model-dir", default="models/experiments")
    experiments.add_argument("--summary", default="results/tables/experiment_matrix.csv")
    experiments.set_defaults(func=cmd_experiments)

    dtw = sub.add_parser("dtw-experiments", help="compare passphrase DTW strategies")
    dtw.add_argument("--features", default="data/features/srdp_features.npz")
    dtw.add_argument("--configs", nargs="+", default=["mfcc:16", "all:8"])
    dtw.add_argument("--passphrase-feature-sets", nargs="+", choices=["mfcc", "mfcc_delta", "all"], default=["mfcc"])
    dtw.add_argument("--strategies", nargs="+", choices=sorted(DTW_STRATEGIES), default=["min", "mean3", "median", "constrained10", "weighted10"])
    dtw.add_argument("--passphrase-threshold-mode", choices=["mean_std", "max_scale"], default="mean_std")
    dtw.add_argument("--passphrase-threshold-factor", type=float, default=1.0)
    dtw.add_argument("--output-dir", default="results/dtw_experiments")
    dtw.add_argument("--model-dir", default="models/dtw_experiments")
    dtw.add_argument("--summary", default="results/tables/dtw_strategy_matrix.csv")
    dtw.set_defaults(func=cmd_dtw_experiments)

    fusion = sub.add_parser("fusion-experiments", help="run rule-based fusion weight/threshold matrix")
    fusion.add_argument("--features", default="data/features/srdp_features.npz")
    fusion.add_argument("--configs", nargs="+", default=["mfcc:16", "all:8"])
    fusion.add_argument("--passphrase-feature-sets", nargs="+", choices=["mfcc", "mfcc_delta", "all"], default=["mfcc", "all"])
    fusion.add_argument("--passphrase-aggregate", choices=["min", "mean3", "mean5", "median"], default="min")
    fusion.add_argument("--passphrase-threshold-mode", choices=["mean_std", "max_scale"], default="mean_std")
    fusion.add_argument("--passphrase-threshold-factors", nargs="+", type=float, default=[0.4, 0.6, 0.8, 1.0])
    fusion.add_argument("--dtw-window-ratio", type=float, default=1.0)
    fusion.add_argument("--dtw-warp-penalty", type=float, default=0.0)
    fusion.add_argument("--fusion-voice-weights", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    fusion.add_argument("--fusion-thresholds", nargs="+", type=float, default=[0.85, 0.9, 0.95, 1.0, 1.05])
    fusion.add_argument("--quality-weights", nargs="+", type=float, default=[0.0, 0.2])
    fusion.add_argument("--output-dir", default="results/fusion_experiments")
    fusion.add_argument("--model-dir", default="models/fusion_experiments")
    fusion.add_argument("--summary", default="results/tables/fusion_matrix.csv")
    fusion.add_argument("--recommended-summary", default="results/tables/recommended_config.csv")
    fusion.set_defaults(func=cmd_fusion_experiments)

    recommend = sub.add_parser("recommend-config", help="write the safety-first recommended fusion config")
    recommend.add_argument("--matrix", default="results/tables/fusion_matrix.csv")
    recommend.add_argument("--output", default="results/tables/recommended_config.csv")
    recommend.set_defaults(func=cmd_recommend_config)

    smoke = sub.add_parser("smoke-test", help="verify metadata, pipeline outputs, and recommended config")
    smoke.add_argument("--metadata", default="data/demo_audio/metadata_srdp.csv")
    smoke.add_argument("--skip-pipeline", action="store_true")
    smoke.add_argument("--fusion-matrix", default="results/tables/fusion_matrix.csv")
    smoke.add_argument("--recommended-summary", default="results/tables/recommended_config.csv")
    smoke.set_defaults(func=cmd_smoke_test)

    demo = sub.add_parser("demo", help="run the formal SRDP audio pipeline")
    demo.set_defaults(
        func=lambda args: cmd_pipeline(
            argparse.Namespace(
                metadata="data/demo_audio/metadata_srdp.csv",
                feature_set="mfcc",
                codebook_size=16,
                passphrase_check=True,
                passphrase_threshold_mode="mean_std",
                passphrase_threshold_factor=1.0,
                passphrase_feature_set="mfcc",
                passphrase_aggregate="min",
                dtw_window_ratio=1.0,
                dtw_warp_penalty=0.0,
                decision_mode="two_layer",
                fusion_voice_weight=0.5,
                fusion_threshold=1.0,
                quality_weight=0.0,
            )
        )
    )
    return parser


def main() -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
