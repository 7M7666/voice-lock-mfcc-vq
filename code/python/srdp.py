from __future__ import annotations

import argparse
import csv
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


@dataclass
class AudioFeature:
    features: np.ndarray
    duration_sec: float
    sample_rate: int
    voiced_start_sec: float
    voiced_end_sec: float


def ensure_dirs() -> None:
    for path in (DEMO_AUDIO_DIR, FEATURE_DIR, MODEL_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_metadata(path: Path) -> list[dict[str, str]]:
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
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


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


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("inf")
    n, m = x.shape[0], y.shape[0]
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev_steps = np.zeros(m + 1, dtype=np.int32)
    curr_steps = np.zeros(m + 1, dtype=np.int32)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr.fill(np.inf)
        curr_steps.fill(0)
        frame_costs = np.linalg.norm(y - x[i - 1], axis=1)
        for j in range(1, m + 1):
            options = (prev[j], curr[j - 1], prev[j - 1])
            choice = int(np.argmin(options))
            if choice == 0:
                best_cost, best_steps = prev[j], prev_steps[j]
            elif choice == 1:
                best_cost, best_steps = curr[j - 1], curr_steps[j - 1]
            else:
                best_cost, best_steps = prev[j - 1], prev_steps[j - 1]
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


def train_passphrase_thresholds(
    train_by_speaker: dict[str, list[np.ndarray]],
    mode: str,
    factor: float,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    thresholds, train_distances = {}, {}
    for speaker, templates in train_by_speaker.items():
        if len(templates) < 2:
            thresholds[speaker] = float("inf")
            train_distances[speaker] = []
            continue
        distances = []
        for idx, template in enumerate(templates):
            others = [other for j, other in enumerate(templates) if j != idx]
            distances.append(min(dtw_distance(template, other) for other in others))
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
    for idx in range(count):
        rows.append(
            {
                "file_path": str(data["file_path"][idx]),
                "speaker_id": str(data["speaker_id"][idx]),
                "environment": str(data["environment"][idx]),
                "split": str(data["split"][idx]).lower(),
                "expected_decision": str(expected_values[idx]),
            }
        )
    names = [str(x) for x in data["feature_names"]]
    return names, rows, features


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
                "duration_sec": f"{duration:.6f}",
                "estimated_snr_db": f"{snr_db:.3f}",
                "method": "rms_percentile_85_vs_10",
            }
        )
    write_rows(
        project_path(args.output),
        rows,
        ["file_path", "speaker_id", "environment", "split", "duration_sec", "estimated_snr_db", "method"],
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
    rows, raw_features = [], []
    for row in metadata_rows:
        audio_path = project_path(row["file_path"])
        if not audio_path.exists():
            print(f"skip missing audio: {audio_path}")
            continue
        result = extract_one(audio_path)
        out = dict(row)
        out["frames"] = str(result.features.shape[0])
        out["duration_sec"] = f"{result.duration_sec:.6f}"
        out["voiced_start_sec"] = f"{result.voiced_start_sec:.6f}"
        out["voiced_end_sec"] = f"{result.voiced_end_sec:.6f}"
        rows.append(out)
        raw_features.append(result.features)
    if not raw_features:
        raise RuntimeError("no valid audio features extracted")

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
        "utterance_vectors": np.vstack(vectors),
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
        "expected_decision",
        "frames",
        "duration_sec",
        "voiced_start_sec",
        "voiced_end_sec",
    ] + [f"mean_{name}" for name in FEATURE_NAMES]
    summary_rows = []
    for row, vec in zip(rows, vectors):
        out = {key: row.get(key, "") for key in fieldnames}
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
            "passphrase_passed",
            "decision",
            "accepted",
            "correct",
            "environment",
            "split",
            "distance",
            "threshold",
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
    passphrase_dims = select_dims(names, "mfcc")
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
        distances = {speaker: quantization_distance(sliced[idx], cb) for speaker, cb in codebooks.items()}
        pred = min(distances, key=distances.get)
        distance = distances[pred]
        threshold = thresholds[pred]
        voice_accepted = distance <= threshold
        voice_decision = pred if voice_accepted else "REJECT"
        passphrase_distance = ""
        passphrase_threshold = ""
        passphrase_passed = ""
        decision = voice_decision
        if args.passphrase_check and voice_accepted and pred in passphrase_train_by_speaker:
            passphrase_distance_value = min(
                dtw_distance(passphrase_features[idx], template)
                for template in passphrase_train_by_speaker[pred]
            )
            passphrase_threshold_value = passphrase_thresholds[pred]
            passphrase_ok = passphrase_distance_value <= passphrase_threshold_value
            passphrase_distance = f"{passphrase_distance_value:.6f}"
            passphrase_threshold = f"{passphrase_threshold_value:.6f}"
            passphrase_passed = str(passphrase_ok)
            decision = pred if passphrase_ok else "REJECT"
        true = row["speaker_id"] if row["speaker_id"] in codebooks else "REJECT"
        expected = (row.get("expected_decision") or "").strip()
        if not expected:
            expected = true if true in codebooks else "REJECT"
        if expected not in labels:
            expected = "REJECT"
        is_correct = decision == expected
        is_voice_correct = voice_decision == expected
        total += 1
        correct += int(is_correct)
        voice_correct += int(is_voice_correct)
        matrix[labels.index(expected), labels.index(decision)] += 1
        voice_matrix[labels.index(expected), labels.index(voice_decision)] += 1
        result_rows.append(
            {
                "file_path": row["file_path"],
                "true_speaker": row["speaker_id"],
                "expected_decision": expected,
                "pred_speaker": pred,
                "voice_decision": voice_decision,
                "voice_accepted": str(voice_accepted),
                "voice_correct": str(is_voice_correct),
                "passphrase_distance": passphrase_distance,
                "passphrase_threshold": passphrase_threshold,
                "passphrase_passed": passphrase_passed,
                "decision": decision,
                "accepted": str(decision != "REJECT"),
                "correct": str(is_correct),
                "environment": row["environment"],
                "split": row["split"],
                "distance": f"{distance:.6f}",
                "threshold": f"{threshold:.6f}",
            }
        )

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
        passphrase_train_distances=passphrase_train_matrix,
        feature_set=np.array([args.feature_set]),
        feature_names=np.array([names[idx] for idx in dims]),
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

    pipeline = sub.add_parser("pipeline", help="run SNR, features, and LBG-VQ")
    pipeline.add_argument("--metadata", default="data/demo_audio/metadata_srdp.csv")
    pipeline.add_argument("--feature-set", choices=["mfcc", "mfcc_delta", "all"], default="mfcc")
    pipeline.add_argument("--codebook-size", type=int, default=16)
    pipeline.add_argument("--disable-passphrase-check", action="store_false", dest="passphrase_check")
    pipeline.set_defaults(passphrase_check=True)
    pipeline.add_argument("--passphrase-threshold-mode", choices=["mean_std", "max_scale"], default="mean_std")
    pipeline.add_argument("--passphrase-threshold-factor", type=float, default=1.0)
    pipeline.set_defaults(func=cmd_pipeline)

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
