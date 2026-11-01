from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import time
from threading import Event
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from srdp import (
    NoSpeechError,
    PROJECT_ROOT,
    archive_float,
    archive_string,
    calibration_threshold,
    display_path,
    extract_one_audio_features,
    lbg_codebook,
    leave_one_out_voice_distances,
    load_model,
    load_audio,
    normalise_metadata_row,
    no_speech_result,
    passphrase_template_distance,
    predict_one,
    project_path,
    quantization_distance,
    safe_ratio,
    select_dims,
    speaker_distance_scores,
    verification_decision,
)
from speech_frames import recording_quality
from user_cleanup import build_user_cleanup, execute_user_cleanup


SAMPLE_RATE = 16000
ENROLLMENT_REQUIRED = 8
TEMP_AUDIO_DIR = PROJECT_ROOT / "tmp" / "gui_recordings"
APP_DATA_DIR = PROJECT_ROOT / "app_data"
USER_MODEL_DIR = APP_DATA_DIR / "users"
HISTORY_PATH = APP_DATA_DIR / "history.csv"
PHRASE_EXPLANATION = "口令模板匹配比较声音相似度，受音色、语速和发音影响；未识别成文字，不能据此断言口令内容正确或错误。"

DEMO_SAMPLES = {
    "正常通过 S01": {
        "wav": "data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav",
        "speaker_id": "S01",
        "split": "test",
        "expected_decision": "S01",
        "trial_type": "normal",
    },
    "正常通过 S02": {
        "wav": "data/demo_audio/srdp_dataset/S02/corridor/S02_corridor_005.wav",
        "speaker_id": "S02",
        "split": "test",
        "expected_decision": "S02",
        "trial_type": "normal",
    },
    "错误口令拒绝": {
        "wav": "data/demo_audio/srdp_dataset/S01_wrong/dorm/S01_wrong_003.wav",
        "speaker_id": "S01_wrong",
        "split": "wrong_passphrase",
        "expected_decision": "REJECT",
        "trial_type": "wrong_passphrase",
    },
    "非注册用户拒绝": {
        "wav": "data/demo_audio/srdp_dataset/U01/dorm/U01_corridor_001.wav",
        "speaker_id": "U01",
        "split": "impostor",
        "expected_decision": "REJECT",
        "trial_type": "impostor",
    },
    "边界样例": {
        "wav": "data/demo_audio/srdp_dataset/S01/lab/S01_lab_004.wav",
        "speaker_id": "S01",
        "split": "test",
        "expected_decision": "S01",
        "trial_type": "normal",
    },
}


APP_QSS = """
QMainWindow {
    background: #F4F6F8;
}
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI";
    color: #17212B;
    font-size: 13px;
}
QFrame#Header {
    background: #FFFFFF;
    border-bottom: 1px solid #CBD5E1;
}
QFrame#Sidebar {
    background: #EEF2F6;
    border-right: 1px solid #CBD5E1;
}
QPushButton {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 7px 10px;
}
QPushButton:hover {
    border-color: #2563A6;
}
QPushButton:checked {
    background: #DCEAF7;
    border-color: #2563A6;
    font-weight: 600;
}
QPushButton:disabled {
    background: #E2E8F0;
    color: #94A3B8;
}
QPushButton#PrimaryButton {
    background: #2563A6;
    color: #FFFFFF;
    border-color: #2563A6;
    font-weight: 600;
}
QPushButton#PrimaryButton:disabled {
    background: #E2E8F0;
    color: #94A3B8;
    border-color: #CBD5E1;
}
QLabel#VerdictAccept {
    background: #0B7A53;
    color: #FFFFFF;
    border-radius: 6px;
    padding: 16px;
    font-size: 28px;
    font-weight: 700;
}
QLabel#VerdictReject {
    background: #B42318;
    color: #FFFFFF;
    border-radius: 6px;
    padding: 16px;
    font-size: 28px;
    font-weight: 700;
}
QLabel#VerdictIdle {
    background: #EEF2F6;
    color: #5B6775;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 16px;
    font-size: 24px;
    font-weight: 700;
}
QGroupBox {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #5B6775;
}
QTableWidget {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    gridline-color: #E2E8F0;
}
QProgressBar {
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    background: #EEF2F6;
    height: 14px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2563A6;
    border-radius: 3px;
}
"""


@dataclass
class HistoryRow:
    time_text: str
    source: str
    decision: str
    pred_speaker: str
    voice_score: str
    passphrase_score: str
    fusion_score: str
    reject_reason: str
    latency_ms: str
    file_path: str = ""
    audio_sha256: str = ""
    model_fingerprint: str = ""
    voice_distance: str = ""
    voice_threshold: str = ""
    passphrase_distance: str = ""
    passphrase_threshold: str = ""
    voice_limit: str = ""
    passphrase_limit: str = ""
    fusion_threshold: str = ""
    duration_sec: str = ""
    voiced_start_sec: str = ""
    voiced_end_sec: str = ""
    estimated_snr_db: str = ""
    quality_score: str = ""
    vad_mode: str = ""
    cepstral_normalization: str = ""
    dtw_max_run: str = ""
    decision_mode: str = ""
    rejection_stage: str = ""
    voice_metric: str = ""


@dataclass
class EnrollmentTake:
    path: Path
    duration_sec: float
    estimated_snr_db: float | None
    quality_score: float | None
    low_quality: bool
    accepted: bool
    reason: str
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SessionEnrollmentModel:
    speaker_id: str
    codebook: np.ndarray
    voice_threshold: float
    passphrase_templates: list[np.ndarray]
    passphrase_threshold: float
    feature_set: str
    passphrase_feature_set: str
    feature_names: list[str]
    zscore_mean: np.ndarray
    zscore_std: np.ndarray
    quality_baseline: dict[str, float]
    passphrase_aggregate: str
    dtw_window_ratio: float
    dtw_warp_penalty: float
    passphrase_accept_score: float
    voice_metric: str = "absolute"
    background_codebooks: dict[str, np.ndarray] = field(default_factory=dict)
    decision_mode: str = "two_layer"
    fusion_voice_weight: float = 0.5
    fusion_threshold: float = 1.0
    calibration_mode: str = "legacy"
    vad_mode: str = "trim"
    cepstral_normalization: str = "none"
    dtw_max_run: int = 0
    voice_guard: float = 1.0
    passphrase_guard: float = 1.0


def user_model_path(speaker_id: str, directory: Path = USER_MODEL_DIR) -> Path:
    return directory / f"{hashlib.sha256(speaker_id.encode('utf-8')).hexdigest()}.npz"


def session_model_fingerprint(model: SessionEnrollmentModel) -> str:
    def array_digest(value):
        if isinstance(value, np.ndarray):
            return {"dtype": value.dtype.str, "shape": value.shape, "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        raise TypeError(f"unsupported model field: {type(value).__name__}")

    payload = json.dumps(asdict(model), sort_keys=True, ensure_ascii=False, default=array_digest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_user_model(model: SessionEnrollmentModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(model)
    templates = values.pop("passphrase_templates")
    background = values.pop("background_codebooks")
    values["quality_baseline"] = json.dumps(values["quality_baseline"])
    values["template_count"] = len(templates)
    values.update({f"template_{index}": matrix for index, matrix in enumerate(templates)})
    values["background_speakers"] = np.array(list(background), dtype=str)
    values.update({f"background_{index}": matrix for index, matrix in enumerate(background.values())})
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary_path.replace(path)


def load_user_model(path: Path) -> SessionEnrollmentModel:
    with np.load(path, allow_pickle=False) as data:
        return SessionEnrollmentModel(
            speaker_id=str(data["speaker_id"].item()),
            codebook=data["codebook"].copy(),
            voice_threshold=float(data["voice_threshold"].item()),
            passphrase_templates=[data[f"template_{index}"].copy() for index in range(int(data["template_count"].item()))],
            passphrase_threshold=float(data["passphrase_threshold"].item()),
            feature_set=str(data["feature_set"].item()),
            passphrase_feature_set=str(data["passphrase_feature_set"].item()),
            feature_names=data["feature_names"].tolist(),
            zscore_mean=data["zscore_mean"].copy(),
            zscore_std=data["zscore_std"].copy(),
            quality_baseline=json.loads(str(data["quality_baseline"].item())),
            passphrase_aggregate=str(data["passphrase_aggregate"].item()),
            dtw_window_ratio=float(data["dtw_window_ratio"].item()),
            dtw_warp_penalty=float(data["dtw_warp_penalty"].item()),
            passphrase_accept_score=float(data["passphrase_accept_score"].item()),
            voice_metric=archive_string(data, "voice_metric", "absolute"),
            background_codebooks={str(speaker): data[f"background_{index}"].copy() for index, speaker in enumerate(data["background_speakers"])} if "background_speakers" in data.files else {},
            decision_mode=archive_string(data, "decision_mode", "two_layer"),
            fusion_voice_weight=archive_float(data, "fusion_voice_weight", 0.5),
            fusion_threshold=archive_float(data, "fusion_threshold", 1.0),
            calibration_mode=archive_string(data, "calibration_mode", "legacy"),
            vad_mode=archive_string(data, "vad_mode", "trim"),
            cepstral_normalization=archive_string(data, "cepstral_normalization", "none"),
            dtw_max_run=int(archive_float(data, "dtw_max_run", 0)),
            voice_guard=archive_float(data, "voice_guard", 1.0),
            passphrase_guard=archive_float(data, "passphrase_guard", 1.0),
        )


def save_history(rows: list[HistoryRow], path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HistoryRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    temporary_path.replace(path)


def load_history(path: Path = HISTORY_PATH) -> list[HistoryRow]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [HistoryRow(**row) for row in csv.DictReader(handle) if row["source"].startswith("用户验证:")][:20]


def format_float(value: str | float | None, empty: str = "-") -> str:
    if value is None or value == "":
        return empty
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return str(value)


def score_percent(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return max(0, min(100, int(round(float(value) * 100))))
    except ValueError:
        return 0


def choose_session_codebook_size(frame_count: int) -> int:
    if frame_count >= 16:
        return 16
    if frame_count >= 8:
        return 8
    if frame_count >= 4:
        return 4
    return 1


class TaskCancelled(Exception):
    pass


def record_audio_to_temp(prefix: str, duration_sec: float, sample_rate: int = SAMPLE_RATE, *, cancel=None, progress=None) -> Path:
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}"
    audio_path = TEMP_AUDIO_DIR / f"{prefix}_{timestamp}.wav"
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    started = time.perf_counter()
    try:
        if progress is not None:
            progress(0.0, 0.0)
        while sd.get_stream().active:
            if cancel is not None and cancel.wait(.04):
                raise TaskCancelled()
            if cancel is None:
                time.sleep(.04)
            elapsed = min(duration_sec, time.perf_counter() - started)
            end = min(len(audio), max(1, int(elapsed * sample_rate)))
            if progress is not None:
                progress(elapsed / duration_sec, float(np.max(np.abs(audio[max(0, end - sample_rate // 10):end]))))
        sd.wait()
        if cancel is not None and cancel.is_set():
            raise TaskCancelled()
    finally:
        sd.stop()
    sf.write(audio_path, audio, sample_rate)
    return audio_path


def prepare_audio_display(path: Path):
    audio, sr = librosa.load(path, sr=None, mono=True)
    normalized = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE) if sr != SAMPLE_RATE else audio
    mfcc = librosa.feature.mfcc(y=normalized, sr=SAMPLE_RATE, n_mfcc=13)
    return audio, sr, mfcc


def assess_enrollment_take(wav_path: Path, model) -> tuple[EnrollmentTake, np.ndarray]:
    raw, sr = load_audio(wav_path)
    diagnostics = recording_quality(raw, sr)
    try:
        extracted = extract_one_audio_features(wav_path, model)
    except NoSpeechError as error:
        return EnrollmentTake(wav_path, len(raw) / sr, None, None, True, False,
            f"未计入注册：{error}。请在安静处完整说出口令后重录", diagnostics), np.empty((0, len(model.feature_names)), dtype=np.float32)
    duration_ok = 1.2 <= extracted.duration_sec <= 5.0
    quality_ok = not extracted.low_quality
    snr_warning = extracted.estimated_snr_db < 10.0
    activity_ok = "short_activity" not in diagnostics["quality_flags"]
    accepted = duration_ok and quality_ok and activity_ok
    if not duration_ok:
        reason = "时长不合格，请录制1.2–5秒的完整口令"
    elif not activity_ok:
        reason = f"有效声音仅{diagnostics['sustained_seconds']:.2f}秒，请完整说出口令后重录"
    elif not quality_ok:
        reason = "录音质量偏低，请靠近麦克风并减少背景声后重录"
    elif snr_warning:
        reason = "SNR偏低，建议重录"
    else:
        reason = "合格"
    warnings = []
    if "clipping" in diagnostics["quality_flags"]:
        warnings.append("声音接近满幅，建议稍远离麦克风或降低输入音量后重录")
    if "isolated_activity" in diagnostics["quality_flags"]:
        warnings.append("检测到与主体分离的短促声音，建议避免碰麦克风或桌面后重录")
    if "recording_edge" in diagnostics["quality_flags"]:
        warnings.append("声音接近录音首尾，建议开始后再说、说完稍停后重录")
    if warnings:
        reason += "；" + "；".join(warnings)
    return (
        EnrollmentTake(
            path=wav_path,
            duration_sec=extracted.duration_sec,
            estimated_snr_db=extracted.estimated_snr_db,
            quality_score=extracted.quality_score,
            low_quality=extracted.low_quality,
            accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        ),
        extracted.features,
    )


def build_session_enrollment_model(
    speaker_id: str,
    official_model,
    enrollment_features: list[np.ndarray],
    *,
    passphrase_features: list[np.ndarray] | None = None,
    codebook_size: int | None = None,
) -> SessionEnrollmentModel:
    dims = select_dims(official_model.feature_names, official_model.feature_set)
    passphrase_dims = select_dims(official_model.feature_names, official_model.passphrase_feature_set)
    speaker_mats = [features[:, dims] for features in enrollment_features]
    if passphrase_features is not None and len(passphrase_features) != len(enrollment_features):
        raise ValueError("声纹与口令注册录音数不一致")
    passphrase_mats = [features[:, passphrase_dims] for features in (enrollment_features if passphrase_features is None else passphrase_features)]
    stacked = np.vstack(speaker_mats)
    size = choose_session_codebook_size(stacked.shape[0]) if codebook_size is None else codebook_size
    codebook = lbg_codebook(stacked, target_size=size)
    own_distances = leave_one_out_voice_distances(speaker_mats, codebook_size=size)
    background = {speaker: centers.copy() for speaker, centers in official_model.codebooks.items() if speaker != speaker_id}
    if official_model.voice_metric == "cohort_ratio":
        own_distances = [
            speaker_distance_scores({"": distance, **{speaker: quantization_distance(matrix, centers) for speaker, centers in background.items()}}, "cohort_ratio")[""]
            for matrix, distance in zip(speaker_mats, own_distances)
        ]
    voice_threshold = calibration_threshold(own_distances)

    passphrase_distances = []
    for index, matrix in enumerate(passphrase_mats):
        distance = passphrase_template_distance(
            matrix,
            passphrase_mats,
            official_model.passphrase_aggregate,
            exclude_index=index,
            window_ratio=official_model.dtw_window_ratio,
            warp_penalty=official_model.dtw_warp_penalty,
            max_run=official_model.dtw_max_run,
        )
        if np.isfinite(distance):
            passphrase_distances.append(distance)
    calibrated = official_model.calibration_mode == "leave_one_recording_out"
    if calibrated:
        passphrase_threshold = calibration_threshold(passphrase_distances, std_factor=1.0)
    elif passphrase_distances:
        values = np.array(passphrase_distances, dtype=float)
        passphrase_threshold = float(max(values.mean() + 2.0 * values.std(), values.max() + 1e-6))
    else:
        passphrase_threshold = float("inf")

    return SessionEnrollmentModel(
        speaker_id=speaker_id,
        codebook=codebook,
        voice_threshold=voice_threshold,
        passphrase_templates=passphrase_mats,
        passphrase_threshold=passphrase_threshold,
        feature_set=official_model.feature_set,
        passphrase_feature_set=official_model.passphrase_feature_set,
        feature_names=list(official_model.feature_names),
        zscore_mean=official_model.zscore_mean.copy(),
        zscore_std=official_model.zscore_std.copy(),
        quality_baseline=dict(official_model.quality_baseline),
        passphrase_aggregate=official_model.passphrase_aggregate,
        dtw_window_ratio=official_model.dtw_window_ratio,
        dtw_warp_penalty=official_model.dtw_warp_penalty,
        passphrase_accept_score=1.0,
        voice_metric=official_model.voice_metric,
        background_codebooks=background if official_model.voice_metric == "cohort_ratio" else {},
        decision_mode=official_model.decision_mode if calibrated else "two_layer",
        fusion_voice_weight=official_model.fusion_voice_weight,
        fusion_threshold=official_model.fusion_threshold,
        calibration_mode="leave_one_recording_out" if calibrated else "loo_voice_legacy_phrase",
        vad_mode=official_model.vad_mode,
        cepstral_normalization=official_model.cepstral_normalization,
        dtw_max_run=official_model.dtw_max_run,
        voice_guard=official_model.voice_guard,
        passphrase_guard=official_model.passphrase_guard,
    )


def enrollment_self_check(features: list[np.ndarray], model: SessionEnrollmentModel) -> list[dict]:
    voice = [matrix[:, select_dims(model.feature_names, model.feature_set)] for matrix in features]
    phrase = [matrix[:, select_dims(model.feature_names, model.passphrase_feature_set)] for matrix in features]
    distances = leave_one_out_voice_distances(voice, codebook_size=len(model.codebook))
    if model.voice_metric == "cohort_ratio":
        distances = [speaker_distance_scores({"": distance, **{speaker: quantization_distance(matrix, book)
            for speaker, book in model.background_codebooks.items()}}, "cohort_ratio")[""] for matrix, distance in zip(voice, distances)]
    checks = []
    for index, distance in enumerate(distances):
        phrase_distance = passphrase_template_distance(phrase[index], phrase, model.passphrase_aggregate,
            exclude_index=index, window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty, max_run=model.dtw_max_run)
        voice_score = safe_ratio(distance, model.voice_threshold)
        phrase_score = safe_ratio(phrase_distance, model.passphrase_threshold)
        accepted, fusion = verification_decision(voice_score, phrase_score / model.passphrase_accept_score,
            model.decision_mode, model.fusion_voice_weight, model.fusion_threshold,
            voice_guard=model.voice_guard, passphrase_guard=model.passphrase_guard)
        checks.append({"voice_score": voice_score, "passphrase_score": phrase_score, "fusion_score": fusion, "accepted": accepted})
    return checks


def predict_with_session_model(wav_path: Path, session_model: SessionEnrollmentModel) -> dict[str, str]:
    diagnostics = {
        "file_path": display_path(wav_path),
        "audio_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        "model_fingerprint": session_model_fingerprint(session_model),
        "voice_threshold": repr(session_model.voice_threshold),
        "passphrase_threshold": repr(session_model.passphrase_threshold),
        "voice_limit": "1.00", "passphrase_limit": repr(session_model.passphrase_accept_score),
        "vad_mode": session_model.vad_mode,
        "cepstral_normalization": session_model.cepstral_normalization,
        "dtw_max_run": str(session_model.dtw_max_run),
        "decision_mode": session_model.decision_mode,
        "voice_metric": session_model.voice_metric,
    }
    try:
        extracted = extract_one_audio_features(wav_path, session_model)
    except NoSpeechError as error:
        return {**no_speech_result({"file_path": display_path(wav_path)}, str(error)), **diagnostics, "rejection_stage": "input"}
    dims = select_dims(session_model.feature_names, session_model.feature_set)
    passphrase_dims = select_dims(session_model.feature_names, session_model.passphrase_feature_set)
    speaker_features = extracted.features[:, dims]
    passphrase_features = extracted.features[:, passphrase_dims]
    voice_distance = quantization_distance(speaker_features, session_model.codebook)
    if session_model.voice_metric == "cohort_ratio":
        voice_distance = speaker_distance_scores({"": voice_distance, **{
            speaker: quantization_distance(speaker_features, centers)
            for speaker, centers in session_model.background_codebooks.items()
        }}, "cohort_ratio")[""]
    voice_score = safe_ratio(voice_distance, session_model.voice_threshold)
    voice_ok = voice_distance <= session_model.voice_threshold
    passphrase_distance = passphrase_template_distance(
        passphrase_features,
        session_model.passphrase_templates,
        session_model.passphrase_aggregate,
        window_ratio=session_model.dtw_window_ratio,
        warp_penalty=session_model.dtw_warp_penalty,
        max_run=session_model.dtw_max_run,
    )
    passphrase_score = safe_ratio(passphrase_distance, session_model.passphrase_threshold)
    passphrase_ok = passphrase_score <= session_model.passphrase_accept_score
    accepted, fusion_score = verification_decision(
        voice_score, passphrase_score / session_model.passphrase_accept_score, session_model.decision_mode,
        session_model.fusion_voice_weight, session_model.fusion_threshold,
        voice_guard=session_model.voice_guard, passphrase_guard=session_model.passphrase_guard,
    )
    decision = session_model.speaker_id if accepted else "REJECT"
    if accepted:
        reject_reason = ""
    elif not passphrase_ok and voice_ok:
        reject_reason = "口令模板DTW距离超过阈值"
    elif not passphrase_ok and not voice_ok:
        reject_reason = "声纹距离超过阈值；口令模板DTW距离也超过阈值"
    elif not voice_ok:
        reject_reason = "声纹距离超过阈值"
    elif not accepted:
        reject_reason = "融合分数超过阈值"
    else:
        reject_reason = ""
    return {
        **diagnostics,
        "decision": decision,
        "pred_speaker": session_model.speaker_id,
        "voice_score": f"{voice_score:.6f}",
        "voice_passed": str(voice_ok),
        "voice_limit": "1.00",
        "passphrase_score": f"{passphrase_score:.6f}",
        "passphrase_passed": str(passphrase_ok),
        "passphrase_limit": f"{session_model.passphrase_accept_score:.2f}",
        "fusion_score": f"{fusion_score:.6f}" if session_model.decision_mode in {"fusion", "bounded_fusion"} else "",
        "fusion_threshold": f"{session_model.fusion_threshold:.6f}" if session_model.decision_mode in {"fusion", "bounded_fusion"} else "",
        "estimated_snr_db": f"{extracted.estimated_snr_db:.6f}",
        "quality_score": f"{extracted.quality_score:.6f}",
        "reject_reason": reject_reason,
        "voice_distance": f"{voice_distance:.6f}",
        "passphrase_distance": f"{passphrase_distance:.6f}",
        "duration_sec": repr(extracted.duration_sec),
        "voiced_start_sec": repr(extracted.voiced_start_sec),
        "voiced_end_sec": repr(extracted.voiced_end_sec),
        "rejection_stage": "" if accepted else "voice_and_phrase" if not voice_ok and not passphrase_ok else "voice" if not voice_ok else "passphrase" if not passphrase_ok else "fusion",
    }


class WaveformCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 2.2), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.plot_empty()

    def plot_empty(self) -> None:
        self.axes.clear()
        self.axes.set_title("Waveform")
        self.axes.set_xlabel("Time / s")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True, alpha=0.25)
        self.draw()

    def plot_wav(self, wav_path: Path) -> None:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        self.plot_audio(y, sr)

    def plot_audio(self, y: np.ndarray, sr: int) -> None:
        times = np.arange(len(y)) / float(sr)
        self.axes.clear()
        self.axes.plot(times, y, color="#2563A6", linewidth=0.9)
        self.axes.set_title("Waveform")
        self.axes.set_xlabel("Time / s")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True, alpha=0.25)
        self.draw()


class MfccCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 2.6), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.plot_empty()

    def plot_empty(self) -> None:
        self.axes.clear()
        self.axes.set_title("MFCC")
        self.axes.set_xlabel("Frame")
        self.axes.set_ylabel("Coefficient")
        self.draw()

    def plot_wav(self, wav_path: Path) -> None:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        self.plot_features(mfcc)

    def plot_features(self, mfcc: np.ndarray) -> None:
        self.axes.clear()
        self.axes.imshow(mfcc, aspect="auto", origin="lower", cmap="magma")
        self.axes.set_title("MFCC")
        self.axes.set_xlabel("Frame")
        self.axes.set_ylabel("Coefficient")
        self.draw()


class SessionTaskWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    stage = Signal(str)
    progress = Signal(float, float)

    def __init__(self, job, cancel: Event, parent=None):
        super().__init__(parent)
        self.job = job
        self.cancel = cancel

    def checkpoint(self):
        if self.cancel.is_set():
            raise TaskCancelled()

    def record(self, prefix, duration, sample_rate):
        self.checkpoint()
        path = record_audio_to_temp(prefix, duration, sample_rate, cancel=self.cancel, progress=self.progress.emit)
        self.checkpoint()
        return path

    def run(self):
        try:
            result = self.job(self)
            self.checkpoint()
            self.completed.emit(result)
        except TaskCancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error))


class PredictWorker(QObject):
    finished = Signal(object, str, float)
    failed = Signal(str)

    def __init__(self, wav_path: Path, model, metadata: dict[str, str], source_label: str) -> None:
        super().__init__()
        self.wav_path = wav_path
        self.model = model
        self.metadata = metadata
        self.source_label = source_label

    def run(self) -> None:
        try:
            start = time.perf_counter()
            result = predict_one(self.wav_path, model=self.model, metadata=self.metadata)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.finished.emit(result, self.source_label, latency_ms)
        except Exception as exc:
            self.failed.emit(str(exc))


class SrdpVoiceLockWindow(QMainWindow):
    def __init__(self, user_model_dir: Path = USER_MODEL_DIR, history_path: Path = HISTORY_PATH) -> None:
        super().__init__()
        self.setWindowTitle("SRDP 轻量化声音锁演示系统")
        self.resize(1280, 820)
        self.model = None
        self.history: list[HistoryRow] = []
        self.demo_history: list[HistoryRow] = []
        self.current_audio_path: Path | None = None
        self.current_source_label = "demo"
        self.predict_thread: QThread | None = None
        self.predict_worker: PredictWorker | None = None
        self.enrollment_takes: list[EnrollmentTake] = []
        self.enrollment_features: list[np.ndarray] = []
        self.session_model: SessionEnrollmentModel | None = None
        self.audio_busy = False
        self.session_thread = None
        self.session_worker = None
        self.session_cancel = Event()
        self.session_kind = ""
        self.session_result_applied = False
        self.close_requested = False
        self.user_model_dir = user_model_dir
        self.history_path = history_path
        self.active_user_path: Path | None = None

        self.status_label = QLabel("模型: 加载中")
        self.mic_label = QLabel("麦克风: 未检测")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_verification_page())
        self.stack.addWidget(self.build_enrollment_page())
        self.stack.addWidget(self.build_history_page())
        self.stack.addWidget(self.build_project_page())
        self.stack.addWidget(self.build_settings_page())

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.build_sidebar())
        body.addWidget(self.stack, 1)
        root_layout.addLayout(body, 1)

        self.setCentralWidget(root)
        self.load_model_once()
        self.refresh_saved_users()
        try:
            self.history = load_history(self.history_path)
            self.refresh_history_table()
        except Exception as exc:
            QMessageBox.warning(self, "历史读取失败", str(exc))

    def build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        title = QLabel("基于 MFCC 与 VQ 的多特征融合声音锁原型系统")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.mic_label)
        return header

    def build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(180)
        layout = QVBoxLayout(sidebar)
        nav_items = [
            ("声音验证", 0),
            ("注册用户", 1),
            ("验证记录", 2),
            ("项目实验", 3),
            ("设置", 4),
        ]
        self.nav_buttons = []
        for text, index in nav_items:
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page=index: self.stack.setCurrentIndex(page))
            self.nav_buttons.append(button)
            layout.addWidget(button)
        self.stack.currentChanged.connect(self.update_navigation)
        self.update_navigation(self.stack.currentIndex())
        layout.addStretch(1)
        return sidebar

    def update_navigation(self, index: int) -> None:
        for page, button in enumerate(self.nav_buttons):
            button.setChecked(page == index)

    def build_verification_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("声音验证")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)
        layout.addWidget(QLabel("选择你的用户，点击录音，然后说“芝麻开门”。"))

        self.verify_empty_label = QLabel(f"还没有注册用户，请先录入 {ENROLLMENT_REQUIRED} 条合格语音完成注册。")
        self.verify_empty_label.setWordWrap(True)
        layout.addWidget(self.verify_empty_label)
        users_row = QHBoxLayout()
        self.saved_user_combo = QComboBox()
        self.saved_user_combo.currentIndexChanged.connect(self.load_saved_user)
        users_row.addWidget(QLabel("验证用户"))
        users_row.addWidget(self.saved_user_combo, 1)
        self.go_register_button = QPushButton("注册新用户")
        self.go_register_button.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        users_row.addWidget(self.go_register_button)
        self.delete_user_button = QPushButton("删除所选用户")
        self.delete_user_button.setEnabled(False)
        self.delete_user_button.clicked.connect(self.delete_saved_user)
        users_row.addWidget(self.delete_user_button)
        layout.addLayout(users_row)
        self.delete_user_data_checkbox = QCheckBox("删除用户时，同时清理该账户在App中的关联录音、记录和快照")
        self.delete_user_data_checkbox.setToolTip("默认只删除注册模型；勾选后先显示清理数量，再确认删除。共用文件和无归属录音保留。")
        layout.addWidget(self.delete_user_data_checkbox)
        self.temp_model_label = QLabel("尚未选择用户")
        layout.addWidget(self.temp_model_label)
        self.verify_duration_spin = QDoubleSpinBox()
        self.verify_duration_spin.setRange(1.0, 5.0)
        self.verify_duration_spin.setValue(3.0)
        self.verify_duration_spin.setSuffix(" 秒")
        record_row = QHBoxLayout()
        record_row.addWidget(QLabel("录音时长"))
        record_row.addWidget(self.verify_duration_spin)
        self.temp_verify_button = QPushButton("开始录音验证")
        self.temp_verify_button.setObjectName("PrimaryButton")
        self.temp_verify_button.setEnabled(False)
        self.temp_verify_button.clicked.connect(self.verify_temporary_user_by_recording)
        record_row.addWidget(self.temp_verify_button, 1)
        self.verify_cancel_button = QPushButton("取消本次操作")
        self.verify_cancel_button.setEnabled(False)
        self.verify_cancel_button.clicked.connect(self.cancel_session_task)
        record_row.addWidget(self.verify_cancel_button)
        layout.addLayout(record_row)
        self.verify_capture_progress = QProgressBar()
        self.verify_capture_progress.setRange(0, 100)
        self.verify_capture_progress.setFormat("录音进度 %p%")
        self.verify_level_label = QLabel("录音电平: 等待开始")
        layout.addWidget(self.verify_capture_progress)
        layout.addWidget(self.verify_level_label)
        self.temp_verify_result_label = QLabel("等待录音")
        self.temp_verify_result_label.setObjectName("VerdictIdle")
        self.temp_verify_result_label.setWordWrap(True)
        self.temp_verify_reason_label = QLabel("先注册用户，再开始验证。")
        self.temp_verify_reason_label.setWordWrap(True)
        layout.addWidget(self.temp_verify_result_label)
        layout.addWidget(self.temp_verify_reason_label)
        phrase_help = QLabel(PHRASE_EXPLANATION)
        phrase_help.setWordWrap(True)
        layout.addWidget(phrase_help)
        record_link = QPushButton("查看验证记录")
        record_link.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        layout.addWidget(record_link)

        details_button = QPushButton("查看分数与录音特征")
        details_button.setCheckable(True)
        layout.addWidget(details_button)
        self.verify_details = QWidget()
        details_layout = QVBoxLayout(self.verify_details)
        self.temp_verify_score_label = QLabel("分数: -")
        self.temp_verify_score_label.setWordWrap(True)
        details_layout.addWidget(self.temp_verify_score_label)
        tabs = QTabWidget()
        self.verify_waveform_canvas = WaveformCanvas()
        self.verify_mfcc_canvas = MfccCanvas()
        tabs.addTab(self.verify_waveform_canvas, "录音波形")
        tabs.addTab(self.verify_mfcc_canvas, "MFCC特征")
        details_layout.addWidget(tabs)
        self.verify_details.setVisible(False)
        details_button.toggled.connect(self.verify_details.setVisible)
        layout.addWidget(self.verify_details)
        layout.addStretch(1)
        return page

    def build_project_page(self) -> QWidget:
        page = QTabWidget()
        page.addTab(self.build_demo_page(), "已有音频演示")
        page.addTab(self.build_results_page(), "算法对比结果")
        return page

    def build_demo_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 2)

        input_group = QGroupBox("输入")
        input_layout = QFormLayout(input_group)
        self.sample_combo = QComboBox()
        self.sample_combo.addItems(list(DEMO_SAMPLES))
        self.current_file_label = QLabel("未选择音频")
        self.current_file_label.setWordWrap(True)
        choose_button = QPushButton("选择 WAV")
        choose_button.clicked.connect(self.choose_wav)
        self.predict_button = QPushButton("开始验证")
        self.predict_button.setObjectName("PrimaryButton")
        self.predict_button.clicked.connect(self.predict_current_audio)
        live_note = QLabel("这里用已有音频演示算法。现场录音请到“声音验证”，这里的操作不计入现场验证记录。")
        live_note.setWordWrap(True)
        live_note.setStyleSheet("color: #5B6775;")

        button_row = QHBoxLayout()
        button_row.addWidget(choose_button)
        input_layout.addRow("固定样例", self.sample_combo)
        input_layout.addRow("当前文件", self.current_file_label)
        input_layout.addRow(button_row)
        input_layout.addRow(self.predict_button)
        input_layout.addRow(live_note)

        visual_group = QGroupBox("音频与特征")
        visual_layout = QVBoxLayout(visual_group)
        self.waveform_canvas = WaveformCanvas()
        self.mfcc_canvas = MfccCanvas()
        visual_layout.addWidget(self.waveform_canvas)
        visual_layout.addWidget(self.mfcc_canvas)

        result_group = QGroupBox("判定")
        result_layout = QVBoxLayout(result_group)
        self.verdict_label = QLabel("等待验证")
        self.verdict_label.setObjectName("VerdictIdle")
        self.pred_label = QLabel("预测用户: -")
        self.reason_label = QLabel("拒绝原因: -")
        self.reason_label.setWordWrap(True)
        self.voice_score_label, self.voice_score_bar = self.build_score_row("声纹分数")
        self.passphrase_score_label, self.passphrase_score_bar = self.build_score_row("口令模板分数")
        self.passphrase_score_label.setToolTip(PHRASE_EXPLANATION)
        self.fusion_score_label, self.fusion_score_bar = self.build_score_row("融合分数")
        self.snr_label = QLabel("SNR: -")
        self.quality_label = QLabel("质量分: -")
        result_layout.addWidget(self.verdict_label)
        result_layout.addWidget(self.pred_label)
        result_layout.addWidget(self.reason_label)
        result_layout.addWidget(self.voice_score_label)
        result_layout.addWidget(self.voice_score_bar)
        result_layout.addWidget(self.passphrase_score_label)
        result_layout.addWidget(self.passphrase_score_bar)
        result_layout.addWidget(self.fusion_score_label)
        result_layout.addWidget(self.fusion_score_bar)
        result_layout.addWidget(self.snr_label)
        result_layout.addWidget(self.quality_label)
        result_layout.addStretch(1)

        history_group = QGroupBox("本次音频演示记录")
        history_layout = QVBoxLayout(history_group)
        self.history_table = QTableWidget(0, 9)
        self.history_table.setHorizontalHeaderLabels(["时间", "来源", "判定", "预测", "声纹", "口令模板", "融合", "原因", "耗时ms"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)

        layout.addWidget(input_group, 0, 0)
        layout.addWidget(visual_group, 0, 1)
        layout.addWidget(result_group, 0, 2)
        layout.addWidget(history_group, 1, 0, 1, 3)
        self.sample_combo.currentTextChanged.connect(self.load_selected_demo_sample)
        self.load_selected_demo_sample()
        return page

    def build_enrollment_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)

        settings_group = QGroupBox("注册新用户")
        settings_layout = QFormLayout(settings_group)
        note = QLabel(f"每次说“芝麻开门”。需要录入 {ENROLLMENT_REQUIRED} 条合格语音才能完成注册。")
        note.setWordWrap(True)
        self.enroll_name_input = QLineEdit()
        self.enroll_name_input.setPlaceholderText("输入你的名字")
        self.enroll_name_input.textChanged.connect(self.update_enrollment_actions)
        self.enroll_duration_spin = QDoubleSpinBox()
        self.enroll_duration_spin.setRange(1.0, 5.0)
        self.enroll_duration_spin.setValue(3.0)
        self.enroll_record_button = QPushButton("开始录入一条注册语音")
        self.enroll_record_button.clicked.connect(self.record_enrollment_take)
        self.enroll_build_button = QPushButton("完成注册，去验证")
        self.enroll_build_button.setObjectName("PrimaryButton")
        self.enroll_build_button.setEnabled(False)
        self.enroll_build_button.clicked.connect(self.complete_registration)
        self.enroll_reset_button = QPushButton("清空本次录音，重新录制")
        self.enroll_reset_button.clicked.connect(self.reset_temporary_enrollment)
        self.enroll_status_label = QLabel("录音状态: 等待开始")
        self.enroll_status_label.setWordWrap(True)
        self.enroll_status_label.setStyleSheet("color: #5B6775;")
        settings_layout.addRow(note)
        settings_layout.addRow("新用户名", self.enroll_name_input)
        settings_layout.addRow("固定口令", QLabel("芝麻开门"))
        settings_layout.addRow("录音时长", self.enroll_duration_spin)
        self.enroll_progress = QProgressBar()
        self.enroll_progress.setRange(0, ENROLLMENT_REQUIRED)
        self.enroll_progress.setValue(0)
        self.enroll_progress.setFormat(f"已完成 %v / {ENROLLMENT_REQUIRED} 条合格录音")
        settings_layout.addRow(self.enroll_progress)
        settings_layout.addRow(self.enroll_record_button)
        self.enroll_capture_progress = QProgressBar()
        self.enroll_capture_progress.setRange(0, 100)
        self.enroll_capture_progress.setFormat("录音进度 %p%")
        self.enroll_level_label = QLabel("录音电平: 等待开始")
        self.enroll_cancel_button = QPushButton("取消本次操作")
        self.enroll_cancel_button.setEnabled(False)
        self.enroll_cancel_button.clicked.connect(self.cancel_session_task)
        settings_layout.addRow(self.enroll_capture_progress)
        settings_layout.addRow(self.enroll_level_label)
        settings_layout.addRow(self.enroll_cancel_button)
        settings_layout.addRow(self.enroll_build_button)
        settings_layout.addRow(self.enroll_reset_button)
        settings_layout.addRow(self.enroll_status_label)

        takes_group = QGroupBox("注册样本与特征")
        takes_layout = QVBoxLayout(takes_group)
        self.enroll_take_list = QListWidget()
        self.enroll_take_list.currentRowChanged.connect(self.show_enrollment_quality)
        self.enroll_quality_label = QLabel("录音后可选择样本，查看有效声音与重录建议。")
        self.enroll_quality_label.setWordWrap(True)
        self.enroll_remove_button = QPushButton("移除所选录音后重录")
        self.enroll_remove_button.setEnabled(False)
        self.enroll_remove_button.clicked.connect(self.remove_enrollment_take)
        self.enroll_tabs = QTabWidget()
        self.enroll_tabs.addTab(self.enroll_take_list, "注册样本")
        audio_page = QWidget()
        audio_layout = QVBoxLayout(audio_page)
        self.enroll_waveform_canvas = WaveformCanvas()
        self.enroll_mfcc_canvas = MfccCanvas()
        audio_layout.addWidget(self.enroll_waveform_canvas)
        audio_layout.addWidget(self.enroll_mfcc_canvas)
        self.enroll_tabs.addTab(audio_page, "当前录音特征")
        takes_layout.addWidget(self.enroll_tabs)
        takes_layout.addWidget(self.enroll_quality_label)
        takes_layout.addWidget(self.enroll_remove_button)

        layout.addWidget(settings_group, 0, 0)
        layout.addWidget(takes_group, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        return page

    def build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        summary = QLabel(
            "这里回答“算法在已有录音上表现怎样”，不是你刚才是否验证成功。\n"
            "实验使用99条已采集录音：36条训练、63条评估，其中正常用户24条、错误口令36条、陌生人3条。"
        )
        summary.setWordWrap(True)
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setMaximumHeight(170)
        self.metrics_table.verticalHeader().setVisible(False)
        self.experiment_note = QLabel()
        self.experiment_note.setWordWrap(True)
        self.calibration_table = QTableWidget()
        self.calibration_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.calibration_table.setMaximumHeight(240)
        self.calibration_note = QLabel()
        self.calibration_note.setWordWrap(True)
        reload_button = QPushButton("刷新已有实验结果")
        reload_button.clicked.connect(self.reload_results_tables)
        layout.addWidget(summary)
        layout.addWidget(self.metrics_table)
        layout.addWidget(self.experiment_note)
        layout.addWidget(QLabel("算法校准与候选消融（尚未替换默认模型）"))
        layout.addWidget(self.calibration_table)
        layout.addWidget(self.calibration_note)
        layout.addWidget(reload_button)
        layout.addStretch(1)
        self.reload_results_tables()
        return page

    def build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("现场验证记录：只记录你在“声音验证”中做的操作，保留最近20条。"))
        self.full_history_table = QTableWidget(0, 5)
        self.full_history_table.setHorizontalHeaderLabels(["时间", "验证用户", "结果", "说明", "耗时ms"])
        self.full_history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.full_history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.full_history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.full_history_table.itemSelectionChanged.connect(self.show_history_details)
        self.full_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        clear_button = QPushButton("清空验证记录")
        clear_button.clicked.connect(self.clear_history)
        export_button = QPushButton("导出验证记录")
        export_button.clicked.connect(self.export_history)
        replay_button = QPushButton("复核所选录音")
        replay_button.clicked.connect(self.replay_history_record)
        actions = QHBoxLayout()
        actions.addWidget(clear_button)
        actions.addWidget(export_button)
        actions.addWidget(replay_button)
        actions.addStretch(1)
        layout.addWidget(self.full_history_table)
        self.history_details_label = QLabel("选择一条记录可查看声纹和口令模板分数。")
        self.history_details_label.setWordWrap(True)
        layout.addWidget(self.history_details_label)
        layout.addLayout(actions)
        return page

    def build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.model_path_input = QLineEdit("models/lbg_vq_codebooks.npz")
        self.features_path_input = QLineEdit("data/features/srdp_features.npz")
        self.sample_rate_input = QSpinBox()
        self.sample_rate_input.setRange(8000, 48000)
        self.sample_rate_input.setValue(SAMPLE_RATE)
        self.reload_model_button = QPushButton("重新加载模型")
        self.reload_model_button.clicked.connect(self.reload_model_from_settings)
        self.mic_test_button = QPushButton("测试麦克风 1 秒")
        self.mic_test_button.clicked.connect(self.test_microphone)
        self.algorithm_label = QLabel("算法配置: 等待加载")
        self.algorithm_label.setWordWrap(True)
        note = QLabel("这些设置用于已有音频演示和后续新注册。已保存用户继续使用注册时的模型；需要应用新算法时请重新注册。")
        note.setWordWrap(True)
        layout.addRow(note)
        layout.addRow("模型文件", self.model_path_input)
        layout.addRow("特征文件", self.features_path_input)
        layout.addRow("录音采样率", self.sample_rate_input)
        layout.addRow(self.algorithm_label)
        layout.addRow(self.reload_model_button)
        layout.addRow(self.mic_test_button)
        return page

    def build_score_row(self, label: str) -> tuple[QLabel, QProgressBar]:
        score_label = QLabel(f"{label}: -")
        score_label.setStyleSheet('font-family: "Consolas", "Microsoft YaHei UI";')
        score_bar = QProgressBar()
        score_bar.setRange(0, 100)
        score_bar.setValue(0)
        score_bar.setFormat("%p%")
        return score_label, score_bar

    def load_model_once(self) -> None:
        try:
            self.model = load_model()
            self.show_algorithm_config()
            self.status_label.setText("模型: 已加载")
        except Exception as exc:
            self.status_label.setText("模型: 加载失败")
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def show_algorithm_config(self) -> None:
        model = self.model
        metric = "相对背景距离" if model.voice_metric == "cohort_ratio" else "绝对距离"
        mode = "融合判定" if model.decision_mode == "fusion" else "双门槛判定"
        self.algorithm_label.setText(
            f"算法配置: 声纹 {model.feature_set}，口令 {model.passphrase_feature_set}；{metric}，{mode}。\n"
            f"融合权重 {model.fusion_voice_weight:.2f}，融合阈值 {model.fusion_threshold:.3f}。"
        )

    def load_selected_demo_sample(self) -> None:
        sample = DEMO_SAMPLES[self.sample_combo.currentText()]
        wav_path = project_path(sample["wav"])
        self.current_audio_path = wav_path
        self.current_source_label = self.sample_combo.currentText()
        self.current_file_label.setText(display_path(wav_path))
        self.update_audio_plots(wav_path)

    def choose_wav(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择 WAV 音频", str(PROJECT_ROOT), "WAV files (*.wav)")
        if not filename:
            return
        wav_path = Path(filename)
        self.current_audio_path = wav_path
        self.current_source_label = "选择WAV"
        self.current_file_label.setText(display_path(wav_path))
        self.update_audio_plots(wav_path)

    def update_audio_plots(self, wav_path: Path) -> None:
        try:
            self.waveform_canvas.plot_wav(wav_path)
            self.mfcc_canvas.plot_wav(wav_path)
        except Exception as exc:
            QMessageBox.warning(self, "音频显示失败", str(exc))

    def current_metadata(self) -> dict[str, str]:
        row = normalise_metadata_row({})
        if self.current_source_label in DEMO_SAMPLES:
            row.update(DEMO_SAMPLES[self.current_source_label])
        row["file_path"] = display_path(self.current_audio_path) if self.current_audio_path else ""
        return row

    def predict_current_audio(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "模型未加载", "模型未加载，无法验证。")
            return
        if self.current_audio_path is None:
            QMessageBox.warning(self, "未选择音频", "请先选择固定样例、选择 WAV 或录音。")
            return
        if self.predict_thread is not None:
            QMessageBox.information(self, "验证中", "当前已有一次验证正在进行。")
            return
        self.start_predict_worker(self.current_audio_path, self.current_metadata(), self.current_source_label)

    def start_predict_worker(self, wav_path: Path, metadata: dict[str, str], source_label: str) -> None:
        self.predict_button.setEnabled(False)
        self.status_label.setText("验证: 计算中")
        self.predict_thread = QThread(self)
        self.predict_worker = PredictWorker(wav_path, self.model, metadata, source_label)
        self.predict_worker.moveToThread(self.predict_thread)
        self.predict_thread.started.connect(self.predict_worker.run)
        self.predict_worker.finished.connect(self.on_prediction_finished)
        self.predict_worker.failed.connect(self.on_prediction_failed)
        self.predict_worker.finished.connect(self.predict_thread.quit)
        self.predict_worker.failed.connect(self.predict_thread.quit)
        self.predict_worker.finished.connect(self.predict_worker.deleteLater)
        self.predict_worker.failed.connect(self.predict_worker.deleteLater)
        self.predict_thread.finished.connect(self.predict_thread.deleteLater)
        self.predict_thread.finished.connect(self.on_predict_thread_finished)
        self.predict_thread.start()

    def on_predict_thread_finished(self) -> None:
        self.predict_thread = None
        self.predict_worker = None
        self.predict_button.setEnabled(True)
        self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")
        if self.close_requested:
            QTimer.singleShot(0, self.close)

    def on_prediction_failed(self, message: str) -> None:
        QMessageBox.critical(self, "验证失败", message)

    def on_prediction_finished(self, result: dict[str, str], source_label: str, latency_ms: float) -> None:
        self.update_result_panel(result)
        self.add_history_row(result, source_label, latency_ms)

    def update_result_panel(self, result: dict[str, str]) -> None:
        decision = result.get("decision", "REJECT")
        if decision == "REJECT":
            self.verdict_label.setObjectName("VerdictReject")
            self.verdict_label.setText("REJECT")
        else:
            self.verdict_label.setObjectName("VerdictAccept")
            self.verdict_label.setText(f"ACCEPT {decision}")
        self.verdict_label.style().unpolish(self.verdict_label)
        self.verdict_label.style().polish(self.verdict_label)
        self.pred_label.setText(f"预测用户: {result.get('pred_speaker', '-')}")
        self.reason_label.setText(f"拒绝原因: {result.get('reject_reason') or '无'}")
        self.set_score(self.voice_score_label, self.voice_score_bar, "声纹分数", result.get("voice_score", ""), "1.000")
        self.set_score(self.passphrase_score_label, self.passphrase_score_bar, "口令模板分数", result.get("passphrase_score", ""), "1.000")
        self.set_score(
            self.fusion_score_label,
            self.fusion_score_bar,
            "融合分数",
            result.get("fusion_score", ""),
            format_float(result.get("fusion_threshold", "")),
        )
        self.snr_label.setText(f"SNR: {format_float(result.get('estimated_snr_db', ''))} dB")
        self.quality_label.setText(f"质量分: {format_float(result.get('quality_score', ''))}")

    def set_score(self, label: QLabel, bar: QProgressBar, name: str, value: str | None, threshold: str) -> None:
        label.setText(f"{name}: {format_float(value)} / 阈值 {threshold}")
        bar.setValue(score_percent(value))

    def add_history_row(self, result: dict[str, str], source_label: str, latency_ms: float) -> None:
        history = HistoryRow(
            time_text=time.strftime("%Y-%m-%d %H:%M:%S"),
            source=source_label,
            decision=result.get("decision", ""),
            pred_speaker=result.get("pred_speaker", ""),
            voice_score=result.get("voice_score") or "-",
            passphrase_score=result.get("passphrase_score") or "-",
            fusion_score=result.get("fusion_score") or "-",
            reject_reason=result.get("reject_reason", ""),
            latency_ms=f"{latency_ms:.1f}",
            **{name: result.get(name, "") for name, definition in HistoryRow.__dataclass_fields__.items() if definition.default == ""},
        )
        if not source_label.startswith("用户验证:"):
            self.demo_history.insert(0, history)
            self.demo_history = self.demo_history[:20]
            self.populate_history_table(self.history_table)
            return
        self.history.insert(0, history)
        self.history = self.history[:20]
        self.refresh_history_table()
        try:
            if history.model_fingerprint:
                if session_model_fingerprint(self.session_model) != history.model_fingerprint:
                    raise ValueError("验证记录与当前用户模型不一致，无法保存模型快照")
                snapshot = self.history_path.parent / "diagnostics/models" / f"{history.model_fingerprint}.npz"
                if not snapshot.exists():
                    save_user_model(self.session_model, snapshot)
            save_history(self.history, self.history_path)
        except Exception as exc:
            QMessageBox.warning(self, "历史保存失败", str(exc))

    def populate_history_table(self, table: QTableWidget) -> None:
        rows = self.demo_history if table is self.history_table else self.history
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.time_text,
                row.source,
                row.decision,
                row.pred_speaker,
                row.voice_score,
                row.passphrase_score,
                row.fusion_score,
                row.reject_reason,
                row.latency_ms,
            ]
            if table is self.full_history_table:
                values = [
                    row.time_text,
                    row.source.removeprefix("用户验证:"),
                    "未通过" if row.decision == "REJECT" else "通过",
                    row.reject_reason or "声纹与口令模板匹配均通过",
                    row.latency_ms,
                ]
            for column_index, value in enumerate(values):
                table.setItem(row_index, column_index, QTableWidgetItem(value))

    def refresh_history_table(self) -> None:
        self.populate_history_table(self.history_table)
        self.populate_history_table(self.full_history_table)
        self.history_details_label.setText("选择一条记录可查看声纹和口令模板分数。")

    def show_history_details(self) -> None:
        index = self.full_history_table.currentRow()
        if 0 <= index < len(self.history):
            row = self.history[index]
            self.history_details_label.setText(
                f"用户 {row.source.removeprefix('用户验证:')}｜声纹分数 {row.voice_score}｜口令模板分数 {row.passphrase_score}。"
                "分数是距离与阈值的比值，不是成功概率。"
                + PHRASE_EXPLANATION
            )
            if row.file_path:
                voiced = f"{float(row.voiced_end_sec) - float(row.voiced_start_sec):.3f}秒" if row.voiced_end_sec and row.voiced_start_sec else "未提取"
                self.history_details_label.setText(
                    self.history_details_label.text()
                    + f"\n录音：{row.file_path}\n有效语音区间：{row.voiced_start_sec or '-'}–{row.voiced_end_sec or '-'}秒，长度 {voiced}。"
                    + f"\n声纹距离/阈值：{row.voice_distance or '-'} / {row.voice_threshold or '-'}；口令距离/阈值：{row.passphrase_distance or '-'} / {row.passphrase_threshold or '-'}。"
                    + f"\n预处理：{row.vad_mode}，{row.cepstral_normalization}；模型：{row.model_fingerprint[:12]}。"
                )

    def replay_history_record(self) -> None:
        index = self.full_history_table.currentRow()
        if not 0 <= index < len(self.history):
            QMessageBox.information(self, "请选择记录", "先选择一条现场验证记录。")
            return
        row = self.history[index]
        if not row.file_path or not row.model_fingerprint:
            QMessageBox.information(self, "旧记录无法复核", "这条旧记录没有保存录音路径或模型快照，原判定保持不变。")
            return
        try:
            path = project_path(row.file_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.audio_sha256:
                raise ValueError("原始录音内容已改变，不能作为同一次验证复核")
            model = load_user_model(self.history_path.parent / "diagnostics/models" / f"{row.model_fingerprint}.npz")
            if session_model_fingerprint(model) != row.model_fingerprint:
                raise ValueError("模型快照与原记录不一致")
            result = predict_with_session_model(path, model)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "复核失败", str(error))
            return
        QMessageBox.information(self, "录音复核结果", f"原判定：{row.decision}；复核判定：{result['decision']}。\n声纹分数：{row.voice_score} → {result['voice_score'] or '-'}\n口令模板分数：{row.passphrase_score} → {result['passphrase_score'] or '-'}\n{PHRASE_EXPLANATION}\n使用当时模型和当前算法重新分析，没有改写原记录。")

    def clear_history(self) -> None:
        try:
            save_history([], self.history_path)
        except Exception as exc:
            QMessageBox.warning(self, "历史清空失败", str(exc))
            return
        self.history.clear()
        self.refresh_history_table()

    def export_history(self) -> None:
        if not self.history:
            QMessageBox.information(self, "无历史", "当前没有可导出的验证历史。")
            return
        out_dir = PROJECT_ROOT / "results" / "gui_history"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["time", *[name for name in HistoryRow.__dataclass_fields__ if name != "time_text"]])
            writer.writeheader()
            for row in self.history:
                values = asdict(row)
                values["time"] = values.pop("time_text")
                writer.writerow(values)
        QMessageBox.information(self, "导出完成", f"已导出: {display_path(path)}")

    def start_session_task(self, kind, job) -> None:
        if self.audio_busy:
            return
        self.session_kind = kind
        self.session_result_applied = False
        self.session_cancel = Event()
        self.set_audio_busy(True)
        self.session_worker = SessionTaskWorker(job, self.session_cancel, self)
        self.session_thread = self.session_worker
        self.session_worker.completed.connect(self.on_session_completed)
        self.session_worker.failed.connect(self.on_session_failed)
        self.session_worker.stage.connect(self.on_session_stage)
        self.session_worker.progress.connect(self.on_session_progress)
        self.session_thread.finished.connect(self.on_session_thread_finished)
        self.session_thread.finished.connect(self.session_thread.deleteLater)
        self.session_thread.start()

    def on_session_stage(self, message):
        if self.session_cancel.is_set():
            return
        self.status_label.setText(message)
        if self.session_kind in {"enrollment", "registration"}:
            self.enroll_status_label.setText(message)
        elif self.session_kind == "microphone":
            self.mic_label.setText(message)
        else:
            self.temp_verify_result_label.setText("正在录音……" if "正在录音" in message else "正在验证……")
            self.temp_verify_reason_label.setText(message)

    def on_session_progress(self, fraction, peak):
        if self.session_cancel.is_set():
            return
        progress = self.enroll_capture_progress if self.session_kind == "enrollment" else self.verify_capture_progress
        level = self.enroll_level_label if self.session_kind == "enrollment" else self.verify_level_label
        progress.setValue(round(100 * fraction))
        level.setText(f"录音电平: {min(100, round(100 * peak))}%" + ("，音量过大" if peak >= .99 else ""))
        if fraction == 0:
            self.on_session_stage("麦克风已启动，正在录音，请说“芝麻开门”。")

    def cancel_session_task(self):
        if self.session_thread is None or self.session_result_applied:
            return
        self.session_cancel.set()
        self.verify_cancel_button.setEnabled(False)
        self.enroll_cancel_button.setEnabled(False)
        self.on_session_cancelled()

    def on_session_cancelled(self):
        message = "正在取消，等待当前步骤结束；本次不会保存结果。"
        if self.session_kind in {"enrollment", "registration"}:
            self.enroll_status_label.setText(message)
        else:
            self.temp_verify_result_label.setText("本次已取消")
            self.temp_verify_reason_label.setText(message)

    def on_session_thread_finished(self):
        self.session_thread.wait()
        self.session_thread = None
        self.session_worker = None
        if self.session_cancel.is_set():
            if self.session_kind in {"enrollment", "registration"}:
                self.enroll_status_label.setText("本次已取消，已有注册样本和用户保留。")
            elif self.session_kind == "microphone":
                self.mic_label.setText("麦克风: 测试已取消")
            else:
                self.temp_verify_reason_label.setText("本次已取消，没有新增验证记录。")
        self.set_audio_busy(False)
        self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")
        if self.close_requested:
            QTimer.singleShot(0, self.close)

    def on_session_failed(self, message):
        if self.session_cancel.is_set() or self.close_requested:
            return
        if self.session_kind in {"enrollment", "registration"}:
            self.enroll_status_label.setText("本次未完成，已有录音保留，可以重试。")
        elif self.session_kind == "microphone":
            self.mic_label.setText("麦克风: 测试失败")
        else:
            self.temp_verify_result_label.setText("本次录音失败")
            self.temp_verify_reason_label.setText("本次没有产生验证结果，请检查麦克风后重试。")
        QMessageBox.critical(self, "操作未完成", message)

    def on_session_completed(self, payload):
        if self.session_cancel.is_set() or self.close_requested:
            return
        self.session_result_applied = True
        self.verify_cancel_button.setEnabled(False)
        self.enroll_cancel_button.setEnabled(False)
        try:
            if self.session_kind == "enrollment":
                take, features, display = payload
                self.enrollment_takes.append(take)
                self.enrollment_features.append(features)
                self.enroll_waveform_canvas.plot_audio(display[0], display[1])
                self.enroll_mfcc_canvas.plot_features(display[2])
                self.enroll_tabs.setCurrentIndex(1)
                self.enroll_capture_progress.setValue(100)
                self.refresh_enrollment_list()
                count = self.accepted_enrollment_count()
                self.enroll_status_label.setText(f"第{len(self.enrollment_takes)}次：{take.reason}。" +
                    (f"已达到{ENROLLMENT_REQUIRED}条，可以完成注册。" if count >= ENROLLMENT_REQUIRED
                     else f"还需要{ENROLLMENT_REQUIRED - count}条合格录音。"))
            elif self.session_kind == "verification":
                result, latency, display = payload
                self.verify_waveform_canvas.plot_audio(display[0], display[1])
                self.verify_mfcc_canvas.plot_features(display[2])
                self.verify_capture_progress.setValue(100)
                self.show_user_verification(result, latency)
            elif self.session_kind == "registration":
                self.finish_registration(payload)
            elif self.session_kind == "microphone":
                self.mic_label.setText(f"麦克风: 可用 峰值 {payload:.3f}")
        except Exception as error:
            self.on_session_failed(str(error))

    def closeEvent(self, event):
        if self.session_thread is not None or self.predict_thread is not None:
            self.close_requested = True
            self.cancel_session_task()
            event.ignore()
            return
        event.accept()

    def record_enrollment_take(self) -> None:
        if self.audio_busy or self.accepted_enrollment_count() >= ENROLLMENT_REQUIRED:
            return
        if self.model is None:
            QMessageBox.warning(self, "模型未加载", "模型未加载，无法提取注册特征。")
            return
        duration, rate, model = self.enroll_duration_spin.value(), self.sample_rate_input.value(), self.model
        self.enroll_capture_progress.setValue(0)
        self.enroll_status_label.setText(f"正在启动麦克风，录音时长{duration:.1f}秒，请等候录音提示。")
        def job(worker):
            path = worker.record("enroll", duration, rate)
            worker.stage.emit("录音已结束，正在分析注册质量。")
            take, features = assess_enrollment_take(path, model)
            worker.checkpoint()
            return take, features, prepare_audio_display(path)
        self.start_session_task("enrollment", job)

    def refresh_enrollment_list(self) -> None:
        self.enroll_take_list.clear()
        for index, take in enumerate(self.enrollment_takes, start=1):
            status = "合格" if take.accepted else "不合格"
            self.enroll_take_list.addItem(
                f"第 {index} 次｜{status}｜{take.duration_sec:.1f} 秒"
                + (f"｜{take.reason}" if take.reason != "合格" else "")
            )
        if self.enrollment_takes:
            self.enroll_take_list.setCurrentRow(len(self.enrollment_takes) - 1)
        self.update_enrollment_actions()

    def show_enrollment_quality(self, index: int) -> None:
        if not 0 <= index < len(self.enrollment_takes):
            self.enroll_quality_label.setText("录音后可选择样本，查看有效声音与重录建议。")
            self.enroll_remove_button.setEnabled(False)
            return
        take = self.enrollment_takes[index]
        metrics = take.diagnostics
        text = f"第{index + 1}条：{take.reason}"
        if metrics:
            text += (f"\n有效活动约{metrics['sustained_seconds']:.2f}秒，近满幅采样{metrics['clip_fraction']:.1%}，"
                f"孤立短促声音{metrics['isolated_bursts']}处。能量检测不能判断口令内容。")
        self.enroll_quality_label.setText(text)
        self.enroll_remove_button.setEnabled(not self.audio_busy)

    def remove_enrollment_take(self) -> None:
        if self.audio_busy:
            return
        index = self.enroll_take_list.currentRow()
        if 0 <= index < len(self.enrollment_takes):
            self.enrollment_takes.pop(index)
            self.enrollment_features.pop(index)
            self.refresh_enrollment_list()
            self.enroll_status_label.setText("已从本次注册移除所选录音，原文件保留。请重录一条完整口令。")

    def accepted_enrollment_count(self) -> int:
        return sum(1 for take in self.enrollment_takes if take.accepted)

    def update_enrollment_actions(self) -> None:
        accepted_count = self.accepted_enrollment_count()
        self.enroll_build_button.setEnabled(
            not self.audio_busy and accepted_count >= ENROLLMENT_REQUIRED and bool(self.enroll_name_input.text().strip())
        )
        self.enroll_record_button.setEnabled(
            not self.audio_busy and self.model is not None and accepted_count < ENROLLMENT_REQUIRED
        )
        self.temp_verify_button.setEnabled(not self.audio_busy and self.session_model is not None)
        self.delete_user_button.setEnabled(not self.audio_busy and self.session_model is not None)
        self.enroll_remove_button.setEnabled(not self.audio_busy and self.enroll_take_list.currentRow() >= 0)
        self.enroll_progress.setValue(min(accepted_count, ENROLLMENT_REQUIRED))
        self.enroll_record_button.setText(
            f"已录满{ENROLLMENT_REQUIRED}条" if accepted_count >= ENROLLMENT_REQUIRED
            else f"录制第 {accepted_count + 1} 条语音"
        )

    def set_audio_busy(self, busy: bool) -> None:
        self.audio_busy = busy
        for widget in (
            self.saved_user_combo, self.go_register_button, self.enroll_name_input,
            self.enroll_duration_spin, self.verify_duration_spin, self.enroll_reset_button,
            self.sample_rate_input, self.reload_model_button, self.mic_test_button,
            self.delete_user_data_checkbox,
        ):
            widget.setEnabled(not busy)
        self.verify_cancel_button.setEnabled(busy)
        self.enroll_cancel_button.setEnabled(busy)
        for button in self.nav_buttons:
            button.setEnabled(not busy)
        self.update_enrollment_actions()

    def reset_temporary_enrollment(self) -> None:
        self.enrollment_takes.clear()
        self.enrollment_features.clear()
        self.enroll_take_list.clear()
        self.enroll_waveform_canvas.plot_empty()
        self.enroll_mfcc_canvas.plot_empty()
        self.enroll_status_label.setText(f"录音状态: 已重新开始，请录入 {ENROLLMENT_REQUIRED} 条合格注册语音。")
        self.update_enrollment_actions()

    def reset_user_verdict(self) -> None:
        self.temp_verify_result_label.setText("等待录音")
        self.temp_verify_result_label.setObjectName("VerdictIdle")
        self.temp_verify_result_label.style().unpolish(self.temp_verify_result_label)
        self.temp_verify_result_label.style().polish(self.temp_verify_result_label)
        self.temp_verify_reason_label.setText(
            f"请用 {self.session_model.speaker_id} 的声音说“芝麻开门”。"
            if self.session_model is not None else "先注册用户，再开始验证。"
        )
        self.temp_verify_score_label.setText("分数: -")
        self.verify_waveform_canvas.plot_empty()
        self.verify_mfcc_canvas.plot_empty()

    def update_enrollment_plots(self, wav_path: Path) -> None:
        self.enroll_waveform_canvas.plot_wav(wav_path)
        self.enroll_mfcc_canvas.plot_wav(wav_path)
        self.enroll_tabs.setCurrentIndex(1)

    def show_user_model(self) -> None:
        model = self.session_model
        self.temp_model_label.setText(f"当前验证用户：{model.speaker_id}")
        self.temp_verify_button.setText("开始录音验证")
        self.reset_user_verdict()
        self.update_enrollment_actions()

    def refresh_saved_users(self, selected_path: Path | None = None) -> None:
        previous = str(selected_path) if selected_path else self.saved_user_combo.currentData()
        self.saved_user_combo.blockSignals(True)
        self.saved_user_combo.clear()
        errors = []
        for path in sorted(self.user_model_dir.glob("*.npz")):
            try:
                model = load_user_model(path)
                self.saved_user_combo.addItem(model.speaker_id, str(path))
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        if self.saved_user_combo.count() == 0:
            self.saved_user_combo.addItem("还没有注册用户", None)
        index = self.saved_user_combo.findData(previous)
        self.saved_user_combo.setCurrentIndex(max(index, 0))
        self.saved_user_combo.blockSignals(False)
        self.verify_empty_label.setVisible(self.saved_user_combo.currentData() is None)
        self.load_saved_user()
        if errors:
            QMessageBox.warning(self, "部分用户模型无法读取", "\n".join(errors))

    def load_saved_user(self) -> None:
        if self.session_thread is not None:
            return
        self.session_model = None
        self.active_user_path = None
        self.temp_model_label.setText("尚未选择用户")
        self.reset_user_verdict()
        self.update_enrollment_actions()
        path_text = self.saved_user_combo.currentData()
        if not path_text:
            return
        try:
            model = load_user_model(Path(path_text))
        except Exception as exc:
            QMessageBox.critical(self, "用户加载失败", str(exc))
            return
        self.session_model = model
        self.active_user_path = Path(path_text)
        self.delete_user_data_checkbox.setChecked(False)
        self.show_user_model()

    def delete_saved_user(self) -> None:
        if self.audio_busy or self.session_model is None or self.active_user_path is None:
            return
        path, name = self.active_user_path, self.session_model.speaker_id
        plan = None
        if self.delete_user_data_checkbox.isChecked():
            try:
                plan = build_user_cleanup(name, path, self.user_model_dir, self.history_path, PROJECT_ROOT)
            except (OSError, ValueError, KeyError) as error:
                QMessageBox.critical(self, "无法准备清理", str(error))
                return
        message = (f"确定清理用户“{name}”？\n{plan.summary()}" if plan else
            f"确定删除用户“{name}”的注册模型？\n原录音和验证记录会保留，删除后可重新注册同名用户。")
        if QMessageBox.question(self, "删除注册用户",
            message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            if plan:
                execute_user_cleanup(plan)
                self.history = load_history(self.history_path)
                self.refresh_history_table()
            else:
                path.unlink()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "删除未完成", f"{error}\n清理可能已执行一部分，请保留当前账户并重新检查；不会回填或伪造已删除文件。")
            return
        self.delete_user_data_checkbox.setChecked(False)
        self.refresh_saved_users()
        self.temp_verify_reason_label.setText(f"已删除用户“{name}”" + (f"及{len(plan.audio_files)}个关联录音，已清理关联记录和快照。" if plan else "的注册模型，录音和记录保留。"))

    def complete_registration(self) -> None:
        if self.audio_busy:
            return
        accepted_features = [
            features for take, features in zip(self.enrollment_takes, self.enrollment_features) if take.accepted
        ]
        if len(accepted_features) < ENROLLMENT_REQUIRED:
            QMessageBox.warning(self, "注册样本不足", f"至少需要 {ENROLLMENT_REQUIRED} 条合格注册样本。")
            return
        speaker_id = self.enroll_name_input.text().strip()
        if not speaker_id or speaker_id == "REJECT":
            QMessageBox.warning(self, "请填写用户名", "请输入用户名，且不要使用保留名称 REJECT。")
            return
        path = user_model_path(speaker_id, self.user_model_dir)
        if path.exists() and QMessageBox.question(
            self, "更新已有用户", f"用户 {speaker_id} 已存在。是否用本次录音更新注册？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.enroll_status_label.setText("正在完成注册，请稍候……")
        official_model = self.model
        takes = list(self.enrollment_takes)
        def job(worker):
            model = build_session_enrollment_model(speaker_id, official_model, accepted_features)
            worker.checkpoint()
            worker.stage.emit("模型已建立，正在检查注册样本一致性。")
            checks = enrollment_self_check(accepted_features, model)
            worker.checkpoint()
            samples, check_index = [], 0
            for index, take in enumerate(takes, start=1):
                sample = {**asdict(take), "path": display_path(take.path), "take_index": index,
                    "audio_sha256": hashlib.sha256(take.path.read_bytes()).hexdigest()}
                if take.accepted:
                    sample["self_check"] = checks[check_index]
                    check_index += 1
                samples.append(sample)
            report = {"speaker_id": speaker_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_fingerprint": session_model_fingerprint(model), "voice_threshold": model.voice_threshold,
                "passphrase_threshold": model.passphrase_threshold, "vad_mode": model.vad_mode,
                "cepstral_normalization": model.cepstral_normalization, "samples": samples,
                "limitation": "Training-recording self-check only; not independent verification or wrong-passphrase validation"}
            return model, checks, report, path
        self.start_session_task("registration", job)

    def finish_registration(self, payload):
        model, checks, report, path = payload
        report_dir = self.user_model_dir.parent / "diagnostics" / "enrollments"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{time.time_ns()}_{report['model_fingerprint'][:12]}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        save_user_model(model, path)
        self.refresh_saved_users(path)
        self.session_model = model
        self.active_user_path = path
        self.show_user_model()
        self.reset_temporary_enrollment()
        self.enroll_name_input.clear()
        self.stack.setCurrentIndex(0)
        passed = sum(check["accepted"] for check in checks)
        self.temp_verify_reason_label.setText(f"{model.speaker_id} 已注册并保存，样本自检{passed}/{len(checks)}通过。"
            "自检只检查注册样本，还需录音验证；质量记录已保存。")

    def verify_temporary_user_by_recording(self) -> None:
        if self.audio_busy:
            return
        if self.session_model is None:
            QMessageBox.warning(self, "用户模型未加载", "请先注册建立模型，或加载已保存用户。")
            return
        self.reset_user_verdict()
        duration, rate, model = self.verify_duration_spin.value(), self.sample_rate_input.value(), self.session_model
        self.verify_capture_progress.setValue(0)
        self.temp_verify_result_label.setText("正在启动麦克风……")
        self.temp_verify_reason_label.setText(f"录音时长{duration:.1f}秒，请等候录音提示后说“芝麻开门”。")
        def job(worker):
            path = worker.record("temp_verify", duration, rate)
            worker.stage.emit("录音已结束，正在核对声纹和口令。")
            start = time.perf_counter()
            result = predict_with_session_model(path, model)
            latency = (time.perf_counter() - start) * 1000
            worker.checkpoint()
            return result, latency, prepare_audio_display(path)
        self.start_session_task("verification", job)

    def verify_temporary_audio(self, wav_path: Path) -> None:
        try:
            start = time.perf_counter()
            result = predict_with_session_model(wav_path, self.session_model)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.verify_waveform_canvas.plot_wav(wav_path)
            self.verify_mfcc_canvas.plot_wav(wav_path)
        except Exception as exc:
            self.temp_verify_result_label.setText("验证结果: 验证失败")
            self.temp_verify_reason_label.setText("本次未得到有效判定，请重试。")
            self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")
            QMessageBox.critical(self, "用户验证失败", str(exc))
            return
        self.show_user_verification(result, latency_ms)

    def show_user_verification(self, result, latency_ms):
        decision = result.get("decision", "REJECT")
        if decision == "REJECT":
            self.temp_verify_result_label.setText("未通过")
            self.temp_verify_result_label.setObjectName("VerdictReject")
        else:
            self.temp_verify_result_label.setText(f"验证通过 · {decision}")
            self.temp_verify_result_label.setObjectName("VerdictAccept")
        self.temp_verify_result_label.style().unpolish(self.temp_verify_result_label)
        self.temp_verify_result_label.style().polish(self.temp_verify_result_label)
        voice_status = "通过" if result.get("voice_passed") == "True" else "未通过"
        passphrase_status = "通过" if result.get("passphrase_passed") == "True" else "未通过"
        self.temp_verify_reason_label.setText(
            ("录音未进入声纹和口令比对。" if result.get("rejection_stage") == "input" else f"声纹{voice_status}，口令模板匹配{passphrase_status}。")
            + (f"{result.get('reject_reason', '')}。请重新录音，保持自然音量并说完整口令。" if decision == "REJECT" else "本次验证已保存到“验证记录”。")
        )
        self.temp_verify_score_label.setText(
            "分数: "
            f"声纹 {format_float(result.get('voice_score', ''))} / 通过线 {result.get('voice_limit', '1.00')}, "
            f"口令模板 {format_float(result.get('passphrase_score', ''))} / 通过线 {result.get('passphrase_limit', '-')}, "
            + (f"融合 {format_float(result['fusion_score'])} / 通过线 {result['fusion_threshold']}，" if result.get("fusion_score") else "")
            + f"SNR {format_float(result.get('estimated_snr_db', ''))} dB，耗时 {latency_ms:.1f} ms。"
        )
        self.add_history_row(result, f"用户验证:{self.session_model.speaker_id}", latency_ms)
        self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")

    def reload_results_tables(self) -> None:
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(["算法方案", "全部判定正确", "正常用户通过", "错误口令拦截", "陌生人拦截"])
        try:
            with (PROJECT_ROOT / "results/tables/voice_lock_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                metrics = list(csv.DictReader(handle))
            with (PROJECT_ROOT / "results/tables/recommended_config.csv").open(encoding="utf-8-sig", newline="") as handle:
                recommended = next(csv.DictReader(handle))
            rows = []
            for stage, label in (("voice_only", "只验证声纹"), ("two_layer", "声纹 + 口令（原始方案）")):
                group = {row["split"]: row for row in metrics if row["stage"] == stage}
                rows.append([label] + [
                    f"{float(group[split]['accuracy']):.1%}（{group[split]['correct']}/{group[split]['total']}）"
                    for split in ("all", "test", "wrong_passphrase", "impostor")
                ])
            rows.append(["离线筛选候选方案"] + [
                f"{float(recommended[key]):.1%}"
                for key in ("fusion_all_accuracy", "registered_test_accuracy", "wrong_passphrase_reject_rate", "impostor_reject_rate")
            ])
            self.metrics_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, value in enumerate(row):
                    self.metrics_table.setItem(row_index, column, QTableWidgetItem(value))
            self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.experiment_note.setText(
                "怎么看：正常用户通过率反映能否顺利开锁；错误口令和陌生人拦截率反映拒绝能力。\n"
                f"候选方案在这批数据上误拒 {recommended['false_reject']} 条、误接受 {recommended['false_accept']} 条。"
                "样本很少，不能代表所有场景；候选成绩也不代表当前注册用户的现场表现。\n"
                "这里只读取已有实验结果，点击刷新不会训练模型。"
            )
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            self.metrics_table.setRowCount(0)
            self.experiment_note.setText(f"暂时无法读取已有实验结果：{exc}")
        self.calibration_table.setColumnCount(6)
        self.calibration_table.setHorizontalHeaderLabels(["校准方案", "总体正确", "正常通过", "误拒", "误接受", "错认"])
        labels = {
            "converged_in_sample": "修复训练＋原阈值", "loo_voice": "留一校准声纹",
            "loo_two_layer": "声纹/口令均校准", "loo_fusion": "绝对距离＋融合",
            "loo_voice_fusion": "口令保守＋融合", "cohort_two_layer": "相对距离＋双门槛",
            "cohort_fusion": "相对距离＋融合",
            "baseline": "本轮统一校准对照", "dtw_run2": "DTW连续步数≤2", "dtw_run3": "DTW连续步数≤3",
            "dtw_run5": "DTW连续步数≤5", "dtw_medoid": "DTW代表模板", "dtw_single_first": "DTW首条单模板",
            "dtw_mfcc_delta": "口令MFCC＋动态特征", "frame_vad": "帧级能量/频谱VAD",
            "cmn": "倒谱均值归一化", "cmvn": "倒谱均值方差归一化",
            "voice_mfcc_delta": "声纹MFCC＋动态特征", "voice_all": "声纹全部31维特征",
            "fusion_two_layer": "融合对照：双门槛", "fusion_strict": "融合：保留双门槛",
            "fusion_voice_complement": "融合：声纹有限互补", "fusion_both_complement": "融合：双方有限互补",
        }
        paths = [PROJECT_ROOT / "results" / path for path in (
            "calibration_experiments/comparison.csv", "followup_experiments/comparison.csv", "followup_experiments/fusion_comparison.csv",
        )]
        self.calibration_table.setRowCount(0)
        self.calibration_note.setText("尚未生成本轮算法校准对比结果。")
        if any(path.exists() for path in paths):
            try:
                rows = []
                for path in paths:
                    if path.exists():
                        with path.open(encoding="utf-8-sig", newline="") as handle:
                            rows.extend(csv.DictReader(handle))
                self.calibration_table.setRowCount(len(rows))
                for index, row in enumerate(rows):
                    values = [labels[row["variant"]], f"{float(row['fusion_all_accuracy']):.1%}", f"{float(row['registered_test_accuracy']):.1%}", row["false_reject"], row["false_accept"], row["wrong_speaker"]]
                    for column, value in enumerate(values):
                        self.calibration_table.setItem(index, column, QTableWidgetItem(value))
                self.calibration_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                self.calibration_note.setText("同一批曾用于选型的63条开发评估；参数只使用36条训练录音拟合。候选不自动部署；VAD、归一化或融合均未证明安全提升，这些成绩不代表现场新用户表现。")
            except (OSError, ValueError, KeyError) as exc:
                self.calibration_table.setRowCount(0)
                self.calibration_note.setText(f"校准结果读取失败：{exc}")

    def reload_model_from_settings(self) -> None:
        if self.audio_busy:
            return
        try:
            model = load_model(self.model_path_input.text(), self.features_path_input.text())
            self.reset_temporary_enrollment()
            self.model = model
            self.show_algorithm_config()
            self.status_label.setText("模型: 已重新加载")
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def test_microphone(self) -> None:
        if self.audio_busy:
            return
        rate = self.sample_rate_input.value()
        self.mic_label.setText("麦克风: 测试中")
        def job(worker):
            path = worker.record("mic_test", 1.0, rate)
            audio, _ = sf.read(path)
            return float(np.max(np.abs(audio)))
        self.start_session_task("microphone", job)


def gui_smoke_test() -> int:
    model = load_model()
    sample = DEMO_SAMPLES["正常通过 S01"]
    result = predict_one(project_path(sample["wav"]), model=model, metadata=sample)
    train_paths = [
        project_path("data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_001.wav"),
        project_path("data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_002.wav"),
        project_path("data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_003.wav"),
    ]
    enrollment_features = [assess_enrollment_take(path, model)[1] for path in train_paths]
    session_model = build_session_enrollment_model("SmokeS01", model, enrollment_features)
    session_result = predict_with_session_model(project_path(sample["wav"]), session_model)
    with tempfile.TemporaryDirectory(prefix="srdp_app_smoke_") as directory:
        root = Path(directory)
        saved_path = user_model_path(session_model.speaker_id, root / "users")
        save_user_model(session_model, saved_path)
        restored_model = load_user_model(saved_path)
        restored_result = predict_with_session_model(project_path(sample["wav"]), restored_model)
        assert restored_result == session_result, "user prediction changed after save/load"
        np.testing.assert_array_equal(restored_model.zscore_mean, session_model.zscore_mean)
        np.testing.assert_array_equal(restored_model.zscore_std, session_model.zscore_std)
        history = HistoryRow("2026-01-01 12:00:00", "用户验证:SmokeS01", **{
            key: session_result.get(key, "")
            for key in ("decision", "pred_speaker", "voice_score", "passphrase_score", "fusion_score", "reject_reason")
        }, latency_ms="0.0")
        save_history([history], root / "history.csv")
        assert load_history(root / "history.csv") == [history], "history did not survive reload"
        save_history([], root / "history.csv")
        assert load_history(root / "history.csv") == [], "cleared history returned after reload"
    print(
        "gui smoke "
        f"official_decision={result.get('decision')} "
        f"session_decision={session_result.get('decision')} "
        "user_save_load=passed history_save_load=passed"
    )
    return 0 if result.get("decision") == "S01" and session_result.get("decision") == "SmokeS01" else 1


def main() -> int:
    if "--smoke" in sys.argv:
        return gui_smoke_test()
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    window = SrdpVoiceLockWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
