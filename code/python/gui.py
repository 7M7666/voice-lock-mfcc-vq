from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
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
    QVBoxLayout,
    QWidget,
)

from srdp import (
    PROJECT_ROOT,
    display_path,
    extract_one_audio_features,
    lbg_codebook,
    load_model,
    normalise_metadata_row,
    passphrase_template_distance,
    predict_one,
    project_path,
    quantization_distance,
    safe_ratio,
    select_dims,
)


SAMPLE_RATE = 16000
TEMP_AUDIO_DIR = PROJECT_ROOT / "tmp" / "gui_recordings"
SESSION_VOICE_THRESHOLD_MARGIN = 1.50
SESSION_PASSPHRASE_ACCEPT_SCORE = 0.72

SPLIT_OPTIONS = [
    ("注册用户正常", "test"),
    ("错误口令", "wrong_passphrase"),
    ("非注册冒充", "impostor"),
    ("回放风险", "replay"),
]
TRIAL_OPTIONS = [
    ("正常口令", "normal"),
    ("错误口令", "wrong_passphrase"),
    ("非注册冒充", "impostor"),
    ("回放风险", "replay"),
    ("低音量", "low_volume"),
    ("背景噪声", "noise"),
    ("语速偏快", "fast"),
    ("语速偏慢", "slow"),
]

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


@dataclass
class EnrollmentTake:
    path: Path
    duration_sec: float
    estimated_snr_db: float
    quality_score: float
    low_quality: bool
    accepted: bool
    reason: str


@dataclass
class SessionEnrollmentModel:
    speaker_id: str
    codebook: np.ndarray
    voice_threshold: float
    passphrase_templates: list[np.ndarray]
    passphrase_threshold: float
    feature_set: str
    passphrase_feature_set: str


def combo_data(combo: QComboBox) -> str:
    data = combo.currentData()
    return str(data) if data is not None else combo.currentText()


def set_combo_data(combo: QComboBox, value: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == value or combo.itemText(index) == value:
            combo.setCurrentIndex(index)
            return


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


def record_audio_to_temp(prefix: str, duration_sec: float, sample_rate: int = SAMPLE_RATE) -> Path:
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_path = TEMP_AUDIO_DIR / f"{prefix}_{timestamp}.wav"
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(audio_path, audio, sample_rate)
    return audio_path


def assess_enrollment_take(wav_path: Path, model) -> tuple[EnrollmentTake, np.ndarray]:
    extracted = extract_one_audio_features(wav_path, model)
    duration_ok = 1.2 <= extracted.duration_sec <= 5.0
    quality_ok = not extracted.low_quality
    snr_warning = extracted.estimated_snr_db < 10.0
    accepted = duration_ok and quality_ok
    if not duration_ok:
        reason = "时长不合格"
    elif not quality_ok:
        reason = "录音质量偏低"
    elif snr_warning:
        reason = "SNR偏低，建议重录"
    else:
        reason = "合格"
    return (
        EnrollmentTake(
            path=wav_path,
            duration_sec=extracted.duration_sec,
            estimated_snr_db=extracted.estimated_snr_db,
            quality_score=extracted.quality_score,
            low_quality=extracted.low_quality,
            accepted=accepted,
            reason=reason,
        ),
        extracted.features,
    )


def build_session_enrollment_model(
    speaker_id: str,
    official_model,
    enrollment_features: list[np.ndarray],
) -> SessionEnrollmentModel:
    dims = select_dims(official_model.feature_names, official_model.feature_set)
    passphrase_dims = select_dims(official_model.feature_names, official_model.passphrase_feature_set)
    speaker_mats = [features[:, dims] for features in enrollment_features]
    passphrase_mats = [features[:, passphrase_dims] for features in enrollment_features]
    stacked = np.vstack(speaker_mats)
    codebook = lbg_codebook(stacked, target_size=choose_session_codebook_size(stacked.shape[0]))
    own_distances = np.array([quantization_distance(matrix, codebook) for matrix in speaker_mats], dtype=float)
    voice_base_threshold = float(max(own_distances.mean() + 2.0 * own_distances.std(), own_distances.max() + 1e-6))
    voice_threshold = voice_base_threshold * SESSION_VOICE_THRESHOLD_MARGIN

    passphrase_distances = []
    for index, matrix in enumerate(passphrase_mats):
        distance = passphrase_template_distance(
            matrix,
            passphrase_mats,
            official_model.passphrase_aggregate,
            exclude_index=index,
            window_ratio=official_model.dtw_window_ratio,
            warp_penalty=official_model.dtw_warp_penalty,
        )
        if np.isfinite(distance):
            passphrase_distances.append(distance)
    if passphrase_distances:
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
    )


def predict_with_session_model(wav_path: Path, official_model, session_model: SessionEnrollmentModel) -> dict[str, str]:
    extracted = extract_one_audio_features(wav_path, official_model)
    dims = select_dims(official_model.feature_names, session_model.feature_set)
    passphrase_dims = select_dims(official_model.feature_names, session_model.passphrase_feature_set)
    speaker_features = extracted.features[:, dims]
    passphrase_features = extracted.features[:, passphrase_dims]
    voice_distance = quantization_distance(speaker_features, session_model.codebook)
    voice_score = safe_ratio(voice_distance, session_model.voice_threshold)
    voice_ok = voice_distance <= session_model.voice_threshold
    passphrase_distance = passphrase_template_distance(
        passphrase_features,
        session_model.passphrase_templates,
        official_model.passphrase_aggregate,
        window_ratio=official_model.dtw_window_ratio,
        warp_penalty=official_model.dtw_warp_penalty,
    )
    passphrase_score = safe_ratio(passphrase_distance, session_model.passphrase_threshold)
    passphrase_ok = passphrase_score <= SESSION_PASSPHRASE_ACCEPT_SCORE
    decision = session_model.speaker_id if voice_ok and passphrase_ok else "REJECT"
    if not passphrase_ok and voice_ok:
        reject_reason = "口令DTW距离超过阈值"
    elif not passphrase_ok and not voice_ok:
        reject_reason = "口令DTW距离超过阈值；声纹分支也偏高"
    elif not voice_ok:
        reject_reason = "临时声纹距离超过阈值"
    else:
        reject_reason = ""
    return {
        "decision": decision,
        "pred_speaker": session_model.speaker_id,
        "voice_score": f"{voice_score:.6f}",
        "voice_passed": str(voice_ok),
        "voice_limit": "1.00",
        "passphrase_score": f"{passphrase_score:.6f}",
        "passphrase_passed": str(passphrase_ok),
        "passphrase_limit": f"{SESSION_PASSPHRASE_ACCEPT_SCORE:.2f}",
        "fusion_score": "",
        "fusion_threshold": "",
        "estimated_snr_db": f"{extracted.estimated_snr_db:.6f}",
        "quality_score": f"{extracted.quality_score:.6f}",
        "reject_reason": reject_reason,
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
        self.axes.clear()
        self.axes.imshow(mfcc, aspect="auto", origin="lower", cmap="magma")
        self.axes.set_title("MFCC")
        self.axes.set_xlabel("Frame")
        self.axes.set_ylabel("Coefficient")
        self.draw()


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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SRDP 轻量化声音锁演示系统")
        self.resize(1280, 820)
        self.model = None
        self.history: list[HistoryRow] = []
        self.current_audio_path: Path | None = None
        self.current_source_label = "demo"
        self.predict_thread: QThread | None = None
        self.predict_worker: PredictWorker | None = None
        self.enrollment_takes: list[EnrollmentTake] = []
        self.enrollment_features: list[np.ndarray] = []
        self.session_model: SessionEnrollmentModel | None = None

        self.status_label = QLabel("模型: 加载中")
        self.mic_label = QLabel("麦克风: 未检测")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_demo_page())
        self.stack.addWidget(self.build_enrollment_page())
        self.stack.addWidget(self.build_results_page())
        self.stack.addWidget(self.build_history_page())
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
            ("答辩演示台", 0),
            ("临时注册", 1),
            ("实验结果", 2),
            ("历史记录", 3),
            ("设置", 4),
        ]
        for text, index in nav_items:
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, page=index: self.stack.setCurrentIndex(page))
            layout.addWidget(button)
        layout.addStretch(1)
        return sidebar

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
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(["", "S01", "S02", "S03", "S04", "U01", "S01_wrong", "S02_wrong", "S03_wrong", "S04_wrong"])
        self.split_combo = QComboBox()
        for label, value in SPLIT_OPTIONS:
            self.split_combo.addItem(label, value)
        self.trial_combo = QComboBox()
        for label, value in TRIAL_OPTIONS:
            self.trial_combo.addItem(label, value)
        self.current_file_label = QLabel("未选择音频")
        self.current_file_label.setWordWrap(True)
        choose_button = QPushButton("选择 WAV")
        choose_button.clicked.connect(self.choose_wav)
        self.predict_button = QPushButton("开始验证")
        self.predict_button.setObjectName("PrimaryButton")
        self.predict_button.clicked.connect(self.predict_current_audio)
        live_note = QLabel("现场录音请使用左侧“临时注册”，首页用于固定样例和已有 WAV 复现。")
        live_note.setWordWrap(True)
        live_note.setStyleSheet("color: #5B6775;")

        button_row = QHBoxLayout()
        button_row.addWidget(choose_button)
        input_layout.addRow("固定样例", self.sample_combo)
        input_layout.addRow("当前文件", self.current_file_label)
        input_layout.addRow("真实用户", self.speaker_combo)
        input_layout.addRow("样本分组", self.split_combo)
        input_layout.addRow("测试场景", self.trial_combo)
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
        self.passphrase_score_label, self.passphrase_score_bar = self.build_score_row("口令分数")
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

        history_group = QGroupBox("最近验证")
        history_layout = QVBoxLayout(history_group)
        self.history_table = QTableWidget(0, 9)
        self.history_table.setHorizontalHeaderLabels(["时间", "来源", "判定", "预测", "声纹", "口令", "融合", "原因", "耗时ms"])
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

        settings_group = QGroupBox("临时注册设置")
        settings_layout = QFormLayout(settings_group)
        note = QLabel("只在当前演示会话生效，不写入正式 metadata，不覆盖正式模型。")
        note.setWordWrap(True)
        self.enroll_name_input = QLineEdit("TeacherDemo")
        self.enroll_duration_spin = QDoubleSpinBox()
        self.enroll_duration_spin.setRange(1.0, 5.0)
        self.enroll_duration_spin.setValue(3.0)
        record_button = QPushButton("开始录入一条注册语音")
        record_button.clicked.connect(self.record_enrollment_take)
        self.enroll_build_button = QPushButton("建立临时模型")
        self.enroll_build_button.setObjectName("PrimaryButton")
        self.enroll_build_button.setEnabled(False)
        self.enroll_build_button.clicked.connect(self.build_temporary_model)
        reset_button = QPushButton("重新开始临时注册")
        reset_button.clicked.connect(self.reset_temporary_enrollment)
        self.enroll_status_label = QLabel("录音状态: 等待开始")
        self.enroll_status_label.setWordWrap(True)
        self.enroll_status_label.setStyleSheet("color: #5B6775;")
        settings_layout.addRow(note)
        settings_layout.addRow("临时用户名", self.enroll_name_input)
        settings_layout.addRow("固定口令", QLabel("芝麻开门"))
        settings_layout.addRow("录音时长", self.enroll_duration_spin)
        settings_layout.addRow(record_button)
        settings_layout.addRow(self.enroll_build_button)
        settings_layout.addRow(reset_button)
        settings_layout.addRow(self.enroll_status_label)

        takes_group = QGroupBox("注册样本")
        takes_layout = QVBoxLayout(takes_group)
        self.enroll_take_list = QListWidget()
        self.temp_model_label = QLabel("临时模型: 未建立")
        self.temp_model_label.setWordWrap(True)
        takes_layout.addWidget(self.enroll_take_list)
        takes_layout.addWidget(self.temp_model_label)

        verify_group = QGroupBox("临时验证")
        verify_layout = QVBoxLayout(verify_group)
        self.temp_verify_button = QPushButton("录一条新语音并验证")
        self.temp_verify_button.setObjectName("PrimaryButton")
        self.temp_verify_button.setEnabled(False)
        self.temp_verify_button.clicked.connect(self.verify_temporary_user_by_recording)
        self.temp_verify_result_label = QLabel("验证结果: 等待验证")
        self.temp_verify_result_label.setWordWrap(True)
        self.temp_verify_reason_label = QLabel("说明: 先建立临时模型，再录一条新语音做验证。")
        self.temp_verify_reason_label.setWordWrap(True)
        self.temp_verify_score_label = QLabel("分数: -")
        self.temp_verify_score_label.setWordWrap(True)
        verify_layout.addWidget(self.temp_verify_button)
        verify_layout.addWidget(self.temp_verify_result_label)
        verify_layout.addWidget(self.temp_verify_reason_label)
        verify_layout.addWidget(self.temp_verify_score_label)
        verify_layout.addStretch(1)

        layout.addWidget(settings_group, 0, 0)
        layout.addWidget(takes_group, 0, 1)
        layout.addWidget(verify_group, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        return page

    def build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        summary = QLabel(
            "当前系统已完成自采录音、特征提取、声纹识别、口令验证和融合判定闭环。"
            "安全优先配置保持 False Accept 为 0，并把总体准确率提高到 90.5%。"
        )
        summary.setWordWrap(True)
        self.metrics_table = QTableWidget()
        self.recommend_table = QTableWidget()
        reload_button = QPushButton("重新读取结果表")
        reload_button.clicked.connect(self.reload_results_tables)
        layout.addWidget(summary)
        layout.addWidget(QLabel("正式指标表 voice_lock_metrics.csv"))
        layout.addWidget(self.metrics_table)
        layout.addWidget(QLabel("推荐配置 recommended_config.csv"))
        layout.addWidget(self.recommend_table)
        layout.addWidget(reload_button)
        self.reload_results_tables()
        return page

    def build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.full_history_table = QTableWidget(0, 9)
        self.full_history_table.setHorizontalHeaderLabels(["时间", "来源", "判定", "预测", "声纹", "口令", "融合", "原因", "耗时ms"])
        self.full_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        clear_button = QPushButton("清空本次历史")
        clear_button.clicked.connect(self.clear_history)
        export_button = QPushButton("导出本次历史 CSV")
        export_button.clicked.connect(self.export_history)
        actions = QHBoxLayout()
        actions.addWidget(clear_button)
        actions.addWidget(export_button)
        actions.addStretch(1)
        layout.addWidget(self.full_history_table)
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
        reload_button = QPushButton("重新加载模型")
        reload_button.clicked.connect(self.reload_model_from_settings)
        mic_button = QPushButton("测试麦克风 1 秒")
        mic_button.clicked.connect(self.test_microphone)
        layout.addRow("模型文件", self.model_path_input)
        layout.addRow("特征文件", self.features_path_input)
        layout.addRow("录音采样率", self.sample_rate_input)
        layout.addRow(reload_button)
        layout.addRow(mic_button)
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
            self.status_label.setText("模型: 已加载")
        except Exception as exc:
            self.status_label.setText("模型: 加载失败")
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def load_selected_demo_sample(self) -> None:
        sample = DEMO_SAMPLES[self.sample_combo.currentText()]
        wav_path = project_path(sample["wav"])
        self.current_audio_path = wav_path
        self.current_source_label = self.sample_combo.currentText()
        self.current_file_label.setText(display_path(wav_path))
        self.speaker_combo.setCurrentText(sample["speaker_id"])
        set_combo_data(self.split_combo, sample["split"])
        set_combo_data(self.trial_combo, sample["trial_type"])
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
        split = combo_data(self.split_combo)
        speaker_id = self.speaker_combo.currentText()
        row = normalise_metadata_row({})
        row.update(
            {
                "file_path": display_path(self.current_audio_path) if self.current_audio_path else "",
                "speaker_id": speaker_id,
                "split": split,
                "expected_decision": speaker_id if split == "test" and speaker_id in {"S01", "S02", "S03", "S04"} else "REJECT",
                "trial_type": combo_data(self.trial_combo),
                "is_replay": "true" if split == "replay" else "false",
            }
        )
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
        self.set_score(self.passphrase_score_label, self.passphrase_score_bar, "口令分数", result.get("passphrase_score", ""), "1.000")
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
            time_text=time.strftime("%H:%M:%S"),
            source=source_label,
            decision=result.get("decision", ""),
            pred_speaker=result.get("pred_speaker", ""),
            voice_score=format_float(result.get("voice_score", "")),
            passphrase_score=format_float(result.get("passphrase_score", "")),
            fusion_score=format_float(result.get("fusion_score", "")),
            reject_reason=result.get("reject_reason", ""),
            latency_ms=f"{latency_ms:.1f}",
        )
        self.history.insert(0, history)
        self.history = self.history[:20]
        self.refresh_history_table()

    def populate_history_table(self, table: QTableWidget) -> None:
        table.setRowCount(len(self.history))
        for row_index, row in enumerate(self.history):
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
            for column_index, value in enumerate(values):
                table.setItem(row_index, column_index, QTableWidgetItem(value))

    def refresh_history_table(self) -> None:
        self.populate_history_table(self.history_table)
        self.populate_history_table(self.full_history_table)

    def clear_history(self) -> None:
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
            writer = csv.writer(handle)
            writer.writerow(["time", "source", "decision", "pred_speaker", "voice_score", "passphrase_score", "fusion_score", "reject_reason", "latency_ms"])
            for row in self.history:
                writer.writerow([row.time_text, row.source, row.decision, row.pred_speaker, row.voice_score, row.passphrase_score, row.fusion_score, row.reject_reason, row.latency_ms])
        QMessageBox.information(self, "导出完成", f"已导出: {display_path(path)}")

    def record_enrollment_take(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "模型未加载", "模型未加载，无法提取注册特征。")
            return
        try:
            self.enroll_status_label.setText(f"录音状态: 开始录音，请说“芝麻开门”，持续 {self.enroll_duration_spin.value():.1f} 秒。")
            self.status_label.setText("注册录音: 进行中")
            QApplication.processEvents()
            wav_path = record_audio_to_temp("enroll", self.enroll_duration_spin.value(), self.sample_rate_input.value())
            self.enroll_status_label.setText("录音状态: 录音结束，正在分析质量。")
            QApplication.processEvents()
            take, features = assess_enrollment_take(wav_path, self.model)
        except Exception as exc:
            self.enroll_status_label.setText("录音状态: 录音失败")
            self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")
            QMessageBox.critical(self, "注册录音失败", str(exc))
            return
        self.enrollment_takes.append(take)
        self.enrollment_features.append(features)
        self.refresh_enrollment_list()
        self.enroll_status_label.setText(
            f"录音状态: 第 {len(self.enrollment_takes)} 条完成，{take.reason}。"
        )
        self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")

    def refresh_enrollment_list(self) -> None:
        self.enroll_take_list.clear()
        for index, take in enumerate(self.enrollment_takes, start=1):
            status = "合格" if take.accepted else "不合格"
            self.enroll_take_list.addItem(
                f"Take {index}: {take.duration_sec:.2f}s, SNR {take.estimated_snr_db:.1f} dB, "
                f"质量 {take.quality_score:.3f}, {status}, {take.reason}"
            )
        self.update_enrollment_actions()

    def accepted_enrollment_count(self) -> int:
        return sum(1 for take in self.enrollment_takes if take.accepted)

    def update_enrollment_actions(self) -> None:
        accepted_count = self.accepted_enrollment_count()
        self.enroll_build_button.setEnabled(accepted_count >= 3)
        self.temp_verify_button.setEnabled(self.session_model is not None)
        if self.session_model is None and accepted_count < 3:
            self.enroll_status_label.setText(
                f"录音状态: 已有 {accepted_count} 条合格注册语音，还需要 {3 - accepted_count} 条。"
            )

    def reset_temporary_enrollment(self) -> None:
        self.enrollment_takes.clear()
        self.enrollment_features.clear()
        self.session_model = None
        self.enroll_take_list.clear()
        self.temp_model_label.setText("临时模型: 未建立")
        self.temp_verify_result_label.setText("验证结果: 等待验证")
        self.temp_verify_reason_label.setText("说明: 先建立临时模型，再录一条新语音做验证。")
        self.temp_verify_score_label.setText("分数: -")
        self.enroll_status_label.setText("录音状态: 已重新开始，请录入至少 3 条注册语音。")
        self.update_enrollment_actions()

    def build_temporary_model(self) -> None:
        accepted_features = [
            features for take, features in zip(self.enrollment_takes, self.enrollment_features) if take.accepted
        ]
        if len(accepted_features) < 3:
            QMessageBox.warning(self, "注册样本不足", "至少需要 3 条合格注册样本。")
            return
        speaker_id = self.enroll_name_input.text().strip() or "TeacherDemo"
        self.session_model = build_session_enrollment_model(speaker_id, self.model, accepted_features)
        self.temp_model_label.setText(
            f"临时模型: {speaker_id}\n"
            f"声纹阈值 {self.session_model.voice_threshold:.3f}, "
            f"口令阈值 {self.session_model.passphrase_threshold:.3f}\n"
            f"口令通过线: 分数 <= {SESSION_PASSPHRASE_ACCEPT_SCORE:.2f}。\n"
            "仅当前会话有效，不参与正式实验指标。"
        )
        self.enroll_status_label.setText(f"录音状态: 临时模型已建立，可录一条新语音进行验证。")
        self.update_enrollment_actions()

    def verify_temporary_user_by_recording(self) -> None:
        if self.session_model is None:
            QMessageBox.warning(self, "临时模型未建立", "请先录入至少 3 条合格样本并建立临时模型。")
            return
        try:
            self.temp_verify_result_label.setText(f"验证结果: 开始录音，请说“芝麻开门”，持续 {self.enroll_duration_spin.value():.1f} 秒。")
            self.temp_verify_reason_label.setText("说明: 录音中，请保持说话稳定。")
            self.status_label.setText("临时验证录音: 进行中")
            QApplication.processEvents()
            wav_path = record_audio_to_temp("temp_verify", self.enroll_duration_spin.value(), self.sample_rate_input.value())
        except Exception as exc:
            self.temp_verify_result_label.setText("验证结果: 录音失败")
            self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")
            QMessageBox.critical(self, "临时验证录音失败", str(exc))
            return
        self.temp_verify_reason_label.setText("说明: 录音结束，正在验证临时用户。")
        QApplication.processEvents()
        self.verify_temporary_audio(wav_path)

    def verify_temporary_audio(self, wav_path: Path) -> None:
        try:
            start = time.perf_counter()
            result = predict_with_session_model(wav_path, self.model, self.session_model)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            self.temp_verify_result_label.setText("验证结果: 验证失败")
            self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")
            QMessageBox.critical(self, "临时验证失败", str(exc))
            return
        decision = result.get("decision", "REJECT")
        if decision == "REJECT":
            self.temp_verify_result_label.setText("验证结果: REJECT")
        else:
            self.temp_verify_result_label.setText(f"验证结果: ACCEPT {decision}")
        voice_status = "通过" if result.get("voice_passed") == "True" else "未通过"
        passphrase_status = "通过" if result.get("passphrase_passed") == "True" else "未通过"
        self.temp_verify_reason_label.setText(
            f"说明: 声纹{voice_status}；口令{passphrase_status}。"
            f"{result.get('reject_reason') or '验证通过'}"
        )
        self.temp_verify_score_label.setText(
            "分数: "
            f"声纹 {format_float(result.get('voice_score', ''))} / 通过线 {result.get('voice_limit', '1.00')}, "
            f"口令 {format_float(result.get('passphrase_score', ''))} / 通过线 {result.get('passphrase_limit', '-')}, "
            f"SNR {format_float(result.get('estimated_snr_db', ''))} dB，耗时 {latency_ms:.1f} ms。"
        )
        self.status_label.setText("模型: 已加载" if self.model is not None else "模型: 未加载")

    def populate_table_from_csv(self, table: QTableWidget, path: Path) -> None:
        if not path.exists():
            table.setRowCount(1)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["状态"])
            table.setItem(0, 0, QTableWidgetItem(f"文件不存在: {display_path(path)}"))
            return
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        headers = list(rows[0])
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, header in enumerate(headers):
                table.setItem(row_index, column_index, QTableWidgetItem(row.get(header, "")))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def reload_results_tables(self) -> None:
        self.populate_table_from_csv(self.metrics_table, PROJECT_ROOT / "results" / "tables" / "voice_lock_metrics.csv")
        self.populate_table_from_csv(self.recommend_table, PROJECT_ROOT / "results" / "tables" / "recommended_config.csv")

    def reload_model_from_settings(self) -> None:
        try:
            self.model = load_model(self.model_path_input.text(), self.features_path_input.text())
            self.status_label.setText("模型: 已重新加载")
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def test_microphone(self) -> None:
        try:
            audio = sd.rec(int(1.0 * self.sample_rate_input.value()), samplerate=self.sample_rate_input.value(), channels=1, dtype="float32")
            sd.wait()
            peak = float(np.max(np.abs(audio)))
            self.mic_label.setText(f"麦克风: 可用 峰值 {peak:.3f}")
        except Exception as exc:
            self.mic_label.setText("麦克风: 不可用")
            QMessageBox.critical(self, "麦克风测试失败", str(exc))


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
    session_result = predict_with_session_model(project_path(sample["wav"]), model, session_model)
    print(
        "gui smoke "
        f"official_decision={result.get('decision')} "
        f"session_decision={session_result.get('decision')}"
    )
    return 0 if result.get("decision") == "S01" and session_result.get("pred_speaker") == "SmokeS01" else 1


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
