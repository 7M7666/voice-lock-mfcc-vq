# SRDP PySide6 GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySide6 desktop GUI for the SRDP voice-lock prototype that can demonstrate fixed samples, selected WAV files, live recording, scores, waveform, MFCC, rejection reasons, history, and temporary on-site enrollment without rewriting the existing backend algorithm.

**Architecture:** Keep `code/python/srdp.py` as the source of truth for model loading, feature extraction, and official prediction. Add one PySide6 GUI entry file that calls `load_model()` once at startup and reuses `predict_one()` for official verification. Temporary enrollment is session-only, uses existing feature extraction and VQ/DTW helpers, and never writes into the formal dataset unless a future explicit save flow is added.

**Tech Stack:** Python 3.11, PySide6, Matplotlib QtAgg canvas, sounddevice, soundfile, librosa, numpy, existing `srdp.py`, existing `run.ps1`.

---

## Product Direction

### Design Benchmarks

- Audacity: waveform and timeline views are readable because audio data gets the largest continuous area, controls stay compact, and playback/recording actions are visually grouped.
- Windows Security: verdict state is instantly understandable because the first screen has a large status badge, clear severity color, and one sentence explaining what happened.
- Linear: dense operational interfaces feel calm when the left navigation is stable, the main workspace is task-focused, and secondary details stay in right-side or lower panels.

### Visual Thesis

Build a "laboratory verification console": light background, high-contrast text, restrained borders, one large verdict badge, and instrument-like score bars. The interface should look like a system that can be trusted during a defense demo, not like a marketing page.

### Color And Typography

- Background: `#F4F6F8`
- Main panel: `#FFFFFF`
- Secondary panel: `#EEF2F6`
- Border: `#CBD5E1`
- Text primary: `#17212B`
- Text secondary: `#5B6775`
- Accept: `#0B7A53`
- Reject: `#B42318`
- Warning: `#B7791F`
- Accent blue for neutral actions: `#2563A6`
- Font: `Microsoft YaHei UI`, fallback `Segoe UI`
- Monospace score font: `Consolas`
- Card radius: `6px`

### Signature Interaction

When a prediction finishes, the verdict badge updates first, then score bars fill from left to right over about 250 ms. This makes the decision feel causal: audio input leads to score calculation, then final verdict.

---

## Scope

### Building

- A PySide6 desktop GUI launched by `.\run.ps1 gui`.
- A default "答辩演示台" page with fixed examples, WAV selection, live recording, prediction, waveform, MFCC, score bars, rejection reason, and history.
- A "临时注册" page that records 3-5 samples, checks quality, builds an in-memory session model, and verifies against that temporary model.
- A "实验结果" page that displays existing metrics and recommended configuration from CSV files.
- A "设置" page for model paths, recording duration, sample rate, and microphone status.

### Not Building

- No deep learning model.
- No commercial-grade anti-spoofing claim.
- No automatic writing into `metadata_srdp.csv` from the first GUI version.
- No replacement of `srdp.py` algorithm logic.
- No database.
- No network service.
- No packaging into `.exe` in the first implementation pass.

---

## File Structure

- Modify: `code/python/requirements.txt`
  - Add `PySide6`.
- Modify: `run.ps1`
  - Add a `gui` command that launches `code/python/gui.py`.
- Create: `code/python/gui.py`
  - PySide6 application entrypoint.
  - Main window, pages, worker threads, audio recording, plotting, history, and session-only enrollment.
- Modify: `README.md`
  - Add GUI startup and defense-demo instructions after the existing MATLAB frontend section.

This plan intentionally uses one new GUI file. The project currently has a single large backend file, and for a course prototype one GUI entry file keeps execution simple. Split into multiple GUI modules only if `gui.py` grows beyond about 900 lines during implementation.

---

## Page Map

### Page 1: 答辩演示台

Purpose: the first screen should complete the defense demo without navigation.

Layout:

```text
┌────────────────────────────────────────────────────────────────────┐
│ 基于 MFCC 与 VQ 的多特征融合声音锁原型系统     模型: 已加载  麦克风: 可用 │
├──────────────┬───────────────────────────────────┬─────────────────┤
│ 左侧导航      │ 中央音频工作区                      │ 右侧判定面板       │
│              │                                   │                 │
│ ● 答辩演示台  │ [输入来源] 固定样例 / 选择WAV / 录音 │ [ACCEPT S01]     │
│   临时注册    │ [样例下拉] [选择文件] [录音] [验证]  │ 预测用户 S01      │
│   实验结果    │                                   │ 拒绝原因 无       │
│   历史记录    │ 波形图                              │                 │
│   设置        │ MFCC 图                             │ 声纹分数 0.72/1.0 │
│              │                                   │ 口令分数 0.64/1.0 │
│              │ 最近验证历史表                       │ 融合分数 0.69/1.0 │
└──────────────┴───────────────────────────────────┴─────────────────┘
```

Required controls:

- `固定样例` combo box:
  - `正常通过 S01`
  - `正常通过 S02`
  - `错误口令拒绝`
  - `非注册用户拒绝`
  - `边界样例`
- `选择 WAV` button.
- `录音验证` button.
- Recording duration spin box, default `3.0` seconds.
- `开始验证` primary button.
- `播放音频` secondary button if Qt multimedia is available; if not, omit playback in the first pass.
- Metadata controls:
  - speaker id: `S01`, `S02`, `S03`, `S04`, `U01`, empty
  - split: `test`, `wrong_passphrase`, `impostor`, `replay`
  - trial type: `normal`, `wrong_passphrase`, `impostor`, `replay`, `low_volume`, `noise`, `fast`, `slow`

Result fields:

- `decision`
- `pred_speaker`
- `voice_score`
- `passphrase_score`
- `fusion_score`
- `threshold`
- `passphrase_threshold`
- `fusion_threshold`
- `estimated_snr_db`
- `quality_score`
- `reject_reason`
- `latency_ms`, measured in the GUI worker.

### Page 2: 临时注册

Purpose: show that the prototype can support on-site registration as an add-on demo, while keeping formal results reproducible.

Layout:

```text
┌────────────────────────────────────────────────────────────────────┐
│ 临时注册, 只在当前演示会话生效，不写入正式数据集                      │
├───────────────────────┬────────────────────────────────────────────┤
│ 注册设置               │ 录音样本列表                                │
│ 用户名: TeacherDemo    │ Take 1  3.1s  SNR 22.5  合格  [重录]         │
│ 口令: 芝麻开门          │ Take 2  2.9s  SNR 20.8  合格  [重录]         │
│ 录音时长: 3.0s          │ Take 3  3.2s  SNR 18.2  合格  [重录]         │
│ [录一条注册样本]        │ Take 4  0.8s  时长过短 [重录]               │
│ [建立临时模型]          │                                            │
├───────────────────────┴────────────────────────────────────────────┤
│ 临时验证: [录音验证] [选择WAV验证]  判定: ACCEPT / REJECT             │
└────────────────────────────────────────────────────────────────────┘
```

Quality rules:

- Duration must be between `1.2` and `5.0` seconds.
- `low_quality` from backend extraction must be `False`.
- SNR should be displayed. If SNR is below `10 dB`, mark as warning.
- At least `3` accepted enrollment takes are required before building a temporary model.
- Temporary registration output must show: "临时演示模型，不参与正式实验指标".

Session-only model behavior:

- Use existing z-score and feature selection from the loaded official model.
- Use existing `lbg_codebook()` to build a temporary codebook.
- Use enrollment samples as passphrase DTW templates.
- Estimate temporary thresholds from enrollment internal distances.
- Never overwrite `models/lbg_vq_codebooks.npz`.
- Never append metadata automatically.

### Page 3: 实验结果

Purpose: support teacher questions about metrics without opening CSV files.

Panels:

- Current baseline metrics from `results/tables/voice_lock_metrics.csv`.
- Recommended safety-first config from `results/tables/recommended_config.csv`.
- One explanation block:
  - "当前系统已完成自采录音到判定闭环。安全优先配置保持 False Accept 为 0，同时把总体准确率提高到 90.5%。"

### Page 4: 历史记录

Purpose: show the last predictions as evidence of repeatable interaction.

Columns:

- time
- source
- decision
- pred_speaker
- voice_score
- passphrase_score
- fusion_score
- reject_reason
- latency_ms

Actions:

- Clear session history.
- Export session history to `results/gui_history/session_YYYYMMDD_HHMMSS.csv` only after explicit button click.

### Page 5: 设置

Fields:

- model path, default `models/lbg_vq_codebooks.npz`
- features path, default `data/features/srdp_features.npz`
- recording sample rate, default `16000`
- recording duration, default `3.0`
- temporary audio directory, default `tmp/gui_recordings`

Actions:

- Reload model.
- Test microphone for 1 second.
- Open project README.

---

## Implementation Tasks

### Task 1: Add GUI Dependency And Launcher

**Files:**
- Modify: `code/python/requirements.txt`
- Modify: `run.ps1`

- [ ] **Step 1: Add PySide6 to requirements**

Set `code/python/requirements.txt` to:

```text
numpy
sounddevice
soundfile
librosa
matplotlib
PySide6
```

- [ ] **Step 2: Add `gui` command to `run.ps1`**

Replace `run.ps1` with:

```powershell
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot "code\python\venv\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "code\python\srdp.py"
$Gui = Join-Path $ProjectRoot "code\python\gui.py"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python venv not found: $Python"
    exit 1
}

if ($Args.Count -gt 0 -and $Args[0] -eq "gui") {
    $GuiArgs = @()
    if ($Args.Count -gt 1) {
        $GuiArgs = $Args[1..($Args.Count - 1)]
    }
    & $Python $Gui @GuiArgs
    exit $LASTEXITCODE
}

& $Python $Main @Args
exit $LASTEXITCODE
```

- [ ] **Step 3: Install the dependency**

Run:

```powershell
.\code\python\venv\Scripts\python.exe -m pip install -r code/python/requirements.txt
```

Expected: installation completes and reports `PySide6` installed or already satisfied.

- [ ] **Step 4: Verify existing backend still works**

Run:

```powershell
.\run.ps1 smoke-test --skip-pipeline
```

Expected:

```text
smoke test passed
```

- [ ] **Step 5: Commit**

```powershell
git add code/python/requirements.txt run.ps1
git commit -m "chore: add PySide6 GUI launcher"
```

### Task 2: Create GUI Skeleton And Model Loading

**Files:**
- Create: `code/python/gui.py`

- [ ] **Step 1: Create the initial GUI file**

Create `code/python/gui.py` with:

```python
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal
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
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
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


def format_float(value: str, empty: str = "-") -> str:
    if value is None or value == "":
        return empty
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def metadata_for_sample(sample: dict[str, str]) -> dict[str, str]:
    row = normalise_metadata_row({})
    row.update(
        {
            "file_path": sample.get("wav", ""),
            "speaker_id": sample.get("speaker_id", ""),
            "split": sample.get("split", ""),
            "expected_decision": sample.get("expected_decision", ""),
            "trial_type": sample.get("trial_type", ""),
            "is_replay": "false",
        }
    )
    return row
```

- [ ] **Step 2: Add theme constants**

Append to `gui.py`:

```python
APP_QSS = """
QMainWindow {
    background: #F4F6F8;
}
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI";
    color: #17212B;
    font-size: 13px;
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
"""
```

- [ ] **Step 3: Add a minimal main window**

Append to `gui.py`:

```python
class SrdpVoiceLockWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SRDP 轻量化声音锁演示系统")
        self.resize(1280, 820)
        self.model = None
        self.history: list[HistoryRow] = []
        self.current_audio_path: Path | None = None

        self.status_label = QLabel("模型: 加载中")
        self.mic_label = QLabel("麦克风: 未检测")
        self.stack = QStackedWidget()

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

    def load_model_once(self) -> None:
        try:
            self.model = load_model()
            self.status_label.setText("模型: 已加载")
        except Exception as exc:
            self.status_label.setText("模型: 加载失败")
            QMessageBox.critical(self, "模型加载失败", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    window = SrdpVoiceLockWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify syntax**

Run:

```powershell
.\code\python\venv\Scripts\python.exe -m py_compile code/python/gui.py
```

Expected: command exits with code `0`.

- [ ] **Step 5: Verify launch**

Run:

```powershell
.\run.ps1 gui
```

Expected: a desktop window opens, title is visible, model status becomes `模型: 已加载`.

- [ ] **Step 6: Commit**

```powershell
git add code/python/gui.py
git commit -m "feat: add SRDP GUI shell"
```

### Task 3: Build The Defense Demo Page

**Files:**
- Modify: `code/python/gui.py`

- [ ] **Step 1: Add plotting canvases**

Add these classes above `SrdpVoiceLockWindow`:

```python
class WaveformCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 2.2), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.plot_empty()

    def plot_empty(self) -> None:
        self.axes.clear()
        self.axes.set_title("波形")
        self.axes.set_xlabel("时间")
        self.axes.set_ylabel("幅度")
        self.axes.grid(True, alpha=0.25)
        self.draw()

    def plot_wav(self, wav_path: Path) -> None:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        times = np.arange(len(y)) / float(sr)
        self.axes.clear()
        self.axes.plot(times, y, color="#2563A6", linewidth=0.9)
        self.axes.set_title("波形")
        self.axes.set_xlabel("时间 / s")
        self.axes.set_ylabel("幅度")
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
        self.axes.set_xlabel("帧")
        self.axes.set_ylabel("系数")
        self.draw()

    def plot_wav(self, wav_path: Path) -> None:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        self.axes.clear()
        image = self.axes.imshow(mfcc, aspect="auto", origin="lower", cmap="magma")
        self.axes.set_title("MFCC")
        self.axes.set_xlabel("帧")
        self.axes.set_ylabel("系数")
        self.figure.colorbar(image, ax=self.axes, fraction=0.025, pad=0.02)
        self.draw()
```

- [ ] **Step 2: Add demo page construction**

Add this method inside `SrdpVoiceLockWindow`:

```python
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
        self.sample_combo.currentTextChanged.connect(self.load_selected_demo_sample)
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(["", "S01", "S02", "S03", "S04", "U01", "S01_wrong", "S02_wrong", "S03_wrong", "S04_wrong"])
        self.split_combo = QComboBox()
        self.split_combo.addItems(["test", "wrong_passphrase", "impostor", "replay"])
        self.trial_combo = QComboBox()
        self.trial_combo.addItems(["normal", "wrong_passphrase", "impostor", "replay", "low_volume", "noise", "fast", "slow"])
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 5.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setValue(3.0)
        self.current_file_label = QLabel("未选择音频")
        choose_button = QPushButton("选择 WAV")
        choose_button.clicked.connect(self.choose_wav)
        record_button = QPushButton("录音验证")
        record_button.clicked.connect(self.record_for_demo)
        predict_button = QPushButton("开始验证")
        predict_button.setObjectName("PrimaryButton")
        predict_button.clicked.connect(self.predict_current_audio)

        input_layout.addRow("固定样例", self.sample_combo)
        input_layout.addRow("当前文件", self.current_file_label)
        input_layout.addRow("真实用户", self.speaker_combo)
        input_layout.addRow("样本类型", self.split_combo)
        input_layout.addRow("试验类型", self.trial_combo)
        input_layout.addRow("录音秒数", self.duration_spin)
        input_layout.addRow(choose_button, record_button)
        input_layout.addRow(predict_button)

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
        self.voice_score_label = QLabel("声纹分数: -")
        self.passphrase_score_label = QLabel("口令分数: -")
        self.fusion_score_label = QLabel("融合分数: -")
        self.snr_label = QLabel("SNR: -")
        self.quality_label = QLabel("质量分: -")
        result_layout.addWidget(self.verdict_label)
        result_layout.addWidget(self.pred_label)
        result_layout.addWidget(self.reason_label)
        result_layout.addWidget(self.voice_score_label)
        result_layout.addWidget(self.passphrase_score_label)
        result_layout.addWidget(self.fusion_score_label)
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
        self.load_selected_demo_sample()
        return page
```

- [ ] **Step 3: Add the demo page to the stack**

In `SrdpVoiceLockWindow.__init__`, after `self.stack = QStackedWidget()`, add:

```python
        self.stack.addWidget(self.build_demo_page())
        self.stack.addWidget(QWidget())
        self.stack.addWidget(QWidget())
        self.stack.addWidget(QWidget())
        self.stack.addWidget(QWidget())
```

- [ ] **Step 4: Verify launch**

Run:

```powershell
.\run.ps1 gui
```

Expected: first page shows input controls, waveform area, MFCC area, verdict panel, and empty history table.

- [ ] **Step 5: Commit**

```powershell
git add code/python/gui.py
git commit -m "feat: build SRDP defense demo page"
```

### Task 4: Wire WAV Selection, Recording, And Official Prediction

**Files:**
- Modify: `code/python/gui.py`

- [ ] **Step 1: Add recording helper**

Add above `SrdpVoiceLockWindow`:

```python
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
```

- [ ] **Step 2: Add prediction worker**

Add above `SrdpVoiceLockWindow`:

```python
class PredictWorker(QObject):
    finished = Signal(dict, str, float)
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
```

- [ ] **Step 3: Add demo interaction methods**

Add inside `SrdpVoiceLockWindow`:

```python
    def load_selected_demo_sample(self) -> None:
        sample = DEMO_SAMPLES[self.sample_combo.currentText()]
        wav_path = project_path(sample["wav"])
        self.current_audio_path = wav_path
        self.current_file_label.setText(display_path(wav_path))
        self.speaker_combo.setCurrentText(sample["speaker_id"])
        self.split_combo.setCurrentText(sample["split"])
        self.trial_combo.setCurrentText(sample["trial_type"])
        self.update_audio_plots(wav_path)

    def choose_wav(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择 WAV 音频", str(PROJECT_ROOT), "WAV files (*.wav)")
        if not filename:
            return
        wav_path = Path(filename)
        self.current_audio_path = wav_path
        self.current_file_label.setText(display_path(wav_path))
        self.update_audio_plots(wav_path)

    def record_for_demo(self) -> None:
        try:
            wav_path = record_audio_to_temp("verify", self.duration_spin.value())
        except Exception as exc:
            QMessageBox.critical(self, "录音失败", str(exc))
            return
        self.current_audio_path = wav_path
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
        row.update(
            {
                "file_path": display_path(self.current_audio_path) if self.current_audio_path else "",
                "speaker_id": self.speaker_combo.currentText(),
                "split": self.split_combo.currentText(),
                "expected_decision": self.speaker_combo.currentText() if self.split_combo.currentText() == "test" else "REJECT",
                "trial_type": self.trial_combo.currentText(),
                "is_replay": "true" if self.split_combo.currentText() == "replay" else "false",
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
        self.start_predict_worker(self.current_audio_path, self.current_metadata(), "official")

    def start_predict_worker(self, wav_path: Path, metadata: dict[str, str], source_label: str) -> None:
        self.predict_thread = QThread()
        self.predict_worker = PredictWorker(wav_path, self.model, metadata, source_label)
        self.predict_worker.moveToThread(self.predict_thread)
        self.predict_thread.started.connect(self.predict_worker.run)
        self.predict_worker.finished.connect(self.on_prediction_finished)
        self.predict_worker.failed.connect(self.on_prediction_failed)
        self.predict_worker.finished.connect(self.predict_thread.quit)
        self.predict_worker.failed.connect(self.predict_thread.quit)
        self.predict_thread.start()

    def on_prediction_failed(self, message: str) -> None:
        QMessageBox.critical(self, "验证失败", message)

    def on_prediction_finished(self, result: dict, source_label: str, latency_ms: float) -> None:
        self.update_result_panel(result)
        self.add_history_row(result, source_label, latency_ms)
```

- [ ] **Step 4: Add result panel and history updates**

Add inside `SrdpVoiceLockWindow`:

```python
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
        self.voice_score_label.setText(
            f"声纹分数: {format_float(result.get('voice_score', ''))} / 阈值 1.000"
        )
        self.passphrase_score_label.setText(
            f"口令分数: {format_float(result.get('passphrase_score', ''))} / 阈值 1.000"
        )
        self.fusion_score_label.setText(
            f"融合分数: {format_float(result.get('fusion_score', ''))} / 阈值 {format_float(result.get('fusion_threshold', ''))}"
        )
        self.snr_label.setText(f"SNR: {format_float(result.get('estimated_snr_db', ''))} dB")
        self.quality_label.setText(f"质量分: {format_float(result.get('quality_score', ''))}")

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

    def refresh_history_table(self) -> None:
        self.history_table.setRowCount(len(self.history))
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
                self.history_table.setItem(row_index, column_index, QTableWidgetItem(value))
```

- [ ] **Step 5: Verify official sample prediction**

Run:

```powershell
.\run.ps1 gui
```

Manual expected result:

- Select `正常通过 S01`.
- Click `开始验证`.
- Verdict panel shows `ACCEPT S01`.
- Waveform and MFCC remain visible.
- History table gets one row.

- [ ] **Step 6: Commit**

```powershell
git add code/python/gui.py
git commit -m "feat: connect GUI to official voice-lock prediction"
```

### Task 5: Add Temporary Enrollment

**Files:**
- Modify: `code/python/gui.py`

- [ ] **Step 1: Add enrollment model builder**

Add above `SrdpVoiceLockWindow`:

```python
def assess_enrollment_take(wav_path: Path, model) -> tuple[EnrollmentTake, np.ndarray]:
    extracted = extract_one_audio_features(wav_path, model)
    duration_ok = 1.2 <= extracted.duration_sec <= 5.0
    quality_ok = not extracted.low_quality
    snr_ok = extracted.estimated_snr_db >= 10.0
    accepted = duration_ok and quality_ok
    if not duration_ok:
        reason = "时长不合格"
    elif not quality_ok:
        reason = "录音质量偏低"
    elif not snr_ok:
        reason = "SNR偏低，可用于演示但建议重录"
    else:
        reason = "合格"
    take = EnrollmentTake(
        path=wav_path,
        duration_sec=extracted.duration_sec,
        estimated_snr_db=extracted.estimated_snr_db,
        quality_score=extracted.quality_score,
        low_quality=extracted.low_quality,
        accepted=accepted,
        reason=reason,
    )
    return take, extracted.features


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
    target_size = min(16, max(4, stacked.shape[0] // 20))
    codebook = lbg_codebook(stacked, target_size=target_size)
    own_distances = np.array([quantization_distance(matrix, codebook) for matrix in speaker_mats], dtype=float)
    voice_threshold = float(min(own_distances.mean() + 2.0 * own_distances.std(), own_distances.max() * 1.25))
    if len(passphrase_mats) >= 2:
        passphrase_distances = []
        for index, matrix in enumerate(passphrase_mats):
            templates = [template for template_index, template in enumerate(passphrase_mats) if template_index != index]
            passphrase_distances.append(
                passphrase_template_distance(
                    matrix,
                    templates,
                    official_model.passphrase_aggregate,
                    window_ratio=official_model.dtw_window_ratio,
                    warp_penalty=official_model.dtw_warp_penalty,
                )
            )
        passphrase_values = np.array(passphrase_distances, dtype=float)
        passphrase_threshold = float(min(passphrase_values.mean() + 2.0 * passphrase_values.std(), passphrase_values.max() * 1.25))
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
```

- [ ] **Step 2: Add temporary prediction helper**

Add above `SrdpVoiceLockWindow`:

```python
def predict_with_session_model(
    wav_path: Path,
    official_model,
    session_model: SessionEnrollmentModel,
) -> dict[str, str]:
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
    passphrase_ok = passphrase_distance <= session_model.passphrase_threshold
    decision = session_model.speaker_id if voice_ok and passphrase_ok else "REJECT"
    if not voice_ok:
        reject_reason = "临时声纹距离超过阈值"
    elif not passphrase_ok:
        reject_reason = "临时口令DTW距离超过阈值"
    else:
        reject_reason = ""
    return {
        "decision": decision,
        "pred_speaker": session_model.speaker_id,
        "voice_score": f"{voice_score:.6f}",
        "passphrase_score": f"{passphrase_score:.6f}",
        "fusion_score": "",
        "fusion_threshold": "",
        "estimated_snr_db": f"{extracted.estimated_snr_db:.6f}",
        "quality_score": f"{extracted.quality_score:.6f}",
        "reject_reason": reject_reason,
    }
```

- [ ] **Step 3: Add enrollment page**

Add this method inside `SrdpVoiceLockWindow`:

```python
    def build_enrollment_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        settings_group = QGroupBox("临时注册设置")
        settings_layout = QFormLayout(settings_group)
        self.enroll_name_input = QLineEdit("TeacherDemo")
        self.enroll_duration_spin = QDoubleSpinBox()
        self.enroll_duration_spin.setRange(1.0, 5.0)
        self.enroll_duration_spin.setValue(3.0)
        self.enroll_record_button = QPushButton("录一条注册样本")
        self.enroll_record_button.clicked.connect(self.record_enrollment_take)
        self.enroll_build_button = QPushButton("建立临时模型")
        self.enroll_build_button.setObjectName("PrimaryButton")
        self.enroll_build_button.clicked.connect(self.build_temporary_model)
        settings_layout.addRow("临时用户名", self.enroll_name_input)
        settings_layout.addRow("口令", QLabel("芝麻开门"))
        settings_layout.addRow("录音秒数", self.enroll_duration_spin)
        settings_layout.addRow(self.enroll_record_button)
        settings_layout.addRow(self.enroll_build_button)

        takes_group = QGroupBox("注册样本")
        takes_layout = QVBoxLayout(takes_group)
        self.enroll_take_list = QListWidget()
        takes_layout.addWidget(self.enroll_take_list)

        verify_group = QGroupBox("临时验证")
        verify_layout = QHBoxLayout(verify_group)
        self.temp_verify_button = QPushButton("录音验证临时用户")
        self.temp_verify_button.clicked.connect(self.verify_temporary_user_by_recording)
        self.temp_model_label = QLabel("临时模型: 未建立")
        verify_layout.addWidget(self.temp_model_label)
        verify_layout.addWidget(self.temp_verify_button)

        layout.addWidget(settings_group, 0, 0)
        layout.addWidget(takes_group, 0, 1)
        layout.addWidget(verify_group, 1, 0, 1, 2)
        return page
```

- [ ] **Step 4: Store enrollment state**

In `SrdpVoiceLockWindow.__init__`, before `self.status_label = QLabel("模型: 加载中")`, add:

```python
        self.enrollment_takes: list[EnrollmentTake] = []
        self.enrollment_features: list[np.ndarray] = []
        self.session_model: SessionEnrollmentModel | None = None
```

Replace the placeholder second stack page with:

```python
        self.stack.addWidget(self.build_enrollment_page())
```

- [ ] **Step 5: Add enrollment interaction methods**

Add inside `SrdpVoiceLockWindow`:

```python
    def record_enrollment_take(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "模型未加载", "模型未加载，无法提取注册特征。")
            return
        try:
            wav_path = record_audio_to_temp("enroll", self.enroll_duration_spin.value())
            take, features = assess_enrollment_take(wav_path, self.model)
        except Exception as exc:
            QMessageBox.critical(self, "注册录音失败", str(exc))
            return
        self.enrollment_takes.append(take)
        self.enrollment_features.append(features)
        self.refresh_enrollment_list()

    def refresh_enrollment_list(self) -> None:
        self.enroll_take_list.clear()
        for index, take in enumerate(self.enrollment_takes, start=1):
            status = "合格" if take.accepted else "不合格"
            self.enroll_take_list.addItem(
                f"Take {index}: {take.duration_sec:.2f}s, SNR {take.estimated_snr_db:.1f} dB, "
                f"质量 {take.quality_score:.3f}, {status}, {take.reason}"
            )

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
            f"临时模型: {speaker_id}, 声纹阈值 {self.session_model.voice_threshold:.3f}, "
            f"口令阈值 {self.session_model.passphrase_threshold:.3f}"
        )

    def verify_temporary_user_by_recording(self) -> None:
        if self.session_model is None:
            QMessageBox.warning(self, "临时模型未建立", "请先录入至少 3 条合格样本并建立临时模型。")
            return
        try:
            wav_path = record_audio_to_temp("temp_verify", self.enroll_duration_spin.value())
            start = time.perf_counter()
            result = predict_with_session_model(wav_path, self.model, self.session_model)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            QMessageBox.critical(self, "临时验证失败", str(exc))
            return
        self.current_audio_path = wav_path
        self.update_audio_plots(wav_path)
        self.update_result_panel(result)
        self.add_history_row(result, "temporary", latency_ms)
        self.stack.setCurrentIndex(0)
```

- [ ] **Step 6: Verify temporary registration**

Run:

```powershell
.\run.ps1 gui
```

Manual expected result:

- Open `临时注册`.
- Record 3 samples.
- Click `建立临时模型`.
- Label changes to `临时模型: TeacherDemo`.
- Click `录音验证临时用户`.
- App returns to `答辩演示台`, shows waveform, MFCC, verdict, and history row.

- [ ] **Step 7: Commit**

```powershell
git add code/python/gui.py
git commit -m "feat: add session-only voice-lock enrollment"
```

### Task 6: Add Experiment Results Page

**Files:**
- Modify: `code/python/gui.py`

- [ ] **Step 1: Add CSV table helper**

Add inside `SrdpVoiceLockWindow`:

```python
    def populate_table_from_csv(self, table: QTableWidget, path: Path) -> None:
        if not path.exists():
            table.setRowCount(0)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["状态"])
            table.setRowCount(1)
            table.setItem(0, 0, QTableWidgetItem(f"文件不存在: {display_path(path)}"))
            return
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            table.setRowCount(0)
            return
        headers = list(rows[0])
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, header in enumerate(headers):
                table.setItem(row_index, column_index, QTableWidgetItem(row.get(header, "")))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
```

- [ ] **Step 2: Add results page**

Add inside `SrdpVoiceLockWindow`:

```python
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

    def reload_results_tables(self) -> None:
        self.populate_table_from_csv(self.metrics_table, PROJECT_ROOT / "results" / "tables" / "voice_lock_metrics.csv")
        self.populate_table_from_csv(self.recommend_table, PROJECT_ROOT / "results" / "tables" / "recommended_config.csv")
```

Replace the results placeholder stack page with:

```python
        self.stack.addWidget(self.build_results_page())
```

- [ ] **Step 3: Verify results page**

Run:

```powershell
.\run.ps1 gui
```

Expected: `实验结果` page shows rows from `voice_lock_metrics.csv` and `recommended_config.csv`.

- [ ] **Step 4: Commit**

```powershell
git add code/python/gui.py
git commit -m "feat: show SRDP experiment metrics in GUI"
```

### Task 7: Add History And Settings Pages

**Files:**
- Modify: `code/python/gui.py`

- [ ] **Step 1: Add history page**

Add inside `SrdpVoiceLockWindow`:

```python
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
```

Modify `refresh_history_table()` so it also updates `self.full_history_table` when the attribute exists:

```python
        if hasattr(self, "full_history_table"):
            self.full_history_table.setRowCount(len(self.history))
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
                    self.full_history_table.setItem(row_index, column_index, QTableWidgetItem(value))
```

- [ ] **Step 2: Add settings page**

Add inside `SrdpVoiceLockWindow`:

```python
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
```

Replace the history and settings placeholder stack pages with:

```python
        self.stack.addWidget(self.build_history_page())
        self.stack.addWidget(self.build_settings_page())
```

- [ ] **Step 3: Verify pages**

Run:

```powershell
.\run.ps1 gui
```

Expected:

- `历史记录` page mirrors recent predictions.
- Export creates `results/gui_history/session_*.csv` only when the user clicks export.
- `设置` page can test microphone and reload the model.

- [ ] **Step 4: Commit**

```powershell
git add code/python/gui.py
git commit -m "feat: add GUI history and settings pages"
```

### Task 8: Add README Instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add GUI section**

Add after the MATLAB frontend section:

```markdown
## 启动 Python GUI

结题阶段推荐使用 PySide6 一体化界面进行现场展示：

```powershell
.\run.ps1 gui
```

GUI 默认复用 Python 后端模型和特征文件，不重写识别逻辑：

- 模型：`models/lbg_vq_codebooks.npz`
- 特征：`data/features/srdp_features.npz`
- 单条推理入口：`predict_one()`

推荐演示顺序：

1. 在“答辩演示台”选择固定样例，依次展示正常通过、错误口令拒绝、非注册用户拒绝。
2. 点击“录音验证”，现场录一条 2-4 秒音频并验证。
3. 打开“实验结果”，说明当前正式指标和安全优先推荐配置。
4. 如老师希望现场注册，打开“临时注册”，录入 3-5 条样本后建立临时会话模型。

临时注册只在当前 GUI 会话中生效，不会自动写入正式 metadata、模型或实验结果。
```

- [ ] **Step 2: Verify README renders**

Run:

```powershell
Get-Content README.md -Encoding UTF8 | Select-String -Pattern "启动 Python GUI" -Context 0,20
```

Expected: the new section is visible and commands are correct.

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "docs: add PySide6 GUI usage notes"
```

### Task 9: Final Verification And Rehearsal

**Files:**
- No code changes.

- [ ] **Step 1: Run backend smoke test**

```powershell
.\run.ps1 smoke-test --skip-pipeline
```

Expected:

```text
smoke test passed
```

- [ ] **Step 2: Run syntax verification**

```powershell
.\code\python\venv\Scripts\python.exe -m py_compile code/python/gui.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run CLI prediction control sample**

```powershell
.\run.ps1 predict-one data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav --expected-decision S01 --speaker-id S01 --split test
```

Expected: JSON output includes:

```json
{
  "decision": "S01",
  "accepted": "True"
}
```

- [ ] **Step 4: Manual GUI acceptance checks**

Run:

```powershell
.\run.ps1 gui
```

Check all of the following:

- Fixed sample `正常通过 S01` returns `ACCEPT S01`.
- Fixed sample `错误口令拒绝` returns `REJECT` with a口令-related reason.
- Fixed sample `非注册用户拒绝` returns `REJECT`.
- `选择 WAV` can load a WAV and plot waveform plus MFCC.
- `录音验证` creates a new WAV under `tmp/gui_recordings`.
- History table records at least 3 predictions.
- `实验结果` page displays baseline metrics and recommended config.
- `临时注册` accepts at least 3 good takes and builds a temporary model.
- Temporary verification adds a history row labeled `temporary`.

- [ ] **Step 5: Final commit**

```powershell
git status --short
git add code/python/requirements.txt run.ps1 code/python/gui.py README.md
git commit -m "feat: add PySide6 voice-lock demo GUI"
```

---

## Risk Controls

- Risk: PySide6 install is large.
  - Control: keep MATLAB frontend intact as fallback.
- Risk: GUI freezes during prediction.
  - Control: run `predict_one()` in `QThread`.
- Risk: live recording fails because microphone permission or device is unavailable.
  - Control: fixed samples remain the primary defense path.
- Risk: temporary registration is unstable in noisy rooms.
  - Control: require at least 3 accepted takes, show quality/SNR, label it as session-only demo.
- Risk: GUI results diverge from official CSV.
  - Control: official verification always calls `predict_one()` using the same model and features.

---

## Self-Review

Spec coverage:

- Fixed sample demonstration: Task 3 and Task 4.
- WAV selection: Task 4.
- Live recording: Task 4.
- `predict_one()` integration: Task 4.
- ACCEPT/REJECT, predicted user, scores, reason: Task 4.
- Waveform and MFCC: Task 3.
- History: Task 4 and Task 7.
- Temporary registration: Task 5.
- Experiment result explanation: Task 6.
- Launch and documentation: Task 1 and Task 8.

Placeholder scan:

- No forbidden placeholder phrases or unspecified test steps are present.
- Every code-changing task names exact files and includes exact commands.

Type consistency:

- `HistoryRow`, `EnrollmentTake`, and `SessionEnrollmentModel` are defined before use.
- `self.session_model`, `self.enrollment_takes`, and `self.enrollment_features` are initialized before page actions use them.
- Official prediction uses existing `load_model()` and `predict_one()`.
- Temporary enrollment uses existing `extract_one_audio_features()`, `lbg_codebook()`, `quantization_distance()`, `passphrase_template_distance()`, `safe_ratio()`, and `select_dims()`.

---

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-06-05-srdp-pyside6-gui.md`.

Two execution options:

1. **Subagent-Driven (recommended)**: dispatch a fresh subagent per task and review between tasks.
2. **Inline Execution**: execute tasks in this session with checkpoints after major pages.
