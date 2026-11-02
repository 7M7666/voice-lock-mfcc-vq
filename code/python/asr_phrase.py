"""Offline Mandarin passphrase check for the personal verification flow."""
from __future__ import annotations
import hashlib
import os
import sys
import threading
import unicodedata
from pathlib import Path
import numpy as np
from runtime_paths import resource_root, runtime_data_root

EXPECTED_HASHES = {
    'model.int8.onnx': '3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec',
    'tokens.txt': '4b2d964e18b9cf139b473003b6698fb2ed9a2a5ec55b93daa677b28f578897aa',
}
_lock = threading.Lock()
_recognizer = None

def normalize_text(value: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKC', value)
                   if not c.isspace() and not unicodedata.category(c).startswith(('P', 'S'))).lower()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def model_directory() -> Path:
    override = os.environ.get('VOICE_LOCK_ASR_DIR')
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, 'frozen', False):
        return resource_root() / 'models/asr'
    return runtime_data_root(resource_root()) / 'models/asr'

def _load_recognizer():
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    folder = model_directory()
    for name, expected in EXPECTED_HASHES.items():
        path = folder / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f'离线语音识别模型缺失或校验不通过：{path}。请安装指定模型后重试。')
    try:
        import sherpa_onnx
        _recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(folder/'model.int8.onnx'), tokens=str(folder/'tokens.txt'),
            num_threads=2, sample_rate=16000, feature_dim=80, decoding_method='greedy_search')
    except Exception as exc:
        raise RuntimeError('离线语音识别加载失败，请检查依赖与模型文件。') from exc
    return _recognizer

def recognize_fixed_phrase(path: Path) -> str:
    # Serialize loading and inference because the GUI may perform replay and verification.
    from srdp import load_audio
    y, sr = load_audio(path)
    with _lock:
        try:
            rec = _load_recognizer()
            stream = rec.create_stream()
            stream.accept_waveform(sr, np.asarray(y, dtype=np.float32))
            rec.decode_stream(stream)
            return stream.result.text
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError('离线语音识别执行失败，本次没有有效判定。') from exc
