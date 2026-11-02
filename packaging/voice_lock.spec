from pathlib import Path
import os
import sys
import hashlib
from PyInstaller.utils.hooks import collect_dynamic_libs

root = Path(SPECPATH).parent
audio = "data/demo_audio/srdp_dataset"
resources = [
    "models/lbg_vq_codebooks.npz",
    "data/features/srdp_features.npz",
    "results/tables/voice_lock_metrics.csv",
    "results/tables/recommended_config.csv",
    "results/calibration_experiments/comparison.csv",
    "results/followup_experiments/comparison.csv",
    "results/followup_experiments/fusion_comparison.csv",
    f"{audio}/S01/corridor/S01_corridor_001.wav",
    f"{audio}/S01/corridor/S01_corridor_002.wav",
    f"{audio}/S01/corridor/S01_corridor_003.wav",
    f"{audio}/S01/dorm/S01_dorm_001.wav",
    f"{audio}/S01/corridor/S01_corridor_004.wav",
    f"{audio}/S02/corridor/S02_corridor_005.wav",
    f"{audio}/S01_wrong/dorm/S01_wrong_003.wav",
    f"{audio}/U01/dorm/U01_corridor_001.wav",
    f"{audio}/S01/lab/S01_lab_004.wav",
]
datas = [(str(root / p), str(Path(p).parent)) for p in resources]
asr_dir = Path(os.environ.get("VOICE_LOCK_BUILD_ASR_DIR", str(root / "models/asr")))
for name, expected in {
    "model.int8.onnx": "3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec",
    "tokens.txt": "4b2d964e18b9cf139b473003b6698fb2ed9a2a5ec55b93daa677b28f578897aa",
}.items():
    source = asr_dir / name
    assert source.is_file(), f"ASR model missing: {source}"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, f"ASR model hash mismatch: {source}"
    datas.append((str(source), "models/asr"))
a = Analysis(
    [str(root / "packaging/windows_entry.py")],
    pathex=[str(root / "code/python")],
    binaries=collect_dynamic_libs("sherpa_onnx"), datas=datas,
    hiddenimports=["test_runtime_paths", "test_gui_workflow", "test_user_cleanup", "test_asr_phrase", "sherpa_onnx.lib._sherpa_onnx"],
    hookspath=[], runtime_hooks=[],
    excludes=["tkinter", "IPython", "pytest"],
    hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg"]}},
    module_collection_mode={"dtw_paths": "py"},
    noarchive=False,
)
allowed = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), Path(os.environ["WINDIR"]).resolve()]
unexpected = [(name, source) for name, source, kind in a.binaries
              if not any(Path(source).resolve().is_relative_to(directory) for directory in allowed)]
assert not unexpected, f"Unexpected developer-machine DLL sources: {unexpected}"
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SRDPVoiceLock",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SRDPVoiceLock")
