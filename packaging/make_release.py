"""Audit an onedir build and assemble its distributable ZIP; never reads App data."""
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    app, version = Path(sys.argv[1]).resolve(), sys.argv[2]
    root = Path(__file__).resolve().parents[1]
    internal = app / "_internal"
    assert (app / "SRDPVoiceLock.exe").is_file()
    for name in ("models/lbg_vq_codebooks.npz", "data/features/srdp_features.npz"):
        assert sha256(internal / name) == sha256(root / name), name
    for name, expected in {
        "model.int8.onnx": "3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec",
        "tokens.txt": "4b2d964e18b9cf139b473003b6698fb2ed9a2a5ec55b93daa677b28f578897aa",
    }.items():
        assert sha256(internal / "models/asr" / name) == expected, name
    wavs = sorted(internal.rglob("*.wav"))
    assert len(wavs) == 9, [str(p.relative_to(internal)) for p in wavs]
    for wav in wavs:
        assert sha256(wav) == sha256(root / wav.relative_to(internal))
    forbidden = {"app_data", "gui_recordings", "diagnostics"}
    assert not any(forbidden & set(p.relative_to(app).parts) for p in app.rglob("*"))
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        packages[name] = dist.version
        for file in dist.files or []:
            if any("license" in part.lower() or part.lower().startswith("copying") for part in file.parts):
                source = Path(dist.locate_file(file))
                if source.is_file():
                    target = app / "licenses" / name / str(file).replace("..", "_")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copyfile(python_license, app / "licenses/Python-LICENSE.txt")
    info = dict(version=version, python=platform.python_version(), architecture=platform.machine(),
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                source_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
                source_sha256={name: sha256(root / name) for name in (
                    "code/python/gui.py", "code/python/srdp.py", "code/python/asr_phrase.py", "code/python/runtime_paths.py",
                    "code/python/dtw_paths.py", "code/python/speech_frames.py", "code/python/user_cleanup.py",
                    "packaging/windows_entry.py", "packaging/voice_lock.spec")},
                packages=packages, bundled_wav_count=9,
                private_app_data_included=False, layout="PyInstaller onedir",
                user_data="%LOCALAPPDATA%/SRDPVoiceLock")
    (app / "build-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [p for p in sorted(app.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts and p.name != "file-manifest.json"]
    manifest = {p.relative_to(app).as_posix(): sha256(p) for p in files}
    (app / "file-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    target = app.parent / f"SRDPVoiceLock-{version}-windows-x64.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in [*files, app / "file-manifest.json"]:
            archive.write(file, str(Path(app.name) / file.relative_to(app)))
    checksum = target.with_suffix(".zip.sha256")
    checksum.write_text(f"{sha256(target)}  {target.name}\n", encoding="ascii")
    print(json.dumps(dict(zip=str(target), size_bytes=target.stat().st_size,
                         sha256=sha256(target), files=len(files) + 1, wavs=len(wavs))))


if __name__ == "__main__":
    main()
