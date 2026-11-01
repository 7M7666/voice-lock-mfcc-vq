"""Preview and remove explicitly linked App data; never guess ownership of WAVs."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class UserCleanupPlan:
    speaker: str
    model_path: Path
    delete_files: dict[Path, str] = field(default_factory=dict)
    csv_updates: dict[Path, tuple[list[str], list[dict], str]] = field(default_factory=dict)
    audio_files: list[Path] = field(default_factory=list)
    retained_audio: list[Path] = field(default_factory=list)
    removed_rows: int = 0

    def summary(self):
        return (f"将删除{len(self.audio_files)}个关联临时录音、{len(self.delete_files) - len(self.audio_files)}个模型/报告/导出文件，"
            f"移除{self.removed_rows}条历史/导出记录。\n"
            f"{len(self.retained_audio)}个共用或非临时目录的录音保留。\n"
            "范围：App用户模型及备份、注册报告、模型快照、验证记录和导出记录。\n"
            "无归属旧录音、独立实验目录及外部副本不在此清理范围；文件删除后无法通过App恢复。")


def build_user_cleanup(speaker, model_path, user_model_dir, history_path, project_root):
    root = Path(project_root).resolve()
    user_dir, history = Path(user_model_dir).resolve(), Path(history_path).resolve()
    app_dir = history.parent
    audio_dir = (root / "tmp/gui_recordings").resolve()
    exports = (root / "results/gui_history").resolve()
    active = Path(model_path).resolve()
    if not active.is_relative_to(user_dir) or not active.is_file():
        raise ValueError("所选模型不在用户目录中或已不存在")
    roots = (user_dir, app_dir / "diagnostics/enrollments", app_dir / "diagnostics/models", audio_dir, exports)
    plan = UserCleanupPlan(speaker, active)

    def schedule(path):
        path = path.resolve()
        if not any(path.is_relative_to(p.resolve()) for p in roots):
            raise ValueError(f"清理路径越界：{path}")
        if path.is_file():
            plan.delete_files[path] = file_hash(path)

    owned_audio, other_audio = set(), set()

    def reference(value, owner):
        if not value:
            return
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        (owned_audio if owner == speaker else other_audio).add(path)

    for directory in (user_dir, app_dir / "diagnostics/models"):
        for path in directory.rglob("*.npz"):
            with np.load(path, allow_pickle=False) as archive:
                owner = str(archive["speaker_id"].item()) if "speaker_id" in archive else ""
            if owner == speaker:
                schedule(path)
    if active not in plan.delete_files:
        raise ValueError("所选模型的用户名与清理目标不一致")
    for path in (app_dir / "diagnostics/enrollments").glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        owner = report.get("speaker_id", "")
        for sample in report.get("samples", []):
            reference(sample.get("path", ""), owner)
        if owner == speaker:
            schedule(path)
    for path in [history, *exports.glob("*.csv")]:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = list(reader.fieldnames or []), list(reader)
        kept = []
        for row in rows:
            owner = row.get("source", "").removeprefix("用户验证:")
            reference(row.get("file_path", ""), owner)
            if row.get("source") == f"用户验证:{speaker}":
                plan.removed_rows += 1
            else:
                kept.append(row)
        if len(kept) != len(rows):
            if not kept and path.resolve() != history:
                schedule(path)
            else:
                plan.csv_updates[path.resolve()] = (columns, kept, file_hash(path))
    for path in sorted(owned_audio):
        if path in other_audio or not path.is_relative_to(audio_dir):
            plan.retained_audio.append(path)
        elif path.is_file():
            schedule(path)
            plan.audio_files.append(path)
    return plan


def execute_user_cleanup(plan):
    # Validate the complete preview before changing any file.
    for path, expected in plan.delete_files.items():
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("确认期间待删除文件已改变，请重新预览")
    for path, (_, _, expected) in plan.csv_updates.items():
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("确认期间验证记录已改变，请重新预览")
    # Keep ownership reports until audio removal succeeds, so a failed cleanup can be retried.
    ordered = [*plan.audio_files, *[p for p in plan.delete_files if p not in plan.audio_files and p != plan.model_path]]
    for path in ordered:
        path.unlink()
    for path, (columns, rows, _) in plan.csv_updates.items():
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    plan.model_path.unlink()
