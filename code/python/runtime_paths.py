"""Locate bundled resources and keep Windows user data outside the release."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def runtime_data_root(project_root: Path) -> Path:
    if getattr(sys, "frozen", False):
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return local / "SRDPVoiceLock"
    return project_root
