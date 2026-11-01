import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from user_cleanup import build_user_cleanup, execute_user_cleanup


class UserCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="srdp_cleanup_")
        self.root = Path(self.temporary.name)
        self.users = self.root / "app_data/users"
        self.users.mkdir(parents=True)
        self.model = self.users / "A.npz"
        np.savez(self.model, speaker_id="A")
        np.savez(self.users / "B.npz", speaker_id="B")
        backup = self.users / "backups/old_A.npz"
        backup.parent.mkdir()
        np.savez(backup, speaker_id="A")
        self.history = self.root / "app_data/history.csv"
        self.audio = self.root / "tmp/gui_recordings"
        self.audio.mkdir(parents=True)
        for name in ("own.wav", "shared.wav", "unattributed.wav"):
            (self.audio / name).write_bytes(name.encode())
        external = self.root / "data/formal.wav"
        external.parent.mkdir()
        external.write_bytes(b"formal")
        reports = self.root / "app_data/diagnostics/enrollments"
        reports.mkdir(parents=True)
        for owner, paths in (("A", ("tmp/gui_recordings/own.wav", "tmp/gui_recordings/shared.wav", "data/formal.wav")), ("B", ("tmp/gui_recordings/shared.wav",))):
            (reports / f"{owner}.json").write_text(json.dumps({"speaker_id": owner, "samples": [{"path": p} for p in paths]}), encoding="utf-8")
        snapshots = self.root / "app_data/diagnostics/models"
        snapshots.mkdir()
        np.savez(snapshots / "A_snapshot.npz", speaker_id="A")
        self.write_csv(self.history, ["A", "B"])
        exports = self.root / "results/gui_history"
        exports.mkdir(parents=True)
        self.write_csv(exports / "only_A.csv", ["A"])
        self.write_csv(exports / "mixed.csv", ["A", "B"])

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, path, owners):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "file_path"])
            writer.writeheader()
            writer.writerows({"source": f"用户验证:{owner}", "file_path": "tmp/gui_recordings/own.wav" if owner == "A" else "tmp/gui_recordings/shared.wav"} for owner in owners)

    def plan(self):
        return build_user_cleanup("A", self.model, self.users, self.history, self.root)

    def test_cleanup_removes_linked_data_and_keeps_shared_formal_and_unattributed_audio(self):
        plan = self.plan()
        self.assertTrue(self.model.exists(), "preview must not delete")
        self.assertEqual(plan.audio_files, [self.audio / "own.wav"])
        self.assertEqual(len(plan.retained_audio), 2)
        execute_user_cleanup(plan)
        self.assertFalse(self.model.exists())
        self.assertFalse((self.users / "backups/old_A.npz").exists())
        self.assertFalse((self.audio / "own.wav").exists())
        self.assertTrue((self.audio / "shared.wav").exists())
        self.assertTrue((self.audio / "unattributed.wav").exists())
        self.assertTrue((self.root / "data/formal.wav").exists())
        self.assertTrue((self.users / "B.npz").exists())
        self.assertFalse((self.root / "app_data/diagnostics/enrollments/A.json").exists())
        self.assertTrue((self.root / "app_data/diagnostics/enrollments/B.json").exists())
        self.assertFalse((self.root / "app_data/diagnostics/models/A_snapshot.npz").exists())
        self.assertFalse((self.root / "results/gui_history/only_A.csv").exists())
        for path in (self.history, self.root / "results/gui_history/mixed.csv"):
            with path.open(encoding="utf-8-sig") as handle:
                self.assertEqual([r["source"] for r in csv.DictReader(handle)], ["用户验证:B"])

    def test_changed_file_aborts_entire_plan_before_first_deletion(self):
        plan = self.plan()
        self.history.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已改变"):
            execute_user_cleanup(plan)
        self.assertTrue(self.model.exists())
        self.assertTrue((self.audio / "own.wav").exists())

    def test_user_name_mismatch_cannot_delete_another_model(self):
        with self.assertRaisesRegex(ValueError, "用户名"):
            build_user_cleanup("B", self.model, self.users, self.history, self.root)

    def test_outside_recording_path_is_preserved(self):
        outside = self.root.parent / (self.root.name + "_outside.wav")
        outside.write_bytes(b"outside")
        try:
            report = self.root / "app_data/diagnostics/enrollments/A.json"
            report.write_text(json.dumps({"speaker_id": "A", "samples": [{"path": str(outside)}]}), encoding="utf-8")
            execute_user_cleanup(self.plan())
            self.assertTrue(outside.exists())
        finally:
            outside.unlink()


if __name__ == "__main__":
    unittest.main()
