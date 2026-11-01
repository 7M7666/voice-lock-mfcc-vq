import os
import csv
import tempfile
import unittest
import json
import time
from threading import Event, get_ident
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest

import gui


class GuiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyleSheet(gui.APP_QSS)
        cls.base_model = gui.load_model()
        original_training = [
            gui.PROJECT_ROOT / f"data/demo_audio/srdp_dataset/S01/{environment}/S01_{environment}_00{i}.wav"
            for environment, count in (("corridor", 3), ("dorm", 3), ("lab", 2))
            for i in range(1, count + 1)
        ]
        cls.saved_model = gui.build_session_enrollment_model(
            "测试用户", cls.base_model,
            [gui.extract_one_audio_features(path, cls.base_model).features for path in original_training[:3]],
        )
        cls.recordings = tempfile.TemporaryDirectory(prefix="srdp_gui_enrollment_")
        cls.addClassCleanup(cls.recordings.cleanup)
        cls.training = []
        for original in original_training[:gui.ENROLLMENT_REQUIRED]:
            audio, sr = gui.load_audio(original)
            path = Path(cls.recordings.name) / original.name
            gui.sf.write(path, gui.np.pad(audio, (sr // 10, sr // 10)), sr, subtype="FLOAT")
            cls.training.append(path)
        cls.takes = [gui.assess_enrollment_take(path, cls.base_model) for path in cls.training]
        assert all(take.accepted for take, _ in cls.takes), "registration fixtures must pass quality admission"
        cls.sample = gui.PROJECT_ROOT / gui.DEMO_SAMPLES["正常通过 S01"]["wav"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="srdp_gui_workflow_")
        self.root = Path(self.directory.name)
        self.model_patch = patch.object(gui, "load_model", return_value=self.base_model)
        self.model_patch.start()
        self.critical_patch = patch.object(QMessageBox, "critical")
        self.unexpected_critical = self.critical_patch.start()
        self.window = gui.SrdpVoiceLockWindow(self.root / "users", self.root / "history.csv")

    def tearDown(self):
        self.window.close()
        self.wait_session()
        self.app.processEvents()
        self.model_patch.stop()
        self.critical_patch.stop()
        self.directory.cleanup()
        self.assertFalse(self.unexpected_critical.called, f"unexpected error dialog: {self.unexpected_critical.call_args}")

    def wait_session(self):
        deadline = time.monotonic() + 20
        while self.window.session_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(self.window.session_thread, "session worker did not finish")

    def seed_user(self, name):
        path = gui.user_model_path(name, self.root / "users")
        gui.save_user_model(replace(self.saved_model, speaker_id=name), path)
        self.window.refresh_saved_users(path)
        return path

    def record_registration(self, name, count=gui.ENROLLMENT_REQUIRED):
        self.window.stack.setCurrentIndex(1)
        self.window.enroll_name_input.setText(name)
        for path, take in zip(self.training[:count], self.takes[:count]):
            with patch.object(gui, "record_audio_to_temp", return_value=path), patch.object(
                gui, "assess_enrollment_take", return_value=take
            ):
                self.window.record_enrollment_take()
                self.wait_session()

    def test_empty_app_starts_with_verification_and_registration_entry(self):
        self.assertEqual(self.window.enroll_duration_spin.minimum(), 1.2)
        self.assertEqual(self.window.enroll_duration_spin.maximum(), 5.0)
        self.assertEqual(self.window.stack.currentIndex(), 0)
        self.assertEqual(self.window.nav_buttons[0].text(), "声音验证")
        self.assertTrue(self.window.nav_buttons[0].isChecked())
        self.assertFalse(self.window.verify_empty_label.isHidden())
        self.assertFalse(self.window.temp_verify_button.isEnabled())
        self.window.go_register_button.click()
        self.assertEqual(self.window.stack.currentIndex(), 1)
        self.assertTrue(self.window.nav_buttons[1].isChecked())

    def test_complete_registration_saves_and_returns_to_verification(self):
        self.record_registration("新用户")
        self.assertEqual(self.window.accepted_enrollment_count(), 4)
        self.assertTrue(self.window.enroll_build_button.isEnabled())
        self.window.enroll_build_button.click()
        self.wait_session()
        self.assertTrue(gui.user_model_path("新用户", self.root / "users").exists())
        self.assertEqual(self.window.stack.currentIndex(), 0)
        self.assertEqual(self.window.session_model.speaker_id, "新用户")
        self.assertTrue(self.window.temp_verify_button.isEnabled())
        self.assertEqual(self.window.accepted_enrollment_count(), 0)
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        report_path = next((self.root / "diagnostics/enrollments").glob("*.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["model_fingerprint"], gui.session_model_fingerprint(self.window.session_model))
        self.assertEqual(len(report["samples"]), gui.ENROLLMENT_REQUIRED)
        for sample, (take, _) in zip(report["samples"], self.takes):
            self.assertTrue(sample["audio_sha256"])
            self.assertEqual(sample["diagnostics"], take.diagnostics)
            self.assertEqual(sample["estimated_snr_db"], take.estimated_snr_db)
            self.assertEqual(sample["quality_score"], take.quality_score)
            self.assertIn("voice_score", sample["self_check"])
            self.assertIn("passphrase_score", sample["self_check"])
            self.assertIn("consistency_check", sample)
        self.assertIn("还需录音验证", self.window.temp_verify_reason_label.text())
        self.assertFalse(any("注册后独立检查" in group.title() for group in self.window.findChildren(gui.QGroupBox)))

    def test_consistency_warning_precedes_model_build_and_allows_explicit_continue(self):
        self.record_registration("新用户")
        checks = [{"voice_distance": 0., "passphrase_distance": 0., "flagged": False} for _ in range(gui.ENROLLMENT_REQUIRED)]
        checks[-1] = {"voice_distance": 10., "passphrase_distance": 12., "flagged": True}
        with patch.object(gui, "enrollment_consistency_check", return_value=checks), \
             patch.object(QMessageBox, "question", return_value=QMessageBox.No) as question, \
             patch.object(gui, "build_session_enrollment_model") as build:
            self.window.complete_registration()
            self.wait_session()
        build.assert_not_called()
        self.assertEqual(question.call_args.args[-1], QMessageBox.No)
        self.assertIn("第4条录音", question.call_args.args[2])
        self.assertIn("差异较大", self.window.enroll_take_list.item(3).text())
        self.assertIn("不能据此判断说错口令", self.window.enroll_quality_label.text())
        self.assertEqual(self.window.accepted_enrollment_count(), 4)
        self.assertFalse(gui.user_model_path("新用户", self.root / "users").exists())
        with patch.object(gui, "enrollment_consistency_check", return_value=checks), \
             patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.complete_registration()
            self.wait_session()
        self.assertEqual(len(self.window.session_model.passphrase_templates), 4)
        report = json.loads(next((self.root / "diagnostics/enrollments").glob("*.json")).read_text(encoding="utf-8"))
        self.assertTrue(report["samples"][3]["consistency_check"]["flagged"])
        self.assertTrue(report["samples"][3]["accepted"])

    def test_cancel_consistency_does_not_build_or_save_a_model(self):
        self.record_registration("新用户")
        started, released = Event(), Event()
        def compare(*args):
            started.set()
            released.wait(3)
            return []
        with patch.object(gui, "enrollment_consistency_check", side_effect=compare), \
             patch.object(gui, "build_session_enrollment_model") as build:
            self.window.complete_registration()
            deadline = time.monotonic() + 3
            while not started.is_set() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.assertTrue(started.is_set())
            self.window.enroll_cancel_button.click()
            released.set()
            self.wait_session()
        build.assert_not_called()
        self.assertEqual(self.window.accepted_enrollment_count(), 4)
        self.assertFalse(gui.user_model_path("新用户", self.root / "users").exists())

    def test_three_accepted_takes_cannot_register_even_by_direct_call(self):
        self.record_registration("新用户", count=3)
        self.assertEqual(self.window.accepted_enrollment_count(), 3)
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        self.assertTrue(self.window.enroll_record_button.isEnabled())
        self.assertEqual(self.window.enroll_progress.value(), 3)
        self.assertEqual(self.window.enroll_progress.maximum(), 4)
        self.assertEqual(self.window.enroll_record_button.text(), "录制第 4 条语音")
        self.assertIn("还需要1条", self.window.enroll_status_label.text())
        with patch.object(QMessageBox, "warning") as warning, patch.object(gui, "build_session_enrollment_model") as build:
            self.window.complete_registration()
        warning.assert_called_once_with(self.window, "注册样本不足", "至少需要 4 条合格注册样本。")
        build.assert_not_called()
        self.assertIsNone(self.window.session_thread)
        self.assertFalse(gui.user_model_path("新用户", self.root / "users").exists())

    def test_four_accepted_takes_stop_recording_until_reset(self):
        self.record_registration("新用户")
        self.window.set_audio_busy(True)
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        self.window.set_audio_busy(False)
        self.assertTrue(self.window.enroll_build_button.isEnabled())
        self.assertFalse(self.window.enroll_record_button.isEnabled())
        self.assertEqual(self.window.enroll_record_button.text(), "已录满4条")
        self.assertEqual(self.window.enroll_progress.value(), 4)
        self.assertIn("已达到4条", self.window.enroll_status_label.text())
        with patch.object(gui, "record_audio_to_temp") as record:
            self.window.enroll_record_button.click()
            self.window.record_enrollment_take()
        record.assert_not_called()
        self.assertIsNone(self.window.session_thread)
        self.assertEqual(self.window.accepted_enrollment_count(), 4)
        self.window.enroll_name_input.clear()
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        self.assertFalse(self.window.enroll_record_button.isEnabled())
        self.window.reset_temporary_enrollment()
        self.assertEqual(self.window.accepted_enrollment_count(), 0)
        self.assertEqual(self.window.enroll_progress.value(), 0)
        self.assertTrue(self.window.enroll_record_button.isEnabled())
        self.assertEqual(self.window.enroll_record_button.text(), "录制第 1 条语音")
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        self.assertIn("4 条合格", self.window.enroll_status_label.text())

    def test_registration_warning_is_visible_and_one_take_can_be_removed(self):
        self.record_registration("新用户")
        take = self.window.enrollment_takes[1]
        warning = "检测到与主体分离的短促声音，建议重录"
        self.window.enrollment_takes[1] = replace(take, reason=f"合格；{warning}")
        self.window.enrollment_consistency = {2: {"voice_distance": 10., "passphrase_distance": 10., "flagged": True}}
        self.window.refresh_enrollment_list()
        self.assertIn(warning, self.window.enroll_take_list.item(1).text())
        self.window.enroll_take_list.setCurrentRow(1)
        self.assertIn(warning, self.window.enroll_quality_label.text())
        self.assertIn("持续有效声音", self.window.enroll_quality_label.text())
        self.assertIn(f"估计信噪比：{take.estimated_snr_db:.2f} dB", self.window.enroll_quality_label.text())
        self.assertIn(f"质量评分：{take.quality_score:.3f}", self.window.enroll_quality_label.text())
        self.assertIn(f"估计信噪比 {take.estimated_snr_db:.2f} dB", self.window.enroll_take_list.item(1).text())
        self.assertIn(f"质量评分 {take.quality_score:.3f}", self.window.enroll_take_list.item(1).text())
        self.window.set_audio_busy(True)
        self.assertFalse(self.window.enroll_remove_button.isEnabled())
        self.window.remove_enrollment_take()
        self.assertEqual(len(self.window.enrollment_takes), gui.ENROLLMENT_REQUIRED)
        self.window.set_audio_busy(False)
        self.window.enroll_remove_button.click()
        self.assertEqual(self.window.enrollment_consistency, {})
        self.assertEqual([t.path for t in self.window.enrollment_takes], self.training[:1] + self.training[2:])
        gui.np.testing.assert_array_equal(self.window.enrollment_features[1], self.takes[2][1])
        self.assertEqual(self.window.accepted_enrollment_count(), gui.ENROLLMENT_REQUIRED - 1)
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        self.assertTrue(self.window.enroll_record_button.isEnabled())
        self.assertEqual(self.window.enroll_progress.value(), gui.ENROLLMENT_REQUIRED - 1)
        self.assertTrue(take.path.exists())

    def test_silent_take_is_not_counted_and_remains_in_registration_report(self):
        silence = self.root / "silence.wav"
        gui.sf.write(silence, gui.np.zeros(48000), 16000)
        with patch.object(gui, "record_audio_to_temp", return_value=silence), patch.object(QMessageBox, "critical") as dialog:
            self.window.record_enrollment_take()
            self.wait_session()
        dialog.assert_not_called()
        self.assertEqual(self.window.accepted_enrollment_count(), 0)
        self.assertEqual(self.window.enroll_progress.value(), 0)
        self.assertEqual(self.window.enroll_record_button.text(), "录制第 1 条语音")
        self.assertIn("静音", self.window.enroll_take_list.item(0).text())
        self.assertIn("不合格", self.window.enroll_quality_label.text())
        self.assertIn("估计信噪比：未计算", self.window.enroll_quality_label.text())
        self.assertIn("质量评分：未计算", self.window.enroll_quality_label.text())
        self.record_registration("新用户")
        self.window.enroll_build_button.click()
        self.wait_session()
        report_path = next((self.root / "diagnostics/enrollments").glob("*.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(len(report["samples"]), gui.ENROLLMENT_REQUIRED + 1)
        self.assertFalse(report["samples"][0]["accepted"])
        self.assertIsNone(report["samples"][0]["quality_score"])
        self.assertNotIn("self_check", report["samples"][0])
        self.assertTrue(all("self_check" in sample for sample in report["samples"][1:]))

    def test_rejected_fourth_take_can_be_removed_and_rerecorded_without_counting(self):
        self.record_registration("新用户", count=3)
        path = self.training[-1]
        before = path.read_bytes()
        extracted = gui.extract_one_audio_features(path, self.base_model)
        with patch.object(gui, "record_audio_to_temp", return_value=path), patch.object(
            gui, "extract_one_audio_features", return_value=replace(extracted, estimated_snr_db=9.0, quality_score=.2)
        ):
            self.window.record_enrollment_take()
            self.wait_session()
        self.assertEqual(len(self.window.enrollment_takes), 4)
        self.assertEqual(self.window.accepted_enrollment_count(), 3)
        self.assertEqual(self.window.enroll_progress.value(), 3)
        self.assertFalse(self.window.enroll_build_button.isEnabled())
        self.assertTrue(self.window.enroll_record_button.isEnabled())
        self.assertIn("不合格", self.window.enroll_take_list.item(3).text())
        self.assertIn("不低于10 dB", self.window.enroll_take_list.item(3).text())
        self.assertIn("估计信噪比：9.00 dB", self.window.enroll_quality_label.text())
        self.assertIn("质量评分：0.200", self.window.enroll_quality_label.text())
        self.window.enroll_remove_button.click()
        self.assertEqual(len(self.window.enrollment_takes), 3)
        self.assertEqual(self.window.enroll_progress.value(), 3)
        self.assertEqual(path.read_bytes(), before)
        with patch.object(gui, "record_audio_to_temp", return_value=path):
            self.window.record_enrollment_take()
            self.wait_session()
        self.assertEqual(self.window.accepted_enrollment_count(), 4)
        self.assertEqual(self.window.enroll_progress.value(), 4)
        self.assertTrue(self.window.enroll_build_button.isEnabled())
        self.assertFalse(self.window.enroll_record_button.isEnabled())

    def test_user_selection_loads_immediately_without_erasing_registration(self):
        first = self.seed_user("甲")
        second = self.seed_user("乙")
        self.window.enroll_name_input.setText("尚未完成")
        take, features = self.takes[0]
        self.window.enrollment_takes.append(take)
        self.window.enrollment_features.append(features)
        self.window.refresh_enrollment_list()
        self.window.saved_user_combo.setCurrentIndex(self.window.saved_user_combo.findData(str(first)))
        self.assertEqual(self.window.session_model.speaker_id, "甲")
        self.assertEqual(self.window.accepted_enrollment_count(), 1)
        self.assertEqual(self.window.enroll_name_input.text(), "尚未完成")
        self.window.reset_temporary_enrollment()
        self.assertEqual(self.window.session_model.speaker_id, "甲")
        self.assertTrue(second.exists())

    def test_cancel_overwrite_keeps_registration_and_active_user(self):
        path = self.seed_user("已有用户")
        before = path.read_bytes()
        self.record_registration("已有用户")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.window.complete_registration()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.window.stack.currentIndex(), 1)
        self.assertEqual(self.window.accepted_enrollment_count(), gui.ENROLLMENT_REQUIRED)
        self.assertEqual(self.window.session_model.speaker_id, "已有用户")

    def test_delete_user_requires_confirmation_and_is_disabled_during_recording(self):
        path = self.seed_user("测试用户")
        before = path.read_bytes()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.window.delete_user_button.click()
        self.assertEqual(path.read_bytes(), before)
        self.window.set_audio_busy(True)
        self.assertFalse(self.window.delete_user_button.isEnabled())
        with patch.object(QMessageBox, "question") as question:
            self.window.delete_saved_user()
        question.assert_not_called()
        self.window.set_audio_busy(False)

    def test_full_cleanup_requires_checkbox_and_confirmation_and_clears_matching_history(self):
        path = self.seed_user("测试用户")
        self.window.verify_temporary_audio(self.sample)
        self.assertFalse(self.window.delete_user_data_checkbox.isChecked())
        self.window.delete_user_data_checkbox.setChecked(True)
        before = path.read_bytes()
        with patch.object(gui, "PROJECT_ROOT", self.root), patch.object(QMessageBox, "question", return_value=QMessageBox.No) as question:
            self.window.delete_saved_user()
        self.assertIn("清理", question.call_args.args[2])
        self.assertEqual(path.read_bytes(), before)
        with patch.object(gui, "PROJECT_ROOT", self.root), patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.delete_saved_user()
        self.assertFalse(path.exists())
        self.assertEqual(self.window.history, [])
        self.assertEqual(self.window.full_history_table.rowCount(), 0)
        self.assertFalse(self.window.delete_user_data_checkbox.isChecked())
        self.assertTrue(self.sample.exists(), "formal WAV must never be removed")

    def test_delete_user_keeps_history_and_allows_same_name_registration(self):
        path = self.seed_user("测试用户")
        other = self.seed_user("其他用户")
        self.window.saved_user_combo.setCurrentIndex(self.window.saved_user_combo.findData(str(path)))
        self.window.verify_temporary_audio(self.sample)
        history_before = (self.root / "history.csv").read_bytes()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.delete_user_button.click()
        self.assertFalse(path.exists())
        self.assertTrue(other.exists())
        self.assertEqual(self.window.session_model.speaker_id, "其他用户")
        self.assertEqual((self.root / "history.csv").read_bytes(), history_before)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.delete_user_button.click()
        self.assertIsNone(self.window.session_model)
        self.assertFalse(self.window.temp_verify_button.isEnabled())
        self.assertFalse(self.window.delete_user_button.isEnabled())
        self.assertFalse(self.window.verify_empty_label.isHidden())
        self.record_registration("测试用户")
        self.window.enroll_build_button.click()
        self.wait_session()
        self.assertTrue(path.exists())
        self.assertEqual(self.window.session_model.speaker_id, "测试用户")

    def test_verify_records_restore_but_demo_records_stay_separate(self):
        self.seed_user("测试用户")
        with patch.object(gui, "record_audio_to_temp", return_value=self.sample):
            self.window.temp_verify_button.click()
            self.wait_session()
        self.assertIn("验证通过", self.window.temp_verify_result_label.text())
        self.assertEqual(self.window.full_history_table.rowCount(), 1)
        self.assertEqual(self.window.full_history_table.item(0, 1).text(), "测试用户")
        self.assertEqual(self.window.full_history_table.item(0, 2).text(), "通过")
        self.assertEqual(self.window.history_table.rowCount(), 0)
        result = gui.predict_one(self.sample, model=self.base_model)
        self.window.on_prediction_finished(result, "正常通过 S01", 1.0)
        self.assertEqual(self.window.history_table.rowCount(), 1)
        self.assertEqual(self.window.full_history_table.rowCount(), 1)
        self.window.close()
        self.window = gui.SrdpVoiceLockWindow(self.root / "users", self.root / "history.csv")
        self.assertEqual(self.window.session_model.speaker_id, "测试用户")
        self.assertTrue(self.window.temp_verify_button.isEnabled())
        self.assertEqual(self.window.full_history_table.rowCount(), 1)
        self.assertEqual(self.window.history_table.rowCount(), 0)

    def test_user_history_keeps_audio_model_and_unrounded_diagnostics(self):
        self.seed_user("测试用户")
        self.window.verify_temporary_audio(self.sample)
        row = gui.load_history(self.root / "history.csv")[0]
        self.assertEqual(gui.project_path(row.file_path), self.sample)
        self.assertEqual(len(row.model_fingerprint), 64)
        self.assertTrue(row.voice_threshold)
        self.assertTrue(row.passphrase_distance)
        self.assertGreater(float(row.voiced_end_sec), float(row.voiced_start_sec))
        self.assertEqual(row.vad_mode, "trim")
        snapshot = self.root / "diagnostics/models" / f"{row.model_fingerprint}.npz"
        model = gui.load_user_model(snapshot)
        self.assertEqual(gui.session_model_fingerprint(model), row.model_fingerprint)
        replay = gui.predict_with_session_model(self.sample, model)
        self.assertEqual(row.voice_score, replay["voice_score"])
        self.assertEqual(row.passphrase_score, replay["passphrase_score"])
        self.window.full_history_table.selectRow(0)
        self.assertIn("有效语音", self.window.history_details_label.text())
        with patch.object(QMessageBox, "information") as dialog:
            self.window.replay_history_record()
        self.assertIn("复核", dialog.call_args.args[1])
        self.assertEqual(len(self.window.history), 1)
        with patch.object(gui, "PROJECT_ROOT", self.root), patch.object(QMessageBox, "information"):
            self.window.export_history()
        exported = next((self.root / "results/gui_history").glob("*.csv"))
        with exported.open(encoding="utf-8-sig", newline="") as handle:
            exported_row = next(csv.DictReader(handle))
        self.assertEqual(exported_row["model_fingerprint"], row.model_fingerprint)
        self.assertEqual(exported_row["passphrase_distance"], row.passphrase_distance)

    def test_replay_reports_changed_audio_and_keeps_history(self):
        self.seed_user("测试用户")
        audio = self.root / "sample.wav"
        audio.write_bytes(self.sample.read_bytes())
        self.window.verify_temporary_audio(audio)
        audio.write_bytes(b"changed audio")
        self.window.full_history_table.selectRow(0)
        with patch.object(QMessageBox, "warning") as dialog:
            self.window.replay_history_record()
        self.assertIn("内容已改变", dialog.call_args.args[2])
        self.assertEqual(len(self.window.history), 1)

    def test_silence_is_recorded_as_input_rejection_without_voice_mismatch_message(self):
        self.seed_user("测试用户")
        path = self.root / "silence.wav"
        gui.sf.write(path, gui.np.zeros(48000), 16000)
        self.window.verify_temporary_audio(path)
        row = gui.load_history(self.root / "history.csv")[0]
        self.assertEqual(row.rejection_stage, "input")
        self.assertEqual(row.voice_score, "-")
        self.assertNotIn("声纹未通过", self.window.temp_verify_reason_label.text())
        self.assertIn("静音", self.window.temp_verify_reason_label.text())

    def test_old_history_loads_without_fabricating_diagnostics(self):
        path = self.root / "old.csv"
        path.write_text("time_text,source,decision,pred_speaker,voice_score,passphrase_score,fusion_score,reject_reason,latency_ms\n2026-10-29 17:00:00,用户验证:旧用户,REJECT,旧用户,0.9,1.2,-,口令DTW距离超过阈值,100\n", encoding="utf-8")
        rows = gui.load_history(path)
        self.assertEqual(rows[0].decision, "REJECT")
        self.assertEqual(rows[0].file_path, "")
        self.assertEqual(rows[0].model_fingerprint, "")

    def test_phrase_status_describes_template_matching_even_when_voice_fails(self):
        self.seed_user("测试用户")
        for voice, phrase in (("False", "False"), ("False", "True"), ("True", "False"), ("True", "True")):
            with self.subTest(voice=voice, phrase=phrase):
                result = {"decision": "REJECT" if "False" in (voice, phrase) else "测试用户",
                    "voice_passed": voice, "passphrase_passed": phrase, "voice_score": "1.100000",
                    "passphrase_score": "1.234567", "passphrase_limit": "1.00"}
                with patch.object(self.window, "add_history_row"):
                    self.window.show_user_verification(result, 1.)
                text = self.window.temp_verify_reason_label.text()
                self.assertIn("声纹" + ("通过" if voice == "True" else "未通过"), text)
                self.assertIn("口令模板匹配" + ("通过" if phrase == "True" else "未通过"), text)
                self.assertIn("1.235", self.window.temp_verify_score_label.text())
                self.assertIn("未识别成文字", gui.PHRASE_EXPLANATION)

    def test_model_fingerprint_changes_with_parameters_and_survives_save_load(self):
        model = self.saved_model
        self.assertNotEqual(gui.session_model_fingerprint(model), gui.session_model_fingerprint(replace(model, voice_threshold=model.voice_threshold + .01)))
        path = self.root / "roundtrip.npz"
        gui.save_user_model(model, path)
        self.assertEqual(gui.session_model_fingerprint(model), gui.session_model_fingerprint(gui.load_user_model(path)))

    def test_failed_recording_clears_previous_result_and_restores_controls(self):
        self.seed_user("测试用户")
        self.window.verify_temporary_audio(self.sample)
        with patch.object(gui, "record_audio_to_temp", side_effect=RuntimeError("无麦克风")), patch.object(
            QMessageBox, "critical"
        ) as dialog:
            self.window.verify_temporary_user_by_recording()
            self.wait_session()
        dialog.assert_called_once()
        self.assertEqual(self.window.temp_verify_result_label.text(), "本次录音失败")
        self.assertEqual(self.window.temp_verify_score_label.text(), "分数: -")
        self.assertEqual(len(self.window.history), 1)
        self.assertFalse(self.window.audio_busy)
        self.assertTrue(self.window.temp_verify_button.isEnabled())
        self.assertTrue(all(button.isEnabled() for button in self.window.nav_buttons))

    def test_recording_locks_user_and_navigation_until_finished(self):
        self.seed_user("测试用户")
        released = Event()
        identities = []
        timer_fired = []

        def record(*args, **kwargs):
            identities.append(get_ident())
            kwargs["progress"](.5, .25)
            if not released.wait(3):
                raise RuntimeError("GUI timer did not release recording")
            return self.sample

        def tick():
            timer_fired.append(True)
            released.set()

        with patch.object(gui, "record_audio_to_temp", side_effect=record):
            self.window.verify_temporary_user_by_recording()
            self.assertTrue(self.window.audio_busy)
            self.assertFalse(self.window.saved_user_combo.isEnabled())
            self.assertFalse(self.window.temp_verify_button.isEnabled())
            self.assertTrue(all(not button.isEnabled() for button in self.window.nav_buttons))
            QTimer.singleShot(0, tick)
            self.wait_session()
        self.assertTrue(timer_fired)
        self.assertNotEqual(identities, [get_ident()])
        self.assertIn("25%", self.window.verify_level_label.text())
        self.assertTrue(self.window.saved_user_combo.isEnabled())
        self.assertTrue(self.window.temp_verify_button.isEnabled())

    def test_cancel_verification_during_scoring_discards_late_result(self):
        self.seed_user("测试用户")
        original = gui.predict_with_session_model(self.sample, self.window.session_model)
        started, released = Event(), Event()
        def predict(*args):
            started.set()
            released.wait(3)
            return original
        with patch.object(gui, "record_audio_to_temp", return_value=self.sample), patch.object(gui, "predict_with_session_model", side_effect=predict):
            self.window.verify_temporary_user_by_recording()
            deadline = time.monotonic() + 3
            while not started.is_set() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.assertTrue(started.is_set())
            self.window.verify_cancel_button.click()
            released.set()
            self.wait_session()
        self.assertEqual(len(self.window.history), 0)
        self.assertEqual(self.window.temp_verify_score_label.text(), "分数: -")
        self.assertIn("取消", self.window.temp_verify_result_label.text())
        self.assertTrue(self.window.temp_verify_button.isEnabled())

    def test_cancel_registration_does_not_overwrite_user_or_save_report(self):
        path = self.seed_user("已有用户")
        before = path.read_bytes()
        self.record_registration("已有用户")
        started, released = Event(), Event()
        def build(*args):
            started.set()
            released.wait(3)
            return replace(self.saved_model, speaker_id="已有用户")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), \
             patch.object(gui, "enrollment_consistency_check", return_value=[{"voice_distance": 0., "passphrase_distance": 0., "flagged": False}] * gui.ENROLLMENT_REQUIRED), \
             patch.object(gui, "build_session_enrollment_model", side_effect=build):
            self.window.complete_registration()
            deadline = time.monotonic() + 3
            while not started.is_set() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.assertTrue(started.is_set())
            self.window.enroll_cancel_button.click()
            released.set()
            self.wait_session()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.window.accepted_enrollment_count(), gui.ENROLLMENT_REQUIRED)
        self.assertFalse((self.root / "diagnostics/enrollments").exists())

    def test_close_during_recording_cancels_then_finishes_thread(self):
        self.seed_user("测试用户")
        def record(*args, **kwargs):
            kwargs["cancel"].wait(3)
            return self.sample
        self.window.show()
        with patch.object(gui, "record_audio_to_temp", side_effect=record):
            self.window.verify_temporary_user_by_recording()
            self.window.close()
            self.wait_session()
            QTest.qWait(20)
        self.assertFalse(self.window.isVisible())
        self.assertEqual(len(self.window.history), 0)

    def test_recording_cancel_stops_device_and_does_not_save_partial_audio(self):
        cancelled = Event()
        def progress(fraction, peak):
            cancelled.set()
        with patch.object(gui, "TEMP_AUDIO_DIR", self.root / "audio"), patch.object(gui.sd, "rec", return_value=gui.np.zeros((48000, 1))), \
             patch.object(gui.sd, "get_stream") as stream, patch.object(gui.sd, "stop") as stop, patch.object(gui.sf, "write") as write:
            stream.return_value.active = True
            with self.assertRaises(gui.TaskCancelled):
                gui.record_audio_to_temp("cancelled", 3.0, cancel=cancelled, progress=progress)
            stop.assert_called_once()
            write.assert_not_called()

    def test_experiment_table_explains_saved_results_in_chinese(self):
        self.assertEqual(self.window.metrics_table.rowCount(), 3)
        self.assertEqual(self.window.metrics_table.horizontalHeaderItem(2).text(), "正常用户通过")
        self.assertIn("不代表", self.window.experiment_note.text())
        self.assertIn("不会训练", self.window.experiment_note.text())


if __name__ == "__main__":
    unittest.main()
