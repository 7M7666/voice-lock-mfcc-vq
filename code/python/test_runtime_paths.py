import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_paths import resource_root, runtime_data_root


class RuntimePathsTests(unittest.TestCase):
    def test_source_user_data_stays_in_project(self):
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(runtime_data_root(Path("project")), Path("project"))

    def test_frozen_resources_use_bundle_not_working_directory(self):
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", "bundle", create=True):
            self.assertEqual(resource_root(), Path("bundle"))

    def test_frozen_user_data_is_under_local_appdata(self):
        with patch.object(sys, "frozen", True, create=True), patch.dict(os.environ, {"LOCALAPPDATA": "user data"}):
            self.assertEqual(runtime_data_root(Path("bundle")), Path("user data/SRDPVoiceLock"))
