"""Windows launcher; computation and GUI remain in the existing modules."""
import multiprocessing
import os
import sys
from pathlib import Path

from runtime_paths import resource_root, runtime_data_root


def main():
    multiprocessing.freeze_support()
    root = runtime_data_root(resource_root())
    (root / "logs").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(root / "cache/numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    if sys.stdout is None:
        sys.stdout = (root / "logs/startup.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = sys.stdout
    if sys.argv[1:2] == ["--self-test"]:
        import unittest

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        report = Path(sys.argv[2])
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8", buffering=1) as stream:
            print(f"Executable: {sys.executable}\nResources: {resource_root()}\nUser data: {root}\nFrozen: {getattr(sys, 'frozen', False)}", file=stream)
            suite = unittest.defaultTestLoader.loadTestsFromNames(
                ["test_runtime_paths", "test_gui_workflow", "test_user_cleanup"])
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    import gui

    return gui.main()


if __name__ == "__main__":
    raise SystemExit(main())
