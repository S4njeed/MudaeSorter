"""Entry point for Mudae Power Tools Pro v3.0."""

from __future__ import annotations

import os
import sys
import traceback

# Import mudae_tools before PySide6 so Windows Qt DLL paths are preloaded cleanly.
# Create QApplication before importing pyautogui/mss (which call SetProcessDPIAware() at import)
# so Qt6 sets DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 cleanly without Access Denied warnings.
from mudae_tools import APP_NAME, APP_VERSION


def _write_smoke_log(message: str) -> None:
    log_path = os.environ.get("MUDAE_SMOKE_LOG")
    if log_path:
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except Exception:
            pass
    if sys.stdout is not None:
        try:
            print(message, flush=True)
        except Exception:
            pass


def main() -> int:
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication(sys.argv)
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(APP_VERSION)

        from mudae_tools.ui import MainWindow

        window = MainWindow()
        if smoke_test:
            QTimer.singleShot(100, window.close)
            QTimer.singleShot(150, app.quit)
            rc = app.exec()
            app.processEvents()
            del window
            del app
            _write_smoke_log("SMOKE_TEST_OK")
            return rc

        window.show()
        return app.exec()
    except Exception:
        if smoke_test:
            _write_smoke_log(traceback.format_exc())
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
