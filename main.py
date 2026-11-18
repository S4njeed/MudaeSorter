"""Entry point for Mudae Power Tools Pro v3.0."""

from __future__ import annotations

import sys

# Import mudae_tools before PySide6 so Windows Qt DLL paths are preloaded cleanly.
# Create QApplication before importing pyautogui/mss (which call SetProcessDPIAware() at import)
# so Qt6 sets DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 cleanly without Access Denied warnings.
from mudae_tools import APP_NAME, APP_VERSION
from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    from mudae_tools.ui import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
