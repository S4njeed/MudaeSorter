"""Mudae Power Tools Pro - Modern PySide6 + AI Character Scraper & Sorter."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import List

APP_NAME = "Mudae Power Tools Pro"
APP_VERSION = "3.0.0"

_DLL_DIR_HANDLES: List[object] = []


def _ensure_windows_qt_dlls() -> None:
    """Preload PySide6/shiboken6 DLLs with altered search path to prevent Anaconda DLL clashes."""
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("PySide6")
    if spec is None or not spec.origin:
        return
    ps_dir = Path(spec.origin).parent
    shib_dir = ps_dir.parent / "shiboken6"
    for directory in (ps_dir, shib_dir):
        if directory.is_dir():
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            except Exception:
                pass

    k32 = ctypes.windll.kernel32
    k32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    k32.LoadLibraryExW.restype = ctypes.c_void_p
    load_with_altered_search_path = 0x00000008
    for dll_path in (
        shib_dir / "shiboken6.abi3.dll",
        ps_dir / "pyside6.abi3.dll",
        ps_dir / "Qt6Core.dll",
        ps_dir / "Qt6Gui.dll",
        ps_dir / "Qt6Widgets.dll",
    ):
        if dll_path.is_file():
            k32.LoadLibraryExW(str(dll_path), None, load_with_altered_search_path)


_ensure_windows_qt_dlls()
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

__all__ = ["APP_NAME", "APP_VERSION"]
