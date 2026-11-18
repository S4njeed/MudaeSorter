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
    """Preload Windows system ICU + PySide6/shiboken6 DLLs cleanly (normal & frozen builds)."""
    if sys.platform != "win32":
        return

    k32 = ctypes.windll.kernel32
    if "--smoke-test" in sys.argv:
        # Prevent modal OS error popups during headless CI/automated smoke tests.
        k32.SetErrorMode(0x0001 | 0x0002)

    # Ensure Windows system icuuc.dll (exports ucnv_open) is loaded before any Anaconda/PATH ICU DLL.
    sys_icu = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icuuc.dll"
    if sys_icu.is_file():
        try:
            ctypes.WinDLL(str(sys_icu))
        except Exception:
            pass

    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        ps_dir = base / "PySide6" if (base / "PySide6").is_dir() else base
        shib_dir = base / "shiboken6" if (base / "shiboken6").is_dir() else base
        plugins_dir = ps_dir / "plugins"
        if plugins_dir.is_dir():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins_dir))
    else:
        spec = importlib.util.find_spec("PySide6")
        if spec is None or not spec.origin:
            return
        ps_dir = Path(spec.origin).parent
        shib_dir = ps_dir.parent / "shiboken6"

    for directory in dict.fromkeys((ps_dir, shib_dir)):
        if directory.is_dir():
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            except Exception:
                pass

    k32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    k32.LoadLibraryExW.restype = ctypes.c_void_p
    load_with_altered_search_path = 0x00000008
    for dll_path in (
        ps_dir / "msvcp140.dll",
        ps_dir / "msvcp140_1.dll",
        ps_dir / "msvcp140_2.dll",
        ps_dir / "vcruntime140_1.dll",
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
