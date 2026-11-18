"""DPI-aware screen capture and safe mouse automation using MSS and PyAutoGUI."""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Optional, Tuple

import mss
import numpy as np
import pyautogui

from .config import RegionTuple, logger

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02


def enable_dpi_awareness() -> None:
    """Enable Per-Monitor v2 DPI awareness on Windows when not managed by Qt6."""
    if sys.platform != "win32" or "PySide6" in sys.modules:
        return
    try:
        awareness = ctypes.c_int()
        # Only set if process DPI awareness is currently unaware (0)
        if ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness)) == 0 and awareness.value != 0:
            return
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception as exc:
            logger.warning("Could not set Windows DPI awareness", error=str(exc))


def get_physical_cursor_pos() -> Optional[Tuple[int, int]]:
    """Return physical screen cursor (x, y) on Windows using Win32 GetCursorPos."""
    if sys.platform != "win32":
        return None

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return None


def bgra_to_rgb(bgra: np.ndarray) -> np.ndarray:
    """Convert raw MSS BGRA uint8 buffer to contiguous RGB uint8 array."""
    if bgra.ndim != 3 or bgra.shape[2] < 3:
        raise ValueError(f"Expected HxWx3+ array from MSS, got shape {bgra.shape}")
    # MSS returns BGRA on Windows/X11; reverse channel order for RGB.
    return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1], dtype=np.uint8)


class ScreenCapture:
    """Thread-safe screen capture helper (creates a fresh MSS instance per call)."""

    @staticmethod
    def capture_region(region: RegionTuple) -> Optional[np.ndarray]:
        """Capture physical screen coordinates (x1, y1, x2, y2) as an RGB numpy array."""
        x1, y1, x2, y2 = (int(v) for v in region)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        monitor = {"left": x1, "top": y1, "width": width, "height": height}
        try:
            with mss.mss() as sct:
                shot = sct.grab(monitor)
                raw = np.asarray(shot, dtype=np.uint8)
                return bgra_to_rgb(raw)
        except Exception as exc:
            logger.error("Screen capture failed", region=region, error=str(exc))
            return None

    @staticmethod
    def click_region_center(region: RegionTuple, restore_cursor: bool = True) -> bool:
        """Click the center of a screen region, optionally restoring the user's cursor."""
        try:
            x1, y1, x2, y2 = (int(v) for v in region)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            prev_pos = pyautogui.position() if restore_cursor else None
            pyautogui.click(cx, cy)
            if restore_cursor and prev_pos is not None:
                time.sleep(0.05)
                pyautogui.moveTo(prev_pos.x, prev_pos.y)
            return True
        except Exception as exc:
            logger.error("Click region failed", region=region, error=str(exc))
            return False
