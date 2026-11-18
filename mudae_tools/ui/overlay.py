"""DPI-aware multi-monitor screen region selection overlay."""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from ..capture import get_physical_cursor_pos


class RegionSelectorOverlay(QWidget):
    """Fullscreen translucent overlay for selecting capture/OCR/button regions."""

    region_selected = Signal(tuple)
    selection_cancelled = Signal()

    def __init__(self, label: str = "Capture Region", allow_single_click: bool = False) -> None:
        super().__init__(None)
        self.label = label
        self.allow_single_click = allow_single_click

        self._dragging = False
        self._start_local: Optional[QPoint] = None
        self._curr_local: Optional[QPoint] = None
        self._start_phys: Optional[Tuple[int, int]] = None
        self._end_phys: Optional[Tuple[int, int]] = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        virtual_rect = QRect()
        for screen in QGuiApplication.screens():
            virtual_rect = virtual_rect.united(screen.geometry())
        if virtual_rect.isNull():
            virtual_rect = QRect(0, 0, 1920, 1080)
        self.setGeometry(virtual_rect)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.selection_cancelled.emit()
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self.selection_cancelled.emit()
            self.close()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_local = event.position().toPoint()
            self._curr_local = self._start_local
            self._start_phys = get_physical_cursor_pos()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._curr_local = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        self._curr_local = event.position().toPoint()
        self._end_phys = get_physical_cursor_pos()

        region = self._compute_physical_region()
        self.close()
        if region is not None:
            self.region_selected.emit(region)
        else:
            self.selection_cancelled.emit()

    def _local_to_physical(self, pt: QPoint) -> Tuple[int, int]:
        global_pt = self.mapToGlobal(pt)
        screen = QGuiApplication.screenAt(global_pt) or QGuiApplication.primaryScreen()
        dpr = float(screen.devicePixelRatio()) if screen else 1.0
        return int(round(global_pt.x() * dpr)), int(round(global_pt.y() * dpr))

    def _compute_physical_region(self) -> Optional[Tuple[int, int, int, int]]:
        if self._start_local is None or self._curr_local is None:
            return None

        local_dx = abs(self._curr_local.x() - self._start_local.x())
        local_dy = abs(self._curr_local.y() - self._start_local.y())

        if self._start_phys is not None and self._end_phys is not None:
            phys_dx = abs(self._end_phys[0] - self._start_phys[0])
            phys_dy = abs(self._end_phys[1] - self._start_phys[1])
            if phys_dx >= 4 or phys_dy >= 4:
                x1 = min(self._start_phys[0], self._end_phys[0])
                y1 = min(self._start_phys[1], self._end_phys[1])
                x2 = max(self._start_phys[0], self._end_phys[0])
                y2 = max(self._start_phys[1], self._end_phys[1])
                return (x1, y1, x2, y2)
            if self.allow_single_click and local_dx < 4 and local_dy < 4:
                cx, cy = self._end_phys
                return (cx - 14, cy - 14, cx + 14, cy + 14)

        # Fallback for offscreen/synthetic tests or single clicks without physical cursor delta
        p1 = self._local_to_physical(self._start_local)
        p2 = self._local_to_physical(self._curr_local)
        if abs(p2[0] - p1[0]) < 4 and abs(p2[1] - p1[1]) < 4:
            if self.allow_single_click:
                cx, cy = p2
                return (cx - 14, cy - 14, cx + 14, cy + 14)
            return None

        return (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dim backdrop
        painter.fillRect(self.rect(), QColor(12, 14, 22, 150))

        # Banner on primary screen
        primary = QGuiApplication.primaryScreen()
        banner_geo = primary.geometry() if primary else self.rect()
        top_left = self.mapFromGlobal(banner_geo.topLeft())
        banner_rect = QRect(top_left.x() + banner_geo.width() // 2 - 270, top_left.y() + 26, 540, 44)
        painter.setPen(QPen(QColor("#5865F2"), 1.5))
        painter.setBrush(QColor(24, 27, 40, 235))
        painter.drawRoundedRect(banner_rect, 10, 10)

        painter.setPen(QColor("#F2F4F8"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        hint = "Click or drag" if self.allow_single_click else "Click and drag"
        painter.drawText(
            banner_rect,
            Qt.AlignmentFlag.AlignCenter,
            f"{hint} to select: {self.label}  •  ESC / Right-Click to cancel",
        )

        if self._dragging and self._start_local is not None and self._curr_local is not None:
            sel = QRect(self._start_local, self._curr_local).normalized()
            if sel.width() > 2 or sel.height() > 2:
                # Clear backdrop inside selection so user sees Discord clearly
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                painter.fillRect(sel, Qt.GlobalColor.transparent)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

                painter.fillRect(sel, QColor(88, 101, 242, 24))
                painter.setPen(QPen(QColor("#7289DA"), 2))
                painter.drawRect(sel)

                info = f"{sel.width()} × {sel.height()}"
                info_rect = QRect(sel.left(), max(8, sel.top() - 28), 120, 24)
                painter.setBrush(QColor(24, 27, 40, 230))
                painter.setPen(QPen(QColor("#5865F2"), 1))
                painter.drawRoundedRect(info_rect, 6, 6)
                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(info_rect, Qt.AlignmentFlag.AlignCenter, info)
