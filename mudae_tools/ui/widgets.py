"""Reusable Qt6 gallery, card, and edit widgets for Mudae Power Tools Pro."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..config import Character

_MIME_TYPE = "application/x-mudae-card-index"


class CharacterEditDialog(QDialog):
    """Modal dialog to edit a character's Mudae name and series."""

    def __init__(self, character: Character, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Character Metadata")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit(character.name)
        self.series_edit = QLineEdit(character.series)
        self.tags_edit = QLineEdit(", ".join(character.tags))

        form.addRow("Character Name:", self.name_edit)
        form.addRow("Series Name:", self.series_edit)
        form.addRow("Tags (comma-sep):", self.tags_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str, List[str]]:
        tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        return self.name_edit.text().strip(), self.series_edit.text().strip(), tags


class CharacterCardWidget(QFrame):
    """Visual Mudae card tile with OKLab color swatches, rank badge, and drag/swap controls."""

    clicked = Signal(int)
    move_requested = Signal(int, int)
    swap_requested = Signal(int, int)
    edit_requested = Signal(int)
    delete_requested = Signal(int)

    def __init__(
        self,
        index: int,
        character: Character,
        pixmap: Optional[QPixmap] = None,
        selected: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self.character = character
        self._drag_start_pos: Optional[QPoint] = None
        self.setAcceptDrops(True)
        self.setObjectName("CardTileSelected" if selected else "CardTile")
        self.setFixedWidth(182)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Top metadata strip: rank + optional cluster / score badge
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        rank_lbl = QLabel(f"#{index + 1}")
        rank_lbl.setObjectName("RankBadge")
        top_row.addWidget(rank_lbl)
        top_row.addStretch()

        if "prompt_score" in character.metadata:
            score_val = float(character.metadata["prompt_score"])
            badge = QLabel(f"{score_val:.2f}")
            badge.setObjectName("ScoreBadge")
            top_row.addWidget(badge)
        elif character.cluster_id >= 0:
            badge = QLabel(f"Theme {character.cluster_id + 1}")
            badge.setObjectName("ClusterBadge")
            top_row.addWidget(badge)

        root.addLayout(top_row)

        # Thumbnail
        thumb_lbl = QLabel()
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setFixedSize(166, 210)
        thumb_lbl.setObjectName("CardThumb")
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                166,
                210,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumb_lbl.setPixmap(scaled)
        else:
            thumb_lbl.setText("Missing\nImage")
        root.addWidget(thumb_lbl)

        # 5-swatch OKLab palette bar
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(2)
        colors = character.dominant_hex[:5] if character.dominant_hex else ["#2B2D31"] * 5
        while len(colors) < 5:
            colors.append(colors[-1])
        for hx in colors:
            sw = QFrame()
            sw.setFixedHeight(8)
            sw.setStyleSheet(f"background-color: {hx}; border-radius: 2px;")
            sw.setToolTip(f"Dominant OKLab color: {hx}")
            swatch_row.addWidget(sw)
        root.addLayout(swatch_row)

        # Name & series (elide long strings cleanly so glyphs never clip mid-letter)
        name_lbl = QLabel()
        name_lbl.setObjectName("CardName")
        name_lbl.setToolTip(character.name)
        name_lbl.setText(name_lbl.fontMetrics().elidedText(character.name, Qt.TextElideMode.ElideRight, 164))
        root.addWidget(name_lbl)

        series_lbl = QLabel()
        series_lbl.setObjectName("CardSeries")
        series_lbl.setToolTip(character.series)
        series_lbl.setText(series_lbl.fontMetrics().elidedText(character.series, Qt.TextElideMode.ElideRight, 164))
        root.addWidget(series_lbl)

        # Action bar
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        left_btn = QPushButton("◀")
        left_btn.setObjectName("MiniCardBtn")
        left_btn.setToolTip("Move earlier (-1)")
        left_btn.clicked.connect(lambda: self.move_requested.emit(self.index, self.index - 1))

        right_btn = QPushButton("▶")
        right_btn.setObjectName("MiniCardBtn")
        right_btn.setToolTip("Move later (+1)")
        right_btn.clicked.connect(lambda: self.move_requested.emit(self.index, self.index + 1))

        edit_btn = QPushButton("Edit")
        edit_btn.setObjectName("MiniCardBtn")
        edit_btn.setToolTip("Edit character name / series")
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(self.index))

        del_btn = QPushButton("✕")
        del_btn.setObjectName("MiniDangerBtn")
        del_btn.setToolTip("Remove from collection")
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self.index))

        btn_row.addWidget(left_btn)
        btn_row.addWidget(right_btn)
        btn_row.addWidget(edit_btn, 1)
        btn_row.addWidget(del_btn)
        root.addLayout(btn_row)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.position().toPoint()
            self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit_requested.emit(self.index)
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not (event.buttons() & Qt.MouseButton.LeftButton) or self._drag_start_pos is None:
            return
        if (event.position().toPoint() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_MIME_TYPE, str(self.index).encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(_MIME_TYPE):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(_MIME_TYPE):
            src_bytes = bytes(event.mimeData().data(_MIME_TYPE))
            src_idx = int(src_bytes.decode("utf-8"))
            if src_idx != self.index:
                self.swap_requested.emit(src_idx, self.index)
            event.acceptProposedAction()


class HaremGalleryWidget(QWidget):
    """Scrollable multi-column Mudae harem grid supporting swap, move, edit, and delete."""

    order_changed = Signal(list)
    character_edited = Signal(object)
    character_deleted = Signal(object)
    selection_changed = Signal(int)

    def __init__(self, columns: int = 5, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._columns = max(2, columns)
        self._characters: List[Character] = []
        self._selected_index: Optional[int] = None
        self._pixmap_cache: Dict[str, QPixmap] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("GalleryScroll")

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(12)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.scroll.setWidget(self._container)
        outer.addWidget(self.scroll)

        self._empty_label = QLabel(
            "No characters loaded yet.\nUse Live Scraper, Import Folder, or Generate Demo Cards to begin."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setObjectName("EmptyGalleryHint")
        self._grid.addWidget(self._empty_label, 0, 0)

    def set_columns(self, columns: int) -> None:
        self._columns = max(2, int(columns))
        self._rebuild_grid()

    def set_characters(self, characters: List[Character], clear_cache: bool = False) -> None:
        if clear_cache:
            self._pixmap_cache.clear()
        self._characters = list(characters)
        self._selected_index = None
        self._rebuild_grid()

    def characters(self) -> List[Character]:
        return list(self._characters)

    def _get_pixmap(self, path_str: str) -> Optional[QPixmap]:
        if path_str in self._pixmap_cache:
            return self._pixmap_cache[path_str]
        pm = QPixmap(path_str)
        if not pm.isNull():
            scaled = pm.scaled(
                180,
                220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._pixmap_cache[path_str] = scaled
            return scaled
        return None

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._empty_label:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif w is self._empty_label:
                self._empty_label.hide()
                self._empty_label.setParent(None)

    def _rebuild_grid(self) -> None:
        self._clear_grid()
        if not self._characters:
            self._grid.addWidget(self._empty_label, 0, 0)
            self._empty_label.show()
            return

        for idx, ch in enumerate(self._characters):
            row = idx // self._columns
            col = idx % self._columns
            pm = self._get_pixmap(str(ch.image_path))
            card = CharacterCardWidget(
                index=idx,
                character=ch,
                pixmap=pm,
                selected=(self._selected_index == idx),
            )
            card.clicked.connect(self._on_card_clicked)
            card.move_requested.connect(self._on_move_requested)
            card.swap_requested.connect(self._on_swap_requested)
            card.edit_requested.connect(self._on_edit_requested)
            card.delete_requested.connect(self._on_delete_requested)
            self._grid.addWidget(card, row, col)

    def _on_card_clicked(self, idx: int) -> None:
        if self._selected_index is None:
            self._selected_index = idx
            self.selection_changed.emit(idx)
            self._rebuild_grid()
            return

        if self._selected_index == idx:
            self._selected_index = None
            self.selection_changed.emit(-1)
            self._rebuild_grid()
            return

        # Click second card to swap positions immediately
        first = self._selected_index
        self._selected_index = None
        self._on_swap_requested(first, idx)

    def _on_move_requested(self, src: int, dst: int) -> None:
        if not (0 <= src < len(self._characters)) or not (0 <= dst < len(self._characters)):
            return
        self._characters[src], self._characters[dst] = self._characters[dst], self._characters[src]
        for i, ch in enumerate(self._characters):
            ch.sort_index = i
        self._selected_index = dst
        self._rebuild_grid()
        self.order_changed.emit(list(self._characters))

    def _on_swap_requested(self, src: int, dst: int) -> None:
        if not (0 <= src < len(self._characters)) or not (0 <= dst < len(self._characters)):
            return
        item = self._characters.pop(src)
        self._characters.insert(dst, item)
        for i, ch in enumerate(self._characters):
            ch.sort_index = i
        self._selected_index = None
        self._rebuild_grid()
        self.order_changed.emit(list(self._characters))

    def _on_edit_requested(self, idx: int) -> None:
        if not (0 <= idx < len(self._characters)):
            return
        ch = self._characters[idx]
        dlg = CharacterEditDialog(ch, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_name, new_series, new_tags = dlg.values()
            if new_name:
                ch.name = new_name
            if new_series:
                ch.series = new_series
            ch.tags = new_tags
            self._rebuild_grid()
            self.character_edited.emit(ch)

    def _on_delete_requested(self, idx: int) -> None:
        if not (0 <= idx < len(self._characters)):
            return
        removed = self._characters.pop(idx)
        for i, ch in enumerate(self._characters):
            ch.sort_index = i
        self._selected_index = None
        self._rebuild_grid()
        self.character_deleted.emit(removed)
