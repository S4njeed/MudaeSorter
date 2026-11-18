"""Main PySide6 window for Mudae Power Tools Pro v3.0."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, APP_VERSION
from ..config import (
    SUPPORTED_VISION_MODELS,
    AppConfig,
    Character,
    ImageFormat,
    RegionTuple,
    SortMethod,
    logger,
)
from ..database import DatabaseManager
from ..exporter import export_sorted_folder, generate_mma_text, generate_sm_commands
from ..ocr import OCRService, parse_mudae_text_list
from ..vision import CLIPEngine
from .overlay import RegionSelectorOverlay
from .widgets import HaremGalleryWidget
from .workers import DemoGeneratorWorker, FolderImportWorker, ScraperWorker, SortWorker

DARK_STYLESHEET = """
QMainWindow, QDialog, #PageContainer, QStackedWidget {
    background-color: #13151C;
    color: #EAECEF;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QWidget {
    color: #EAECEF;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QLabel, QCheckBox {
    background-color: transparent;
}
#Sidebar {
    background-color: #0E1016;
    border-right: 1px solid #242836;
}
#BrandTitle {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
}
#BrandSub {
    font-size: 11px;
    color: #8E96AD;
}
#SidebarFooterCard {
    background-color: #171A24;
    border: 1px solid #262B3A;
    border-radius: 8px;
    padding: 8px;
}
QPushButton#NavBtn {
    background-color: transparent;
    color: #B5BAC9;
    text-align: left;
    padding: 11px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    border: none;
}
QPushButton#NavBtn:hover {
    background-color: #1D212D;
    color: #FFFFFF;
}
QPushButton#NavBtnActive {
    background-color: #5865F2;
    color: #FFFFFF;
    text-align: left;
    padding: 11px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    border: none;
}
QGroupBox {
    background-color: #1A1D27;
    border: 1px solid #2A2F40;
    border-radius: 10px;
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 600;
    color: #DCE2F2;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    background-color: #13151C;
    border-radius: 4px;
    color: #C7D0EA;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background-color: #11131A;
    border: 1px solid #303648;
    border-radius: 6px;
    padding: 4px 10px;
    color: #F2F4F8;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 22px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #5865F2;
}
QComboBox QAbstractItemView {
    background-color: #171A24;
    border: 1px solid #303648;
    selection-background-color: #5865F2;
    color: #F2F4F8;
}
QPushButton {
    background-color: #272B3A;
    color: #EAECEF;
    border: 1px solid #383E54;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #32374A;
    border-color: #4A526E;
}
QPushButton:disabled {
    background-color: #1C1F29;
    color: #676D80;
    border-color: #262A38;
}
QPushButton#PrimaryBtn {
    background-color: #5865F2;
    border: 1px solid #6D78F6;
    color: #FFFFFF;
}
QPushButton#PrimaryBtn:hover {
    background-color: #4752C4;
}
QPushButton#SuccessBtn {
    background-color: #238636;
    border: 1px solid #2EA043;
    color: #FFFFFF;
}
QPushButton#SuccessBtn:hover {
    background-color: #1E6E2D;
}
QPushButton#DangerBtn {
    background-color: #C93338;
    border: 1px solid #DA454A;
    color: #FFFFFF;
}
QPushButton#DangerBtn:hover {
    background-color: #A8282C;
}
QProgressBar {
    background-color: #11131A;
    border: 1px solid #2A2F40;
    border-radius: 6px;
    text-align: center;
    height: 20px;
    color: #EAECEF;
}
QProgressBar::chunk {
    background-color: #5865F2;
    border-radius: 5px;
}
#RegionBadgeUnset {
    background-color: #12141C;
    border: 1px solid #292E3E;
    border-radius: 6px;
    padding: 5px 10px;
    color: #8F97AE;
}
#RegionBadgeSet {
    background-color: #14261E;
    border: 1px solid #266346;
    border-radius: 6px;
    padding: 5px 10px;
    color: #57F287;
    font-weight: 600;
}
#RuntimeBox {
    background-color: #12141C;
    border: 1px solid #292E3E;
    border-radius: 6px;
    padding: 8px 10px;
    color: #A7B0C7;
    font-size: 12px;
}
#GalleryScroll {
    background-color: #11131A;
    border: 1px solid #2A2F40;
    border-radius: 10px;
}
#CardTile {
    background-color: #1F222E;
    border: 1px solid #303648;
    border-radius: 10px;
}
#CardTile:hover {
    border: 1px solid #5865F2;
}
#CardTileSelected {
    background-color: #242838;
    border: 2px solid #57F287;
    border-radius: 10px;
}
#CardThumb {
    background-color: #12141B;
    border-radius: 6px;
}
#RankBadge {
    background-color: #5865F2;
    color: #FFFFFF;
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 700;
}
#ClusterBadge {
    background-color: #2E6F5E;
    color: #E8FFF6;
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 600;
}
#ScoreBadge {
    background-color: #7D5BA6;
    color: #FFFFFF;
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 600;
}
#CardName {
    font-weight: 700;
    font-size: 13px;
    color: #FFFFFF;
}
#CardSeries {
    font-size: 11px;
    color: #9AA1B5;
}
QPushButton#MiniCardBtn {
    padding: 3px 6px;
    font-size: 11px;
    border-radius: 5px;
}
QPushButton#MiniDangerBtn {
    padding: 3px 6px;
    font-size: 11px;
    border-radius: 5px;
    background-color: #462328;
    border: 1px solid #75343C;
    color: #FFB4BA;
}
QPushButton#MiniDangerBtn:hover {
    background-color: #C93338;
    color: #FFFFFF;
}
#MetricCard {
    background-color: #1A1D27;
    border: 1px solid #2A2F40;
    border-top: 3px solid #5865F2;
    border-radius: 10px;
    padding: 12px 14px;
}
#MetricValue {
    font-size: 26px;
    font-weight: 700;
    color: #FFFFFF;
}
#MetricTitle {
    font-size: 12px;
    color: #949BB0;
}
#EmptyGalleryHint {
    font-size: 14px;
    color: #7E859B;
    padding: 40px;
}
QTableWidget {
    background-color: #13151C;
    border: 1px solid #2A2F40;
    border-radius: 8px;
    gridline-color: #262A38;
}
QHeaderView::section {
    background-color: #1F222E;
    color: #DCE0EC;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #2F3548;
    font-weight: 700;
}
QTableCornerButton::section {
    background-color: #1F222E;
    border: none;
}
QScrollBar:vertical {
    background-color: #11131A;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #343A4E;
    min-height: 28px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background-color: #5865F2;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}
QScrollBar:horizontal {
    background-color: #11131A;
    height: 10px;
    margin: 2px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #343A4E;
    min-width: 28px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #5865F2;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}
"""


class MainWindow(QMainWindow):
    """Desktop workspace for capturing, indexing, visually sorting, and exporting Mudae harems."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        super().__init__()
        self.config = config or AppConfig.load()
        self.db = db or DatabaseManager()
        self.ocr = OCRService(min_confidence=self.config.ocr_min_confidence)
        self.clip_engine = CLIPEngine(
            model_name=self.config.clip_model_name,
            use_gpu=self.config.use_gpu,
        )

        self._characters: List[Character] = []
        self._active_overlay: Optional[RegionSelectorOverlay] = None
        self._scraper_worker: Optional[ScraperWorker] = None
        self._import_worker: Optional[FolderImportWorker] = None
        self._demo_worker: Optional[DemoGeneratorWorker] = None
        self._sort_worker: Optional[SortWorker] = None

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1340, 860)
        self.setMinimumSize(1080, 720)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._reload_characters_from_db()
        self._refresh_region_badges()
        self._refresh_analytics()
        self._rebuild_sm_commands()

    # ------------------------------------------------------------------
    # Layout Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("PageContainer")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(245)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(16, 20, 16, 16)
        sb_layout.setSpacing(8)

        brand = QLabel("Mudae Power Tools")
        brand.setObjectName("BrandTitle")
        sub = QLabel(f"Pro v{APP_VERSION} • AI Harem Sorter")
        sub.setObjectName("BrandSub")
        sb_layout.addWidget(brand)
        sb_layout.addWidget(sub)
        sb_layout.addSpacing(14)

        self._nav_buttons: List[QPushButton] = []
        nav_specs = [
            ("🖼️  Scraper && Import", 0),
            ("🎨  AI Sorter && Grid", 1),
            ("📋  Mudae $sm && Export", 2),
            ("📊  Analytics && Settings", 3),
        ]
        for label, idx in nav_specs:
            btn = QPushButton(label)
            btn.setObjectName("NavBtn")
            btn.setMinimumHeight(42)
            btn.clicked.connect(lambda _=False, i=idx: self._switch_tab(i))
            sb_layout.addWidget(btn)
            self._nav_buttons.append(btn)

        sb_layout.addStretch()

        footer_card = QFrame()
        footer_card.setObjectName("SidebarFooterCard")
        fc_layout = QVBoxLayout(footer_card)
        fc_layout.setContentsMargins(10, 10, 10, 10)
        fc_layout.setSpacing(4)

        self.sidebar_count_lbl = QLabel("Collection: 0 cards")
        self.sidebar_count_lbl.setStyleSheet("color: #DCE2F2; font-weight: 600;")
        fc_layout.addWidget(self.sidebar_count_lbl)

        self.sidebar_status_lbl = QLabel("Ready")
        self.sidebar_status_lbl.setWordWrap(True)
        self.sidebar_status_lbl.setStyleSheet("color: #8890A8; font-size: 11px;")
        fc_layout.addWidget(self.sidebar_status_lbl)
        sb_layout.addWidget(footer_card)

        root.addWidget(sidebar)

        # Stacked content pages
        self.stack = QStackedWidget()
        self.tab_scraper = self._create_scraper_tab()
        self.tab_sorter = self._create_sorter_tab()
        self.tab_export = self._create_export_tab()
        self.tab_settings = self._create_settings_tab()

        self.stack.addWidget(self.tab_scraper)
        self.stack.addWidget(self.tab_sorter)
        self.stack.addWidget(self.tab_export)
        self.stack.addWidget(self.tab_settings)

        root.addWidget(self.stack, 1)
        self._switch_tab(0)

    def _switch_tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setObjectName("NavBtnActive" if i == index else "NavBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        if index == 2:
            self._rebuild_sm_commands()
        elif index == 3:
            self._refresh_analytics()

    # ------------------------------------------------------------------
    # Tab 1: Live Discord Scraper & Folder Import
    # ------------------------------------------------------------------

    def _create_scraper_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("PageContainer")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 16, 24, 18)
        layout.setSpacing(9)

        title = QLabel("🖼️ Live Discord Scraper & Collection Importer")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #FFFFFF;")
        layout.addWidget(title)

        # Output folder row
        folder_group = QGroupBox("Output && Storage Folder")
        fg_layout = QHBoxLayout(folder_group)
        fg_layout.setSpacing(10)
        self.output_folder_edit = QLineEdit(str(self.config.output_folder))
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(96)
        browse_btn.setMinimumHeight(32)
        browse_btn.clicked.connect(self._browse_output_folder)
        fg_layout.addWidget(QLabel("Save Folder:"))
        fg_layout.addWidget(self.output_folder_edit, 1)
        fg_layout.addWidget(browse_btn)
        layout.addWidget(folder_group)

        # Screen regions setup
        region_group = QGroupBox("1. Screen Capture Regions (DPI-Aware)")
        rg_grid = QGridLayout(region_group)
        rg_grid.setHorizontalSpacing(10)
        rg_grid.setVerticalSpacing(6)

        self._region_badges = {}
        region_specs = [
            ("capture_region", "Select Card Image Region", "Card Image (Required)", False, 0),
            ("character_name_region", "Select Character Name OCR", "Character Name (OCR)", False, 1),
            ("show_name_region", "Select Series Name OCR", "Series Name (OCR)", False, 2),
            ("next_button_region", "Select Next Button", "Next Arrow Button", True, 3),
        ]
        for key, btn_text, overlay_lbl, single_click, row in region_specs:
            btn = QPushButton(btn_text)
            btn.setFixedWidth(220)
            btn.setMinimumHeight(32)
            btn.clicked.connect(
                lambda _=False, k=key, lbl=overlay_lbl, sc=single_click: self._start_region_pick(k, lbl, sc)
            )
            badge = QLabel("Not set")
            badge.setObjectName("RegionBadgeUnset")
            badge.setMinimumHeight(30)
            clr_btn = QPushButton("Clear")
            clr_btn.setFixedWidth(74)
            clr_btn.setMinimumHeight(30)
            clr_btn.clicked.connect(lambda _=False, k=key: self._clear_region(k))
            rg_grid.addWidget(btn, row, 0)
            rg_grid.addWidget(badge, row, 1)
            rg_grid.addWidget(clr_btn, row, 2)
            self._region_badges[key] = badge

        rg_grid.setColumnStretch(1, 1)
        layout.addWidget(region_group)

        # Controls + Actions
        ctrl_group = QGroupBox("2. Capture && Import Actions")
        cg_layout = QVBoxLayout(ctrl_group)
        cg_layout.setSpacing(10)

        params_row = QHBoxLayout()
        params_row.setSpacing(10)
        params_row.addWidget(QLabel("Pages to capture:"))
        self.capture_count_spin = QSpinBox()
        self.capture_count_spin.setRange(1, 5000)
        self.capture_count_spin.setMinimumWidth(90)
        self.capture_count_spin.setValue(self.config.capture_count)
        params_row.addWidget(self.capture_count_spin)

        params_row.addSpacing(8)
        params_row.addWidget(QLabel("Delay (sec):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.2, 30.0)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setMinimumWidth(90)
        self.delay_spin.setValue(self.config.capture_delay)
        params_row.addWidget(self.delay_spin)

        params_row.addSpacing(12)
        self.auto_scroll_chk = QCheckBox("Auto-click Next Button")
        self.auto_scroll_chk.setChecked(self.config.auto_scroll)
        params_row.addWidget(self.auto_scroll_chk)

        params_row.addSpacing(8)
        self.restore_cursor_chk = QCheckBox("Restore cursor")
        self.restore_cursor_chk.setToolTip("Restore mouse cursor position after auto-clicking Next")
        self.restore_cursor_chk.setChecked(self.config.restore_cursor)
        params_row.addWidget(self.restore_cursor_chk)
        params_row.addStretch()
        cg_layout.addLayout(params_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_scrape_btn = QPushButton("▶ Start Scraping")
        self.start_scrape_btn.setObjectName("SuccessBtn")
        self.start_scrape_btn.setMinimumHeight(34)
        self.start_scrape_btn.clicked.connect(self._start_scraping)

        self.stop_scrape_btn = QPushButton("■ Stop")
        self.stop_scrape_btn.setObjectName("DangerBtn")
        self.stop_scrape_btn.setMinimumHeight(34)
        self.stop_scrape_btn.setEnabled(False)
        self.stop_scrape_btn.clicked.connect(self._stop_scraping)

        self.import_folder_btn = QPushButton("📁 Import Folder...")
        self.import_folder_btn.setObjectName("PrimaryBtn")
        self.import_folder_btn.setMinimumHeight(34)
        self.import_folder_btn.clicked.connect(self._import_local_folder)

        self.paste_list_btn = QPushButton("📋 Paste Text List...")
        self.paste_list_btn.setMinimumHeight(34)
        self.paste_list_btn.clicked.connect(self._open_text_list_import)

        self.demo_cards_btn = QPushButton("✨ 12 Demo Cards")
        self.demo_cards_btn.setMinimumHeight(34)
        self.demo_cards_btn.clicked.connect(self._generate_demo_cards)

        btn_row.addWidget(self.start_scrape_btn)
        btn_row.addWidget(self.stop_scrape_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(self.import_folder_btn)
        btn_row.addWidget(self.paste_list_btn)
        btn_row.addWidget(self.demo_cards_btn)
        btn_row.addStretch()
        cg_layout.addLayout(btn_row)

        self.scrape_progress = QProgressBar()
        self.scrape_progress.setValue(0)
        self.scrape_status_lbl = QLabel("Idle — select screen regions or import a folder.")
        self.scrape_status_lbl.setStyleSheet("color: #A5ACBF;")
        cg_layout.addWidget(self.scrape_progress)
        cg_layout.addWidget(self.scrape_status_lbl)
        layout.addWidget(ctrl_group)

        # Bottom split: live preview + log
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        preview_box = QGroupBox("Last Captured Card")
        preview_box.setFixedWidth(232)
        pb_layout = QVBoxLayout(preview_box)
        pb_layout.setSpacing(6)
        self.preview_img_lbl = QLabel("No card yet")
        self.preview_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_img_lbl.setFixedSize(192, 118)
        self.preview_img_lbl.setStyleSheet(
            "background-color: #11131A; border: 1px solid #282D3D; border-radius: 8px; color: #7C8398;"
        )
        self.preview_meta_lbl = QLabel("")
        self.preview_meta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_meta_lbl.setWordWrap(True)
        self.preview_meta_lbl.setStyleSheet("color: #CBD2E6; font-size: 12px;")
        pb_layout.addWidget(self.preview_img_lbl, 0, Qt.AlignmentFlag.AlignCenter)
        pb_layout.addWidget(self.preview_meta_lbl)
        pb_layout.addStretch()
        bottom_row.addWidget(preview_box)

        log_box = QGroupBox("Activity Log")
        lb_layout = QVBoxLayout(log_box)
        self.activity_log = QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMinimumHeight(120)
        lb_layout.addWidget(self.activity_log)
        bottom_row.addWidget(log_box, 1)

        layout.addLayout(bottom_row, 1)
        return page

    # ------------------------------------------------------------------
    # Tab 2: AI Sorter & Interactive Mudae Grid
    # ------------------------------------------------------------------

    def _create_sorter_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("PageContainer")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        header_row = QHBoxLayout()
        title = QLabel("🎨 AI Visual Sorter & Mudae Harem Grid")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #FFFFFF;")
        header_row.addWidget(title)
        header_row.addStretch()

        header_row.addWidget(QLabel("Grid Columns:"))
        self.columns_combo = QComboBox()
        self.columns_combo.setMinimumWidth(160)
        self.columns_combo.addItem("5 cols (Mudae Page)", 5)
        self.columns_combo.addItem("6 cols", 6)
        self.columns_combo.addItem("8 cols", 8)
        self.columns_combo.addItem("10 cols", 10)
        idx = max(0, self.columns_combo.findData(self.config.grid_columns))
        self.columns_combo.setCurrentIndex(idx)
        self.columns_combo.currentIndexChanged.connect(self._on_columns_changed)
        header_row.addWidget(self.columns_combo)
        layout.addLayout(header_row)

        # Sorting algorithms bar (2-row responsive grid so labels never clip)
        algo_group = QGroupBox("Sorting Algorithms")
        ag_layout = QVBoxLayout(algo_group)
        ag_layout.setSpacing(10)

        btn_grid = QGridLayout()
        btn_grid.setHorizontalSpacing(10)
        btn_grid.setVerticalSpacing(8)
        algo_buttons = [
            ("🌈 OKLab Gradient (2-Opt)", SortMethod.COLOR_GRADIENT_TSP, "PrimaryBtn", 0, 0),
            ("🌀 3D Hilbert Curve", SortMethod.HILBERT_COLOR, "", 0, 1),
            ("🎨 Rainbow (OKLCh)", SortMethod.RAINBOW_SPECTRUM, "", 0, 2),
            ("🧠 Vision Similarity (TSP)", SortMethod.VISUAL_SIMILARITY_TSP, "PrimaryBtn", 0, 3),
            ("🧩 Theme Clusters (HDBSCAN)", SortMethod.THEME_CLUSTERS, "", 1, 0),
            ("📺 Group by Series", SortMethod.SERIES_GROUPING, "", 1, 1),
            ("🔤 Alphabetical (A–Z)", SortMethod.ALPHABETICAL, "", 1, 2),
        ]
        self._sort_buttons: List[QPushButton] = []
        for label, method, obj_name, r, c in algo_buttons:
            btn = QPushButton(label)
            if obj_name:
                btn.setObjectName(obj_name)
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _=False, m=method: self._run_sort(m))
            btn_grid.addWidget(btn, r, c)
            self._sort_buttons.append(btn)
        for col in range(4):
            btn_grid.setColumnStretch(col, 1)
        ag_layout.addLayout(btn_grid)

        # Natural-language CLIP search + local filter row
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(10)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "Natural-language visual rank via SigLIP / CLIP (e.g. 'pink hair', 'dark armor', 'blue glow')..."
        )
        self.prompt_edit.returnPressed.connect(self._run_prompt_sort)
        self.prompt_btn = QPushButton("🔍 Rank by Visual Prompt")
        self.prompt_btn.setObjectName("PrimaryBtn")
        self.prompt_btn.setMinimumHeight(34)
        self.prompt_btn.clicked.connect(self._run_prompt_sort)
        self._sort_buttons.append(self.prompt_btn)

        prompt_row.addWidget(self.prompt_edit, 3)
        prompt_row.addWidget(self.prompt_btn)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter grid by name / series...")
        self.filter_edit.textChanged.connect(self._apply_grid_filter)
        prompt_row.addWidget(self.filter_edit, 2)
        ag_layout.addLayout(prompt_row)

        self.sort_status_lbl = QLabel(
            "Tip: Click any 2 cards to swap them, drag & drop cards, or double-click to edit Name / Series."
        )
        self.sort_status_lbl.setStyleSheet("color: #9AA1B5; font-size: 12px;")
        ag_layout.addWidget(self.sort_status_lbl)
        layout.addWidget(algo_group)

        # Gallery
        self.gallery = HaremGalleryWidget(columns=self.config.grid_columns)
        self.gallery.order_changed.connect(self._on_gallery_order_changed)
        self.gallery.character_edited.connect(self._on_character_edited)
        self.gallery.character_deleted.connect(self._on_character_deleted)
        self.gallery.selection_changed.connect(self._on_gallery_selection_changed)
        layout.addWidget(self.gallery, 1)
        return page

    # ------------------------------------------------------------------
    # Tab 3: Mudae $sm Generator & Folder Exporter
    # ------------------------------------------------------------------

    def _create_export_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("PageContainer")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("📋 Mudae $sm ($sortmarry) Command Builder & Image Exporter")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #FFFFFF;")
        layout.addWidget(title)

        # $sm controls (2-row layout so checkbox & buttons never clip)
        sm_ctrl = QGroupBox("Discord $sm Batch Settings (Auto-split < 2,000 chars)")
        sc_outer = QVBoxLayout(sm_ctrl)
        sc_outer.setSpacing(10)

        sc_row1 = QHBoxLayout()
        sc_row1.setSpacing(10)
        sc_row1.addWidget(QLabel("Prefix:"))
        self.prefix_edit = QLineEdit(self.config.mudae_prefix)
        self.prefix_edit.setFixedWidth(50)
        sc_row1.addWidget(self.prefix_edit)

        sc_row1.addSpacing(4)
        sc_row1.addWidget(QLabel("Command:"))
        self.cmd_edit = QLineEdit(self.config.mudae_command)
        self.cmd_edit.setFixedWidth(68)
        sc_row1.addWidget(self.cmd_edit)

        sc_row1.addSpacing(4)
        sc_row1.addWidget(QLabel("Max Chars:"))
        self.max_chars_spin = QSpinBox()
        self.max_chars_spin.setRange(200, 2000)
        self.max_chars_spin.setMinimumWidth(96)
        self.max_chars_spin.setValue(self.config.mudae_max_chars)
        sc_row1.addWidget(self.max_chars_spin)

        sc_row1.addSpacing(10)
        self.pos_chain_chk = QCheckBox("Chain multi-batch sorts with '$smp <LastChar>' ($sortmarrypos)")
        self.pos_chain_chk.setChecked(self.config.use_pos_chaining)
        sc_row1.addWidget(self.pos_chain_chk)
        sc_row1.addStretch()
        sc_outer.addLayout(sc_row1)

        sc_row2 = QHBoxLayout()
        sc_row2.setSpacing(10)
        regen_btn = QPushButton("🔄 Regenerate Batches")
        regen_btn.setObjectName("PrimaryBtn")
        regen_btn.setMinimumHeight(34)
        regen_btn.clicked.connect(self._rebuild_sm_commands)
        sc_row2.addWidget(regen_btn)

        copy_all_btn = QPushButton("📋 Copy All Batches")
        copy_all_btn.setMinimumHeight(34)
        copy_all_btn.clicked.connect(self._copy_all_sm_batches)
        sc_row2.addWidget(copy_all_btn)

        copy_list_btn = QPushButton("📄 Copy Plain List")
        copy_list_btn.setMinimumHeight(34)
        copy_list_btn.clicked.connect(self._copy_plain_list)
        sc_row2.addWidget(copy_list_btn)

        sc_row2.addStretch()
        sc_outer.addLayout(sc_row2)
        layout.addWidget(sm_ctrl)

        # Scroll area for batch boxes
        self.sm_scroll = QScrollArea()
        self.sm_scroll.setWidgetResizable(True)
        self.sm_scroll.setObjectName("GalleryScroll")
        self.sm_container = QWidget()
        self.sm_container.setObjectName("PageContainer")
        self.sm_layout = QVBoxLayout(self.sm_container)
        self.sm_layout.setContentsMargins(12, 12, 12, 12)
        self.sm_layout.setSpacing(10)
        self.sm_scroll.setWidget(self.sm_container)
        layout.addWidget(self.sm_scroll, 1)

        # Folder export section
        exp_group = QGroupBox("Export Sorted Images to Folder")
        eg_layout = QHBoxLayout(exp_group)
        eg_layout.setSpacing(10)

        eg_layout.addWidget(QLabel("Group By:"))
        self.export_group_combo = QComboBox()
        self.export_group_combo.setMinimumWidth(210)
        self.export_group_combo.addItem("Flat Ranked (0001_Name.png)", "flat")
        self.export_group_combo.addItem("Folders by Series", "series")
        self.export_group_combo.addItem("Folders by Theme Cluster", "cluster")
        eg_layout.addWidget(self.export_group_combo)

        eg_layout.addSpacing(8)
        eg_layout.addWidget(QLabel("Format:"))
        self.export_fmt_combo = QComboBox()
        self.export_fmt_combo.setMinimumWidth(90)
        for fmt in ImageFormat:
            self.export_fmt_combo.addItem(fmt.value, fmt)
        eg_layout.addWidget(self.export_fmt_combo)

        eg_layout.addSpacing(8)
        export_btn = QPushButton("💾 Export Sorted Folder...")
        export_btn.setObjectName("SuccessBtn")
        export_btn.setMinimumHeight(34)
        export_btn.clicked.connect(self._export_sorted_folder)
        eg_layout.addWidget(export_btn)
        eg_layout.addStretch()
        layout.addWidget(exp_group)
        return page

    # ------------------------------------------------------------------
    # Tab 4: Analytics & Settings
    # ------------------------------------------------------------------

    def _create_settings_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("PageContainer")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("📊 Collection Analytics & Engine Settings")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #FFFFFF;")
        layout.addWidget(title)

        # Metrics row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self._metric_labels = {}
        accent_colors = {
            "total": "#5865F2",
            "series": "#3BA55D",
            "clip": "#9B59B6",
            "ocr": "#FAA61A",
        }
        for key, caption in [
            ("total", "Total Characters"),
            ("series", "Unique Series"),
            ("clip", "CLIP / Vision Indexed"),
            ("ocr", "Avg OCR Confidence"),
        ]:
            card = QFrame()
            card.setObjectName("MetricCard")
            card.setStyleSheet(
                f"#MetricCard {{ border-top: 3px solid {accent_colors[key]}; }}"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(4)
            val_lbl = QLabel("0")
            val_lbl.setObjectName("MetricValue")
            cap_lbl = QLabel(caption)
            cap_lbl.setObjectName("MetricTitle")
            cl.addWidget(val_lbl)
            cl.addWidget(cap_lbl)
            metrics_row.addWidget(card)
            self._metric_labels[key] = val_lbl
        layout.addLayout(metrics_row)

        mid_row = QHBoxLayout()
        mid_row.setSpacing(14)

        # Series breakdown table
        series_box = QGroupBox("Top Series in Collection")
        sb_layout = QVBoxLayout(series_box)
        self.series_table = QTableWidget(0, 2)
        self.series_table.setHorizontalHeaderLabels(["Series", "Cards"])
        self.series_table.verticalHeader().setVisible(False)
        self.series_table.setAlternatingRowColors(False)
        self.series_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.series_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.series_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        sb_layout.addWidget(self.series_table)
        mid_row.addWidget(series_box, 1)

        # Engine settings
        cfg_box = QGroupBox("AI, OCR && Database Settings")
        cf_layout = QVBoxLayout(cfg_box)
        cf_layout.setSpacing(12)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)

        self.enable_clip_chk = QCheckBox("Use Neural Vision Model (SigLIP / CLIP)")
        self.enable_clip_chk.setChecked(self.config.enable_clip)
        form.addRow(self.enable_clip_chk)

        self.vision_model_combo = QComboBox()
        for model_id in SUPPORTED_VISION_MODELS:
            self.vision_model_combo.addItem(model_id, model_id)
        if self.vision_model_combo.findData(self.config.clip_model_name) < 0:
            self.vision_model_combo.addItem(self.config.clip_model_name, self.config.clip_model_name)
        self.vision_model_combo.setCurrentIndex(
            max(0, self.vision_model_combo.findData(self.config.clip_model_name))
        )
        form.addRow("Vision-Language Model:", self.vision_model_combo)

        self.use_gpu_chk = QCheckBox("Use CUDA GPU Acceleration if available")
        self.use_gpu_chk.setChecked(self.config.use_gpu)
        form.addRow(self.use_gpu_chk)

        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setRange(2, 30)
        self.min_cluster_spin.setValue(self.config.min_cluster_size)
        form.addRow("HDBSCAN Min Cluster Size:", self.min_cluster_spin)

        self.dup_thresh_spin = QDoubleSpinBox()
        self.dup_thresh_spin.setRange(0.0, 0.5)
        self.dup_thresh_spin.setSingleStep(0.01)
        self.dup_thresh_spin.setValue(self.config.duplicate_distance_threshold)
        form.addRow("Duplicate Frame Threshold:", self.dup_thresh_spin)

        self.runtime_info_lbl = QLabel(self._detect_runtime_capabilities())
        self.runtime_info_lbl.setObjectName("RuntimeBox")
        self.runtime_info_lbl.setWordWrap(True)
        form.addRow("Runtime:", self.runtime_info_lbl)
        cf_layout.addLayout(form)

        cf_layout.addStretch()

        actions_row = QHBoxLayout()
        actions_row.setSpacing(10)
        save_cfg_btn = QPushButton("💾 Save Settings")
        save_cfg_btn.setObjectName("PrimaryBtn")
        save_cfg_btn.setMinimumHeight(36)
        save_cfg_btn.clicked.connect(self._save_settings_from_ui)

        clear_db_btn = QPushButton("🗑️ Clear Database")
        clear_db_btn.setObjectName("DangerBtn")
        clear_db_btn.setMinimumHeight(36)
        clear_db_btn.clicked.connect(self._clear_database)

        actions_row.addWidget(save_cfg_btn)
        actions_row.addWidget(clear_db_btn)
        actions_row.addStretch()
        cf_layout.addLayout(actions_row)

        mid_row.addWidget(cfg_box, 1)
        layout.addLayout(mid_row, 1)
        return page

    # ------------------------------------------------------------------
    # State & Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_runtime_capabilities() -> str:
        gpu_text = "CPU"
        try:
            import torch

            if torch.cuda.is_available():
                gpu_text = f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass
        return f"PyTorch Device: {gpu_text} | OCR: RapidOCR ONNX | Color Space: OKLab + 2-Opt TSP"

    def _log(self, msg: str) -> None:
        self.activity_log.appendPlainText(msg)

    def _set_status(self, msg: str) -> None:
        self.sidebar_status_lbl.setText(msg)
        self.scrape_status_lbl.setText(msg)

    def _reload_characters_from_db(self) -> None:
        self._characters = self.db.get_all_characters()
        self.ocr.corrector.add_entries(
            names=(c.name for c in self._characters),
            series=(c.series for c in self._characters),
        )
        self._apply_grid_filter()
        self.sidebar_count_lbl.setText(f"Collection: {len(self._characters)} cards")

    def _refresh_region_badges(self) -> None:
        for key, badge in self._region_badges.items():
            val: Optional[RegionTuple] = getattr(self.config, key, None)
            if val:
                w = val[2] - val[0]
                h = val[3] - val[1]
                badge.setText(f"({val[0]}, {val[1]}) → ({val[2]}, {val[3]})  [{w}×{h}px]")
                badge.setObjectName("RegionBadgeSet")
            else:
                badge.setText("Not set")
                badge.setObjectName("RegionBadgeUnset")
            badge.style().unpolish(badge)
            badge.style().polish(badge)

    # ------------------------------------------------------------------
    # Scraper & Region Selection Handlers
    # ------------------------------------------------------------------

    def _browse_output_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(self.config.output_folder))
        if chosen:
            self.config.output_folder = Path(chosen)
            self.output_folder_edit.setText(chosen)
            self.config.save()

    def _start_region_pick(self, config_key: str, label: str, allow_single_click: bool) -> None:
        is_offscreen = QApplication.platformName() == "offscreen"
        was_visible = self.isVisible() and not is_offscreen
        if was_visible:
            self.showMinimized()

        def launch_overlay() -> None:
            overlay = RegionSelectorOverlay(label=label, allow_single_click=allow_single_click)
            self._active_overlay = overlay
            overlay.region_selected.connect(
                lambda region: self._on_region_picked(config_key, region, was_visible)
            )
            overlay.selection_cancelled.connect(lambda: self._on_region_cancelled(was_visible))
            overlay.show()

        QTimer.singleShot(180 if was_visible else 0, launch_overlay)

    def _on_region_picked(self, config_key: str, region: tuple, restore_window: bool) -> None:
        self._active_overlay = None
        if restore_window:
            self.showNormal()
            self.activateWindow()
        setattr(self.config, config_key, tuple(int(v) for v in region))
        self.config.save()
        self._refresh_region_badges()
        self._log(f"Region '{config_key}' set to {region}")
        self._set_status(f"Set {config_key}: {region}")

    def _on_region_cancelled(self, restore_window: bool) -> None:
        self._active_overlay = None
        if restore_window:
            self.showNormal()
            self.activateWindow()
        self._set_status("Region selection cancelled.")

    def _clear_region(self, config_key: str) -> None:
        setattr(self.config, config_key, None)
        self.config.save()
        self._refresh_region_badges()

    def _sync_scraper_config_from_ui(self) -> None:
        self.config.output_folder = Path(self.output_folder_edit.text().strip() or str(self.config.output_folder))
        self.config.capture_count = self.capture_count_spin.value()
        self.config.capture_delay = self.delay_spin.value()
        self.config.auto_scroll = self.auto_scroll_chk.isChecked()
        self.config.restore_cursor = self.restore_cursor_chk.isChecked()
        self.config.save()

    def _start_scraping(self) -> None:
        if self._scraper_worker and self._scraper_worker.isRunning():
            return
        self._sync_scraper_config_from_ui()
        if not self.config.capture_region:
            QMessageBox.warning(self, "Capture Region Missing", "Please select the Card Image Region first.")
            return

        self.start_scrape_btn.setEnabled(False)
        self.stop_scrape_btn.setEnabled(True)
        self.scrape_progress.setValue(0)
        self._scraper_worker = ScraperWorker(self.config, self.db, self.ocr, self.clip_engine)
        self._scraper_worker.progress_updated.connect(self._on_worker_progress)
        self._scraper_worker.card_captured.connect(self._on_card_captured)
        self._scraper_worker.log_message.connect(self._log)
        self._scraper_worker.finished_summary.connect(self._on_scraper_finished)
        self._scraper_worker.start()

    def _stop_scraping(self) -> None:
        if self._scraper_worker and self._scraper_worker.isRunning():
            self._scraper_worker.stop()
            self._set_status("Stopping scraper...")

    def _on_worker_progress(self, current: int, total: int, message: str) -> None:
        pct = int(round((current / max(1, total)) * 100))
        self.scrape_progress.setValue(pct)
        self._set_status(message)

    def _on_card_captured(self, character: Character) -> None:
        pm = QPixmap(str(character.image_path))
        if not pm.isNull():
            self.preview_img_lbl.setPixmap(
                pm.scaled(200, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        self.preview_meta_lbl.setText(f"{character.name}\n({character.series})")

    def _on_scraper_finished(self, captured: int, skipped: int, summary: str) -> None:
        self.start_scrape_btn.setEnabled(True)
        self.stop_scrape_btn.setEnabled(False)
        self.scrape_progress.setValue(100)
        self._log(summary)
        self._set_status(summary)
        self._reload_characters_from_db()
        self._refresh_analytics()

    def _import_local_folder(self) -> None:
        if self._import_worker and self._import_worker.isRunning():
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Character Images", str(self.config.output_folder))
        if not folder:
            return
        self._sync_scraper_config_from_ui()
        self.import_folder_btn.setEnabled(False)
        self.scrape_progress.setValue(0)
        self._import_worker = FolderImportWorker(Path(folder), self.config, self.db, self.clip_engine)
        self._import_worker.progress_updated.connect(self._on_worker_progress)
        self._import_worker.log_message.connect(self._log)
        self._import_worker.finished_summary.connect(self._on_import_finished)
        self._import_worker.start()

    def _on_import_finished(self, count: int, summary: str) -> None:
        self.import_folder_btn.setEnabled(True)
        self.scrape_progress.setValue(100)
        self._log(summary)
        self._set_status(summary)
        self._reload_characters_from_db()
        self._refresh_analytics()

    def _open_text_list_import(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Import Mudae Text List ($mm / $mms / $mma / $sm)")
        dlg.resize(560, 400)
        v = QVBoxLayout(dlg)
        hint = QLabel(
            "Paste your Discord Mudae $mm / $mms / $mmk / $mma / $sm output below.\n"
            "This teaches the OCR typo corrector exact character & series names and updates matching cards."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        editor = QPlainTextEdit()
        editor.setPlaceholderText("1. Zero Two - Darling in the FranXX · 520 ka\n2. Rem - Re:Zero · 490 ka\n...")
        v.addWidget(editor, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        parsed = parse_mudae_text_list(editor.toPlainText())
        if not parsed:
            QMessageBox.information(self, "No Entries Parsed", "Could not detect character names in pasted text.")
            return

        self.ocr.corrector.add_entries(
            names=(n for n, _ in parsed),
            series=(s for _, s in parsed if s),
        )

        # Snap any existing cards with noisy OCR names to the parsed lexicon
        updated = 0
        for ch in self._characters:
            best_pair = None
            best_ratio = 0.0
            for name, series in parsed:
                r = difflib.SequenceMatcher(None, ch.name.lower(), name.lower()).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_pair = (name, series)
            if best_pair and best_ratio >= 0.80:
                new_name, new_series = best_pair
                target_series = new_series or ch.series
                if ch.name != new_name or (new_series and ch.series != target_series):
                    self.db.update_character_fields(ch.id, name=new_name, series=target_series)
                    updated += 1

        self._reload_characters_from_db()
        self._rebuild_sm_commands()
        self._refresh_analytics()
        summary = f"Loaded {len(parsed)} Mudae lexicon names; updated {updated} matching cards."
        self._log(summary)
        self._set_status(summary)

    def _generate_demo_cards(self) -> None:
        if self._demo_worker and self._demo_worker.isRunning():
            return
        self._sync_scraper_config_from_ui()
        self.demo_cards_btn.setEnabled(False)
        self.scrape_progress.setValue(0)
        self._demo_worker = DemoGeneratorWorker(self.config, self.db, self.clip_engine)
        self._demo_worker.progress_updated.connect(self._on_worker_progress)
        self._demo_worker.finished_summary.connect(self._on_demo_finished)
        self._demo_worker.start()

    def _on_demo_finished(self, count: int, summary: str) -> None:
        self.demo_cards_btn.setEnabled(True)
        self.scrape_progress.setValue(100)
        self._log(summary)
        self._set_status(summary)
        self._reload_characters_from_db()
        self._refresh_analytics()

    # ------------------------------------------------------------------
    # Sorter & Gallery Handlers
    # ------------------------------------------------------------------

    def _on_columns_changed(self) -> None:
        cols = int(self.columns_combo.currentData() or 5)
        self.config.grid_columns = cols
        self.config.save()
        self.gallery.set_columns(cols)

    def _apply_grid_filter(self) -> None:
        q = self.filter_edit.text().strip().lower() if hasattr(self, "filter_edit") else ""
        if not q:
            self.gallery.set_characters(self._characters)
            return
        filtered = [
            c
            for c in self._characters
            if q in c.name.lower() or q in c.series.lower() or any(q in t.lower() for t in c.tags)
        ]
        self.gallery.set_characters(filtered)

    def _run_prompt_sort(self) -> None:
        prompt = self.prompt_edit.text().strip()
        if not prompt:
            QMessageBox.information(self, "Visual Prompt", "Enter a prompt like 'pink hair' or 'dark background'.")
            return
        self._run_sort(SortMethod.NATURAL_LANGUAGE, prompt=prompt)

    def _run_sort(self, method: SortMethod, prompt: str = "") -> None:
        if not self._characters:
            QMessageBox.information(self, "Empty Collection", "Import or capture characters first.")
            return
        if self._sort_worker and self._sort_worker.isRunning():
            return

        for b in self._sort_buttons:
            b.setEnabled(False)
        self.sort_status_lbl.setText("Computing sort order...")
        self.config.last_sort_method = method
        self.config.save()

        self._sort_worker = SortWorker(
            characters=self._characters,
            method=method,
            db=self.db,
            clip_engine=self.clip_engine,
            min_cluster_size=self.config.min_cluster_size,
            prompt=prompt,
            enable_neural=self.config.enable_clip,
        )
        self._sort_worker.finished_sorted.connect(self._on_sort_finished)
        self._sort_worker.failed.connect(self._on_sort_failed)
        self._sort_worker.start()

    def _on_sort_finished(self, sorted_chars: list, msg: str) -> None:
        for b in self._sort_buttons:
            b.setEnabled(True)
        self._characters = list(sorted_chars)
        self._apply_grid_filter()
        self.sort_status_lbl.setText(msg)
        self._set_status(msg)
        self._rebuild_sm_commands()
        self._refresh_analytics()

    def _on_sort_failed(self, err: str) -> None:
        for b in self._sort_buttons:
            b.setEnabled(True)
        self.sort_status_lbl.setText(err)
        QMessageBox.warning(self, "Sort Error", err)

    def _on_gallery_order_changed(self, new_order: list) -> None:
        if not self.filter_edit.text().strip():
            self._characters = list(new_order)
            self.db.update_sort_orders(self._characters)
            self._rebuild_sm_commands()
            self.sort_status_lbl.setText("Manual card order saved.")

    def _on_gallery_selection_changed(self, idx: int) -> None:
        if idx >= 0:
            self.sort_status_lbl.setText(f"Selected card #{idx + 1}. Click another card to swap positions.")
        else:
            self.sort_status_lbl.setText("Selection cleared.")

    def _on_character_edited(self, char: Character) -> None:
        self.db.update_character_fields(char.id, name=char.name, series=char.series, tags=char.tags)
        self._rebuild_sm_commands()
        self._refresh_analytics()
        self._set_status(f"Updated '{char.name}'")

    def _on_character_deleted(self, char: Character) -> None:
        self.db.delete_character(char.id)
        self._characters = [c for c in self._characters if c.id != char.id]
        self.db.update_sort_orders(self._characters)
        self.sidebar_count_lbl.setText(f"Collection: {len(self._characters)} cards")
        self._rebuild_sm_commands()
        self._refresh_analytics()
        self._set_status(f"Removed '{char.name}'")

    # ------------------------------------------------------------------
    # Mudae $sm & Folder Export Handlers
    # ------------------------------------------------------------------

    def _clear_sm_batches(self) -> None:
        while self.sm_layout.count():
            item = self.sm_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _current_sm_batches(self) -> List[str]:
        return generate_sm_commands(
            self._characters,
            prefix=self.prefix_edit.text().strip() or "$",
            command=self.cmd_edit.text().strip() or "sm",
            max_length=self.max_chars_spin.value(),
            use_pos_chaining=self.pos_chain_chk.isChecked(),
        )

    def _rebuild_sm_commands(self) -> None:
        self._clear_sm_batches()
        batches = self._current_sm_batches()
        if not batches:
            hint = QLabel("No characters available yet. Capture or import cards to generate $sm commands.")
            hint.setObjectName("EmptyGalleryHint")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sm_layout.addWidget(hint)
            return

        for idx, cmd in enumerate(batches, start=1):
            box = QGroupBox(f"Batch #{idx} of {len(batches)}  —  {len(cmd)} / {self.max_chars_spin.value()} chars")
            bl = QVBoxLayout(box)
            bl.setSpacing(8)
            text_box = QPlainTextEdit(cmd)
            text_box.setReadOnly(True)
            text_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            text_box.setMinimumHeight(76)
            text_box.setMaximumHeight(110)
            bl.addWidget(text_box)

            row = QHBoxLayout()
            copy_btn = QPushButton(f"📋 Copy Batch #{idx}")
            copy_btn.setObjectName("PrimaryBtn")
            copy_btn.setMinimumHeight(32)
            copy_btn.clicked.connect(lambda _=False, c=cmd, i=idx: self._copy_text_to_clipboard(c, f"Batch #{i}"))
            row.addWidget(copy_btn)
            row.addStretch()
            bl.addLayout(row)
            self.sm_layout.addWidget(box)

        self.sm_layout.addStretch()

    def _copy_text_to_clipboard(self, text: str, label: str) -> None:
        QApplication.clipboard().setText(text)
        self._set_status(f"Copied {label} to clipboard ({len(text)} chars).")

    def _copy_all_sm_batches(self) -> None:
        batches = self._current_sm_batches()
        if not batches:
            return
        self._copy_text_to_clipboard("\n\n".join(batches), f"all {len(batches)} $sm batches")

    def _copy_plain_list(self) -> None:
        text = generate_mma_text(self._characters)
        if text:
            self._copy_text_to_clipboard(text, "plain character list")

    def _export_sorted_folder(self) -> None:
        if not self._characters:
            QMessageBox.information(self, "Empty Collection", "No characters to export.")
            return
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Export Destination Folder", str(self.config.output_folder)
        )
        if not chosen:
            return
        group_by = self.export_group_combo.currentData()
        img_fmt: ImageFormat = self.export_fmt_combo.currentData()
        count = export_sorted_folder(
            self._characters,
            Path(chosen),
            group_by=group_by,
            image_format=img_fmt,
            jpeg_quality=self.config.jpeg_quality,
        )
        QMessageBox.information(self, "Export Complete", f"Exported {count} sorted images to:\n{chosen}")
        self._set_status(f"Exported {count} images to {chosen}")

    # ------------------------------------------------------------------
    # Analytics & Settings Handlers
    # ------------------------------------------------------------------

    def _refresh_analytics(self) -> None:
        summary = self.db.get_analytics_summary()
        self._metric_labels["total"].setText(str(summary["total_characters"]))
        self._metric_labels["series"].setText(str(summary["unique_series"]))
        self._metric_labels["clip"].setText(str(summary["clip_indexed"]))
        self._metric_labels["ocr"].setText(f"{summary['avg_ocr_confidence'] * 100:.0f}%")

        top_series = summary["top_series"]
        self.series_table.setRowCount(len(top_series))
        for row, (series, cnt) in enumerate(top_series):
            self.series_table.setItem(row, 0, QTableWidgetItem(series))
            self.series_table.setItem(row, 1, QTableWidgetItem(str(cnt)))

    def _save_settings_from_ui(self) -> None:
        self.config.enable_clip = self.enable_clip_chk.isChecked()
        self.config.clip_model_name = str(self.vision_model_combo.currentData() or self.config.clip_model_name)
        self.config.use_gpu = self.use_gpu_chk.isChecked()
        self.config.min_cluster_size = self.min_cluster_spin.value()
        self.config.duplicate_distance_threshold = self.dup_thresh_spin.value()
        self.config.mudae_prefix = self.prefix_edit.text().strip() or "$"
        self.config.mudae_command = self.cmd_edit.text().strip() or "sm"
        self.config.mudae_max_chars = self.max_chars_spin.value()
        self.config.use_pos_chaining = self.pos_chain_chk.isChecked()
        self.config.save()
        self.clip_engine.use_gpu = self.config.use_gpu
        self.clip_engine.set_model_name(self.config.clip_model_name)
        self._set_status("Settings saved.")

    def _clear_database(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear Database",
            "Remove all character entries from the local database? (Image files on disk will NOT be deleted.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            removed = self.db.clear_all()
            self._reload_characters_from_db()
            self._rebuild_sm_commands()
            self._refresh_analytics()
            self._set_status(f"Cleared {removed} records from database.")
