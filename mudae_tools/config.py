"""Configuration, logging setup, and Pydantic data models for Mudae Power Tools Pro."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, ConfigDict, Field

from . import APP_NAME, APP_VERSION

CONFIG_DIR = Path.home() / ".mudae_tools"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATABASE_FILE = CONFIG_DIR / "characters.db"
DEFAULT_OUTPUT_FOLDER = Path.home() / "MudaeCharacters"


def configure_logging(log_dir: Optional[Path] = None) -> structlog.stdlib.BoundLogger:
    """Configure structured logging to console and rotating-safe file."""
    target_dir = log_dir or CONFIG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.FileHandler(target_dir / "mudae_tools.log", encoding="utf-8"),
    ]
    if sys.stderr is not None:
        handlers.insert(0, logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=handlers,
    )
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if False else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("mudae_tools")


logger = configure_logging()


class SortMethod(str, Enum):
    COLOR_GRADIENT_TSP = "color_gradient_tsp"
    HILBERT_COLOR = "hilbert_color"
    RAINBOW_SPECTRUM = "rainbow_spectrum"
    VISUAL_SIMILARITY_TSP = "visual_similarity_tsp"
    THEME_CLUSTERS = "theme_clusters"
    SERIES_GROUPING = "series_grouping"
    ALPHABETICAL = "alphabetical"
    NATURAL_LANGUAGE = "natural_language"


SUPPORTED_VISION_MODELS = (
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
)


class ImageFormat(str, Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"


RegionTuple = Tuple[int, int, int, int]


class AppConfig(BaseModel):
    """Application settings persisted as JSON."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    output_folder: Path = Field(default_factory=lambda: DEFAULT_OUTPUT_FOLDER)
    image_format: ImageFormat = ImageFormat.PNG
    jpeg_quality: int = Field(default=95, ge=1, le=100)

    # Screen regions (physical pixels: x1, y1, x2, y2)
    capture_region: Optional[RegionTuple] = None
    character_name_region: Optional[RegionTuple] = None
    show_name_region: Optional[RegionTuple] = None
    next_button_region: Optional[RegionTuple] = None

    # Scraper controls
    capture_count: int = Field(default=50, ge=1, le=5000)
    capture_delay: float = Field(default=1.2, ge=0.2, le=30.0)
    auto_scroll: bool = True
    restore_cursor: bool = True
    duplicate_distance_threshold: float = Field(default=0.06, ge=0.0, le=0.5)
    ocr_min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)

    # AI / Vision settings
    use_gpu: bool = True
    enable_clip: bool = True
    clip_model_name: str = "openai/clip-vit-base-patch32"
    min_cluster_size: int = Field(default=2, ge=2, le=50)

    # Mudae export settings
    mudae_prefix: str = "$"
    mudae_command: str = "sm"
    mudae_max_chars: int = Field(default=1950, ge=200, le=2000)
    use_pos_chaining: bool = True

    # UI state
    grid_columns: int = Field(default=5, ge=2, le=12)
    last_sort_method: SortMethod = SortMethod.COLOR_GRADIENT_TSP

    def save(self, path: Path = CONFIG_FILE) -> None:
        """Persist configuration to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "AppConfig":
        """Load configuration from JSON, returning defaults if missing or corrupt."""
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if "output_folder" in raw and isinstance(raw["output_folder"], str):
                raw["output_folder"] = Path(raw["output_folder"])
            for reg_key in (
                "capture_region",
                "character_name_region",
                "show_name_region",
                "next_button_region",
            ):
                if isinstance(raw.get(reg_key), list) and len(raw[reg_key]) == 4:
                    raw[reg_key] = tuple(int(v) for v in raw[reg_key])
            return cls.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed to load config; using defaults", error=str(exc))
            return cls()


class Character(BaseModel):
    """Represents a Mudae character card and its visual/semantic descriptors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    series: str = Field(default="Unknown Series")
    image_path: Path
    phash: str = Field(default="")
    dominant_hex: List[str] = Field(default_factory=list)
    oklab_feature: Optional[List[float]] = None
    clip_embedding: Optional[List[float]] = None
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cluster_id: int = Field(default=-1)
    sort_index: int = Field(default=0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
