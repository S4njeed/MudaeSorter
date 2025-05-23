"""
Mudae Power Tools - Modern Rewrite
A high-performance, async tool for scraping and sorting Mudae character images.
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import uuid
import tkinter as tk
from tkinter import messagebox
import threading

# Add these imports at the top of your file
import pyautogui
import time
from PIL import ImageGrab

# Modern UI Framework
import customtkinter as ctk

# High-performance image processing
import cv2
import numpy as np
from PIL import Image, ImageTk
import imagehash

# Modern OCR
import easyocr

# Machine Learning
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Async and performance
import aiofiles
import aiohttp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# System integration
import mss
import pyautogui
import keyboard
from pynput import mouse, keyboard as kb

# Data validation
from pydantic import BaseModel, Field, validator
import msgpack

# Rich console output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Logging
import structlog

# Configuration
console = Console()

# Configure structured logging
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ================================
# Constants and Configuration
# ================================

APP_NAME = "Mudae Power Tools Pro"
APP_VERSION = "2.0.0"
CONFIG_DIR = Path.home() / ".mudae_tools"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATABASE_FILE = CONFIG_DIR / "characters.db"

class OCREngine(Enum):
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"

class SortMethod(Enum):
    VISUAL_SIMILARITY = "visual_similarity"
    ALPHABETICAL = "alphabetical"
    SHOW_GROUPING = "show_grouping"
    NEURAL_CLUSTERING = "neural_clustering"

class ImageFormat(Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    AVIF = "AVIF"

# ================================
# Data Models
# ================================

class AppConfig(BaseModel):
    """Application configuration with validation"""
    output_folder: Path = Path.home() / "MudaeCharacters"
    ocr_engine: OCREngine = OCREngine.EASYOCR
    image_format: ImageFormat = ImageFormat.PNG
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    capture_region: Optional[Tuple[int, int, int, int]] = None
    next_button_region: Optional[Tuple[int, int, int, int]] = None
    character_name_region: Optional[Tuple[int, int, int, int]] = None
    show_name_region: Optional[Tuple[int, int, int, int]] = None
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=128)
    max_workers: int = Field(default=4, ge=1, le=16)
    use_gpu: bool = True
    auto_backup: bool = True
    theme: str = "dark"
    
    class Config:
        arbitrary_types_allowed = True
        
    def save(self, path: Path = CONFIG_FILE):
        """Save configuration to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = self.dict()
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, Enum):
                config_dict[key] = value.value
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> 'AppConfig':
        """Load configuration from file"""
        if not path.exists():
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert strings back to appropriate types
        if 'output_folder' in data:
            data['output_folder'] = Path(data['output_folder'])
        if 'ocr_engine' in data:
            data['ocr_engine'] = OCREngine(data['ocr_engine'])
        if 'image_format' in data:
            data['image_format'] = ImageFormat(data['image_format'])
        
        return cls(**data)

class Character(BaseModel):
    """Character data model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    show: str = Field(default="Unknown Show")
    image_path: Path
    hash_value: str
    embedding: Optional[List[float]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    class Config:
        arbitrary_types_allowed = True

# ================================
# Modern Database Manager
# ================================

class DatabaseManager:
    """High-performance SQLite database manager with migrations"""
    
    def __init__(self, db_path: Path = DATABASE_FILE):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with proper schema and indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            
            # Characters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS characters (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    show_name TEXT NOT NULL,
                    image_path TEXT NOT NULL UNIQUE,
                    hash_value TEXT NOT NULL,
                    embedding BLOB,
                    confidence REAL DEFAULT 0.0,
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL DEFAULT (unixepoch()),
                    updated_at REAL DEFAULT (unixepoch())
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_show ON characters(show_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_hash ON characters(hash_value)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_updated ON characters(updated_at)")
            
            # Full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS characters_fts USING fts5(
                    name, show_name, tags, content=characters, content_rowid=rowid
                )
            """)
            
            # Triggers for FTS
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS characters_fts_insert AFTER INSERT ON characters
                BEGIN
                    INSERT INTO characters_fts(rowid, name, show_name, tags)
                    VALUES (new.rowid, new.name, new.show_name, new.tags);
                END
            """)
    
    async def add_character(self, character: Character) -> bool:
        """Add character to database asynchronously"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO characters 
                    (id, name, show_name, image_path, hash_value, embedding, confidence, tags, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    character.id,
                    character.name,
                    character.show,
                    str(character.image_path),
                    character.hash_value,
                    msgpack.packb(character.embedding) if character.embedding else None,
                    character.confidence,
                    json.dumps(character.tags),
                    json.dumps(character.metadata),
                    time.time()
                ))
            return True
        except Exception as e:
            logger.error("Failed to add character", character_id=character.id, error=str(e))
            return False
    
    async def search_characters(self, query: str) -> List[Character]:
        """Full-text search for characters"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT c.* FROM characters c
                JOIN characters_fts fts ON c.rowid = fts.rowid
                WHERE characters_fts MATCH ?
                ORDER BY rank
            """, (query,))
            
            results = []
            for row in cursor.fetchall():
                character = self._row_to_character(row)
                if character:
                    results.append(character)
            return results
    
    def _row_to_character(self, row) -> Optional[Character]:
        """Convert database row to Character object"""
        try:
            embedding = None
            if row[5]:  # embedding column
                embedding = msgpack.unpackb(row[5])
            
            return Character(
                id=row[0],
                name=row[1],
                show=row[2],
                image_path=Path(row[3]),
                hash_value=row[4],
                embedding=embedding,
                confidence=row[6],
                tags=json.loads(row[7]),
                metadata=json.loads(row[8]),
                created_at=row[9],
                updated_at=row[10]
            )
        except Exception as e:
            logger.error("Failed to convert row to character", error=str(e))
            return None

# ================================
# Modern OCR Engine
# ================================

class ModernOCR:
    """High-performance OCR with multiple backends"""
    
    def __init__(self, engine: OCREngine = OCREngine.EASYOCR):
        self.engine = engine
        self.reader = None
        self._init_reader()
    
    def _init_reader(self):
        """Initialize OCR reader"""
        try:
            if self.engine == OCREngine.EASYOCR:
                self.reader = easyocr.Reader(['en', 'ja'], gpu=torch.cuda.is_available())
            elif self.engine == OCREngine.PADDLEOCR:
                from paddleocr import PaddleOCR
                self.reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        except Exception as e:
            logger.error("Failed to initialize OCR", engine=self.engine.value, error=str(e))
    
    async def extract_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Extract text from image asynchronously"""
        if self.reader is None:
            return ""
        
        try:
            # Crop to region if specified
            if region:
                x1, y1, x2, y2 = region
                image = image[y1:y2, x1:x2]
            
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                if self.engine == OCREngine.EASYOCR:
                    result = await loop.run_in_executor(executor, self.reader.readtext, image)
                    text = " ".join([item[1] for item in result if item[2] > 0.5])  # Confidence > 0.5
                else:  # PaddleOCR
                    result = await loop.run_in_executor(executor, self.reader.ocr, image)
                    text = " ".join([line[1][0] for line in result[0] if line[1][1] > 0.5])
                
                return text.strip()
        except Exception as e:
            logger.error("OCR extraction failed", error=str(e))
            return ""

# ================================
# AI-Powered Image Processor
# ================================

class AIImageProcessor:
    """Advanced image processing with neural networks"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.models = {}
        self.transforms = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize AI models"""
        try:
            # Object detection model (YOLO)
            self.models['yolo'] = YOLO('yolov8n.pt')
            
            # Feature extraction model
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.models['feature_extractor'] = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.models['feature_extractor'].eval()
            self.models['feature_extractor'].to(self.device)
            
            # Image transforms
            self.transforms['feature'] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("AI models initialized", device=str(self.device))
        except Exception as e:
            logger.error("Failed to initialize AI models", error=str(e))
    
    async def detect_characters(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect character cards in image using YOLO"""
        try:
            results = self.models['yolo'](image)
            bboxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.conf > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            bboxes.append((x1, y1, x2, y2))
            
            return bboxes
        except Exception as e:
            logger.error("Character detection failed", error=str(e))
            return []
    
    async def extract_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract deep features from image"""
        try:
            if 'feature_extractor' not in self.models:
                return None
            
            # Preprocess image
            image_tensor = self.transforms['feature'](image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.models['feature_extractor'](image_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            return None

# ================================
# Modern Screen Capture
# ================================

class ScreenCapture:
    """High-performance screen capture using MSS"""
    
    def __init__(self):
        self.sct = mss.mss()
    
    async def capture_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture screen region asynchronously"""
        try:
            x1, y1, x2, y2 = region
            monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
            
            # Capture in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                screenshot = await loop.run_in_executor(executor, self.sct.grab, monitor)
                return np.array(screenshot)[:, :, :3]  # Remove alpha channel
        except Exception as e:
            logger.error("Screen capture failed", region=region, error=str(e))
            return None

# ================================
# Advanced Image Hasher
# ================================

class ImageHasher:
    """Advanced perceptual hashing for duplicate detection"""
    
    @staticmethod
    def compute_hash(image: Image.Image, hash_size: int = 16) -> str:
        """Compute perceptual hash of image"""
        try:
            # Use multiple hash algorithms for better accuracy
            dhash = str(imagehash.dhash(image, hash_size))
            phash = str(imagehash.phash(image, hash_size))
            whash = str(imagehash.whash(image, hash_size))
            
            # Combine hashes
            combined = f"{dhash}_{phash}_{whash}"
            return combined
        except Exception as e:
            logger.error("Hash computation failed", error=str(e))
            return ""
    
    @staticmethod
    def hash_distance(hash1: str, hash2: str) -> float:
        """Compute distance between two hashes"""
        try:
            if not hash1 or not hash2:
                return float('inf')
            
            # Split combined hashes
            h1_parts = hash1.split('_')
            h2_parts = hash2.split('_')
            
            if len(h1_parts) != 3 or len(h2_parts) != 3:
                return float('inf')
            
            # Compute distances for each hash type
            distances = []
            for i in range(3):
                try:
                    if i == 0:  # dhash
                        h1 = imagehash.hex_to_hash(h1_parts[i])
                        h2 = imagehash.hex_to_hash(h2_parts[i])
                    elif i == 1:  # phash
                        h1 = imagehash.hex_to_hash(h1_parts[i])
                        h2 = imagehash.hex_to_hash(h2_parts[i])
                    else:  # whash
                        h1 = imagehash.hex_to_hash(h1_parts[i])
                        h2 = imagehash.hex_to_hash(h2_parts[i])
                    
                    distances.append(h1 - h2)
                except:
                    distances.append(float('inf'))
            
            # Return average distance
            valid_distances = [d for d in distances if d != float('inf')]
            return sum(valid_distances) / len(valid_distances) if valid_distances else float('inf')
        except Exception as e:
            logger.error("Hash distance computation failed", error=str(e))
            return float('inf')

# ================================
# Modern UI Components
# ================================

class RegionSelector:
    """Interactive region selector for screen capture"""
    
    def __init__(self, callback):
        self.callback = callback
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.canvas = None
        self.root = None
        
    def select_region(self):
        """Start region selection process"""
        # Create fullscreen window for selection
        self.root = tk.Toplevel()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.configure(bg='black')
        self.root.attributes('-topmost', True)
        
        # Create canvas for drawing selection rectangle
        self.canvas = tk.Canvas(
            self.root, 
            highlightthickness=0,
            bg='black'
        )
        self.canvas.pack(fill='both', expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
        # Instructions
        label = tk.Label(
            self.root,
            text="Click and drag to select region. Press ESC to cancel.",
            fg='white',
            bg='black',
            font=('Arial', 16)
        )
        label.pack(pady=20)
        
        # Bind escape key
        self.root.bind('<Escape>', lambda e: self.cancel_selection())
        self.root.focus_set()
        
    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def on_drag(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
        
    def on_release(self, event):
        if self.start_x and self.start_y:
            # Get actual screen coordinates
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            region = (x1, y1, x2, y2)
            self.callback(region)
            self.root.destroy()
            
    def cancel_selection(self):
        self.root.destroy()

class ModernUI:
    """Modern UI using CustomTkinter"""
    
    def __init__(self):
        self.app = None
        self.config = AppConfig.load()
        self.db = DatabaseManager()
        self.ocr = ModernOCR(self.config.ocr_engine)
        self.ai_processor = AIImageProcessor(self.config.use_gpu)
        self.screen_capture = ScreenCapture()
        self.is_capturing = False
        self.captured_images = []
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the modern UI"""
        # Set theme
        ctk.set_appearance_mode(self.config.theme)
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.app = ctk.CTk()
        self.app.title(f"{APP_NAME} v{APP_VERSION}")
        self.app.geometry("1200x800")
        self.app.minsize(1000, 700)
        
        # Configure grid
        self.app.grid_columnconfigure(1, weight=1)
        self.app.grid_rowconfigure(0, weight=1)
        
        # Create sidebar
        self._create_sidebar()
        
        # Create main content area
        self._create_main_content()
        
        # Show scraper tab by default after all frames are created
        self.app.after(0, lambda: self._show_scraper_tab())
        
        # Status bar
        self._create_status_bar()
    
    def _create_sidebar(self):
        """Create modern sidebar navigation"""
        self.sidebar = ctk.CTkFrame(self.app, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(8, weight=1)
        
        # Logo/Title
        title_label = ctk.CTkLabel(
            self.sidebar, 
            text=APP_NAME, 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Navigation buttons
        nav_buttons = [
            ("🖼️ Scraper", self._show_scraper_tab),
            ("🎯 Sorter", self._show_sorter_tab),
            ("📊 Analytics", self._show_analytics_tab),
            ("⚙️ Settings", self._show_settings_tab),
        ]
        
        self.nav_buttons = []
        for i, (text, command) in enumerate(nav_buttons, 1):
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=40,
                font=ctk.CTkFont(size=14)
            )
            btn.grid(row=i, column=0, padx=20, pady=10, sticky="ew")
            self.nav_buttons.append(btn)
        
        # Theme toggle
        theme_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Dark Mode",
            command=self._toggle_theme
        )
        theme_switch.grid(row=7, column=0, padx=20, pady=10)
        if self.config.theme == "dark":
            theme_switch.select()
    
    def _create_main_content(self):
        """Create main content area with tabs"""
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Create tab frames
        self.scraper_frame = self._create_scraper_tab()
        self.sorter_frame = self._create_sorter_tab()
        self.analytics_frame = self._create_analytics_tab()
        self.settings_frame = self._create_settings_tab()
        
        # Show scraper tab by default
        # self._show_scraper_tab() # Moved to _init_ui
    
    def _create_scraper_tab(self):
        """Create modern scraper interface"""
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(4, weight=1)  # Changed to accommodate preview
        
        # Header
        header = ctk.CTkLabel(
            frame,
            text="🖼️ Character Scraper",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.grid(row=0, column=0, padx=20, pady=20)
        
        # Configuration section
        config_frame = ctk.CTkFrame(frame)
        config_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        config_frame.grid_columnconfigure(1, weight=1)
        
        # Output folder
        ctk.CTkLabel(config_frame, text="Output Folder:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.output_folder_entry = ctk.CTkEntry(config_frame, placeholder_text="Select output folder...")
        self.output_folder_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.output_folder_entry.insert(0, str(self.config.output_folder))
        
        folder_btn = ctk.CTkButton(
            config_frame,
            text="Browse",
            command=self._browse_output_folder,
            width=100
        )
        folder_btn.grid(row=0, column=2, padx=10, pady=10)
        
        # Region selection buttons
        region_frame = ctk.CTkFrame(frame)
        region_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        
        region_buttons = [
            ("Select Capture Region", self._select_capture_region_sync),
            ("Select Next Button", self._select_next_button_sync),
            ("Start Scraping", self._start_scraping_sync),
            ("Stop Scraping", self._stop_scraping)
        ]
        
        for i, (text, command) in enumerate(region_buttons):
            btn = ctk.CTkButton(
                region_frame,
                text=text,
                command=command,
                height=40
            )
            btn.grid(row=0, column=i, padx=10, pady=20, sticky="ew")
            region_frame.grid_columnconfigure(i, weight=1)
        
        # Progress and status
        self.progress_frame = ctk.CTkFrame(frame)
        self.progress_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        self.progress_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(self.progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Settings frame
        settings_frame = ctk.CTkFrame(frame)
        settings_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=10)
        settings_frame.grid_columnconfigure(1, weight=1)
        
        # Capture settings
        ctk.CTkLabel(settings_frame, text="Images to capture:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.capture_count_entry = ctk.CTkEntry(settings_frame, width=100, placeholder_text="50")
        self.capture_count_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.capture_count_entry.insert(0, "50")
        
        ctk.CTkLabel(settings_frame, text="Delay between captures (seconds):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.delay_entry = ctk.CTkEntry(settings_frame, width=100, placeholder_text="1.0")
        self.delay_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.delay_entry.insert(0, "1.0")
        
        # Auto-scroll checkbox
        self.auto_scroll_var = ctk.BooleanVar(value=True)
        self.auto_scroll_check = ctk.CTkCheckBox(
            settings_frame,
            text="Auto-scroll to next character",
            variable=self.auto_scroll_var
        )
        self.auto_scroll_check.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Preview area
        preview_frame = ctk.CTkFrame(frame)
        preview_frame.grid(row=5, column=0, sticky="ew", padx=20, pady=10)
        
        ctk.CTkLabel(preview_frame, text="Last Captured:", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        
        self.preview_label = ctk.CTkLabel(preview_frame, text="No image captured yet")
        self.preview_label.pack(pady=10)
        
        return frame
    
    # Synchronous wrapper functions for button commands
    def _select_capture_region_sync(self):
        """Synchronous wrapper for region selection"""
        self.status_label.configure(text="Click and drag to select capture region...")
        selector = RegionSelector(self._on_capture_region_selected)
        selector.select_region()
    
    def _select_next_button_sync(self):
        """Synchronous wrapper for next button selection"""
        self.status_label.configure(text="Click on the next button...")
        selector = RegionSelector(self._on_next_button_selected)
        selector.select_region()
    
    def _start_scraping_sync(self):
        """Start scraping in a separate thread"""
        if self.is_capturing:
            return
            
        if not self.config.capture_region:
            self.status_label.configure(text="Please select capture region first!")
            return
            
        self.is_capturing = True
        thread = threading.Thread(target=self._run_scraping_async, daemon=True)
        thread.start()
    
    def _stop_scraping(self):
        """Stop the scraping process"""
        self.is_capturing = False
        self.status_label.configure(text="Stopping scraper...")
    
    def _run_scraping_async(self):
        """Run the async scraping in a thread"""
        asyncio.run(self._start_scraping())
    
    def _on_capture_region_selected(self, region):
        """Callback when capture region is selected"""
        self.config.capture_region = region
        self.config.save()
        self.status_label.configure(text=f"Capture region set: {region}")
        logger.info("Capture region selected", region=region)
    
    def _on_next_button_selected(self, region):
        """Callback when next button region is selected"""
        self.config.next_button_region = region
        self.config.save()
        self.status_label.configure(text=f"Next button region set: {region}")
        logger.info("Next button region selected", region=region)
    
    async def _start_scraping(self):
        """Main scraping logic"""
        try:
            capture_count = int(self.capture_count_entry.get() or "50")
            delay = float(self.delay_entry.get() or "1.0")
            
            self.status_label.configure(text="Starting scraper...")
            self.progress_bar.set(0)
            self.captured_images = []
            
            # Ensure output directory exists
            output_dir = Path(self.config.output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(capture_count):
                if not self.is_capturing:
                    break
                
                # Update status
                self.status_label.configure(text=f"Capturing image {i + 1}/{capture_count}")
                
                # Capture the character region
                screenshot = await self.screen_capture.capture_region(self.config.capture_region)
                
                if screenshot is not None:
                    # Save the image
                    timestamp = int(time.time() * 1000)
                    filename = f"character_{timestamp}_{i:04d}.png"
                    filepath = output_dir / filename
                    
                    # Convert numpy array to PIL Image and save
                    image = Image.fromarray(screenshot)
                    image.save(filepath)
                    
                    self.captured_images.append(filepath)
                    
                    # Update preview (run in main thread)
                    self.app.after(0, self._update_preview, image)
                    
                    logger.info("Image captured", filepath=str(filepath))
                    
                    # Extract character name using OCR if name region is set
                    if self.config.character_name_region:
                        name_screenshot = await self.screen_capture.capture_region(self.config.character_name_region)
                        if name_screenshot is not None:
                            character_name = await self.ocr.extract_text(name_screenshot)
                            if character_name:
                                # Rename file with character name
                                clean_name = "".join(c for c in character_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                new_filename = f"{clean_name}_{timestamp}.png"
                                new_filepath = output_dir / new_filename
                                filepath.rename(new_filepath)
                                logger.info("Character identified", name=character_name, filepath=str(new_filepath))
                
                # Update progress
                progress = (i + 1) / capture_count
                self.progress_bar.set(progress)
                
                # Auto-scroll to next character if enabled and region is set
                if self.auto_scroll_var.get() and self.config.next_button_region and i < capture_count - 1:
                    await self._click_next_button()
                
                # Wait before next capture
                await asyncio.sleep(delay)
            
            self.status_label.configure(text=f"Scraping complete! Captured {len(self.captured_images)} images")
            
        except Exception as e:
            logger.error("Scraping failed", error=str(e))
            self.status_label.configure(text=f"Scraping failed: {str(e)}")
        finally:
            self.is_capturing = False
    
    def _update_preview(self, image):
        """Update the preview with the latest captured image"""
        try:
            # Resize image for preview
            preview_size = (200, 200)
            image.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for display
            photo = ImageTk.PhotoImage(image)
            
            # Update the preview label
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo  # Keep a reference
            
        except Exception as e:
            logger.error("Failed to update preview", error=str(e))
    
    async def _click_next_button(self):
        """Click the next button to scroll to next character"""
        try:
            if not self.config.next_button_region:
                return
            
            x1, y1, x2, y2 = self.config.next_button_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Use pyautogui to click
            pyautogui.click(center_x, center_y)
            
            # Small delay to let the UI update
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logger.error("Failed to click next button", error=str(e))
    
    def run(self):
        """Run the application"""
        self.app.mainloop()

# ================================
# Advanced Sorting Engine
# ================================

class AdvancedSorter:
    """AI-powered sorting with multiple algorithms"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.feature_index = None
        self.characters = []
    
    def build_feature_index(self, embeddings: List[np.ndarray]):
        """Build FAISS index for fast similarity search"""
        if not embeddings:
            return
        
        dimension = embeddings[0].shape[0]
        self.feature_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        self.feature_index.add(embeddings_array)
    
    def sort_by_visual_similarity(self, characters: List[Character]) -> List[Character]:
        """Sort characters by visual similarity using AI embeddings"""
        if not characters or not self.feature_index:
            return characters
        
        # Start with first character
        sorted_chars = [characters[0]]
        remaining = set(range(1, len(characters)))
        
        current_idx = 0
        while remaining:
            # Find most similar character to current
            query_embedding = np.array([characters[current_idx].embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            _, indices = self.feature_index.search(query_embedding, len(characters))
            
            # Find next character from remaining set
            for idx in indices[0]:
                if idx in remaining:
                    sorted_chars.append(characters[idx])
                    remaining.remove(idx)
                    current_idx = idx
                    break
        
        return sorted_chars
    
    def sort_by_neural_clustering(self, characters: List[Character]) -> List[Character]:
        """Sort using neural clustering algorithms"""
        if not characters:
            return characters
        
        # Extract embeddings
        embeddings = np.array([char.embedding for char in characters if char.embedding])
        
        if len(embeddings) == 0:
            return characters
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group by clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(characters[i])
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Flatten clusters into final sorted list
        sorted_chars = []
        for _, cluster_chars in sorted_clusters:
            # Sort within cluster by similarity
            if len(cluster_chars) > 1:
                cluster_chars = self.sort_by_visual_similarity(cluster_chars)
            sorted_chars.extend(cluster_chars)
        
        return sorted_chars

# ================================
# File Watcher for Auto-Updates
# ================================

class FileWatcher(FileSystemEventHandler):
    """Watch for new images and auto-process"""
    
    def __init__(self, callback):
        self.callback = callback
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            logger.info("New image detected", path=event.src_path)
            asyncio.create_task(self.callback(event.src_path))

# ================================
# Main Application Entry Point
# ================================

async def main():
    """Main application entry point"""
    try:
        # Initialize logging
        console.print(Panel.fit(f"[bold blue]{APP_NAME} v{APP_VERSION}[/bold blue]\n[green]Starting application...[/green]"))
        
        # Check dependencies
        if not torch.cuda.is_available():
            console.print("[yellow]Warning: CUDA not available. Using CPU for AI processing.[/yellow]")
        
        # Create UI and run
        ui = ModernUI()
        
        # Setup file watcher
        observer = Observer()
        observer.schedule(
            FileWatcher(lambda path: logger.info("New file", path=path)),
            str(ui.config.output_folder),
            recursive=True
        )
        observer.start()
        
        try:
            ui.run()
        finally:
            observer.stop()
            observer.join()
            
    except Exception as e:
        logger.error("Application failed to start", error=str(e))
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run application
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)
