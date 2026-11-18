"""Background Qt worker threads for scraping, folder import, AI indexing, and sorting."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QThread, Signal

from ..capture import ScreenCapture
from ..config import AppConfig, Character, ImageFormat, SortMethod, logger
from ..database import DatabaseManager
from ..exporter import sanitize_filename
from ..ocr import OCRService
from ..sorter import AdvancedSorter
from ..vision import CLIPEngine, ColorAnalyzer, ImageHasher

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_RANK_PREFIX_RE = re.compile(r"^\d{3,5}_")
_TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{10,14}(?:_\d+)?$")


def infer_name_and_series(path: Path, root_folder: Path) -> Tuple[str, str]:
    """Infer character name and series from filename and parent directory."""
    stem = path.stem
    stem = _RANK_PREFIX_RE.sub("", stem)
    stem = _TIMESTAMP_SUFFIX_RE.sub("", stem)

    series = "Unknown Series"
    if path.parent != root_folder:
        series = path.parent.name.replace("_", " ").strip() or series

    if " - " in stem:
        left, right = stem.split(" - ", 1)
        if series == "Unknown Series":
            series = left.replace("_", " ").strip() or series
        name = right.replace("_", " ").strip()
    else:
        name = stem.replace("_", " ").strip()

    return name or path.stem, series


def save_pil_image(img: Image.Image, dest_path: Path, image_format: ImageFormat, jpeg_quality: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    save_img = img.convert("RGB") if image_format == ImageFormat.JPEG else img
    kwargs = {"quality": jpeg_quality} if image_format in (ImageFormat.JPEG, ImageFormat.WEBP) else {}
    save_img.save(dest_path, format=image_format.value, **kwargs)


class ScraperWorker(QThread):
    """Thread-safe Discord region scraper with duplicate guard, RapidOCR, OKLab, and CLIP."""

    progress_updated = Signal(int, int, str)
    card_captured = Signal(object)
    log_message = Signal(str)
    finished_summary = Signal(int, int, str)

    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        ocr: OCRService,
        clip_engine: CLIPEngine,
    ) -> None:
        super().__init__()
        self.config = config
        self.db = db
        self.ocr = ocr
        self.clip_engine = clip_engine
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        self.requestInterruption()

    def _sleep_interruptible(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self._stop_event.is_set() or self.isInterruptionRequested():
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return not (self._stop_event.is_set() or self.isInterruptionRequested())

    def run(self) -> None:
        captured = 0
        skipped_dups = 0
        total = self.config.capture_count
        prev_hash = ""

        if not self.config.capture_region:
            self.finished_summary.emit(0, 0, "No Card Capture Region configured.")
            return

        output_dir = Path(self.config.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if self.config.image_format == ImageFormat.JPEG else self.config.image_format.value.lower()

        self.log_message.emit(f"Starting live scraper: {total} pages, delay {self.config.capture_delay:.2f}s")

        try:
            for step in range(total):
                if self._stop_event.is_set() or self.isInterruptionRequested():
                    break

                self.progress_updated.emit(step, total, f"Capturing card {step + 1}/{total}...")
                rgb = ScreenCapture.capture_region(self.config.capture_region)
                if rgb is None:
                    self.log_message.emit(f"[Step {step + 1}] Screen capture returned empty frame.")
                    if not self._sleep_interruptible(self.config.capture_delay):
                        break
                    continue

                pil_img = Image.fromarray(rgb, mode="RGB")
                phash = ImageHasher.compute_hash(pil_img)

                if prev_hash:
                    dist = ImageHasher.normalized_distance(prev_hash, phash)
                    if dist <= self.config.duplicate_distance_threshold:
                        skipped_dups += 1
                        self.log_message.emit(
                            f"[Step {step + 1}] Duplicate frame detected (dist={dist:.3f}); advancing."
                        )
                        if self.config.auto_scroll and self.config.next_button_region:
                            ScreenCapture.click_region_center(
                                self.config.next_button_region, restore_cursor=self.config.restore_cursor
                            )
                        if not self._sleep_interruptible(self.config.capture_delay):
                            break
                        continue

                prev_hash = phash

                char_name = f"Character_{step + 1:04d}"
                series_name = "Unknown Series"
                best_conf = 0.0

                if self.config.character_name_region:
                    name_rgb = ScreenCapture.capture_region(self.config.character_name_region)
                    if name_rgb is not None:
                        ocr_name, conf = self.ocr.extract_text(name_rgb, field_kind="name")
                        if ocr_name:
                            char_name = ocr_name
                            best_conf = conf

                if self.config.show_name_region:
                    show_rgb = ScreenCapture.capture_region(self.config.show_name_region)
                    if show_rgb is not None:
                        ocr_show, _ = self.ocr.extract_text(show_rgb, field_kind="series")
                        if ocr_show:
                            series_name = ocr_show

                hex_palette, oklab_vec = ColorAnalyzer.extract_palette(pil_img)
                clip_vec = self.clip_engine.encode_images(
                    [pil_img], batch_size=1, enable_neural=self.config.enable_clip
                )[0].tolist()

                ts = int(time.time() * 1000)
                safe_name = sanitize_filename(char_name, f"character_{step + 1}")
                file_path = output_dir / f"{safe_name}_{ts}.{ext}"
                save_pil_image(pil_img, file_path, self.config.image_format, self.config.jpeg_quality)

                character = Character(
                    name=char_name,
                    series=series_name,
                    image_path=file_path,
                    phash=phash,
                    dominant_hex=hex_palette,
                    oklab_feature=oklab_vec,
                    clip_embedding=clip_vec,
                    ocr_confidence=best_conf,
                )
                self.db.upsert_character(character)
                captured += 1
                self.card_captured.emit(character)
                self.log_message.emit(f"Captured #{captured}: {char_name} ({series_name})")
                self.progress_updated.emit(step + 1, total, f"Captured {captured} cards")

                if self.config.auto_scroll and self.config.next_button_region and step < total - 1:
                    ScreenCapture.click_region_center(
                        self.config.next_button_region, restore_cursor=self.config.restore_cursor
                    )

                if step < total - 1 and not self._sleep_interruptible(self.config.capture_delay):
                    break

            msg = f"Completed: {captured} captured, {skipped_dups} duplicate frames skipped."
            self.finished_summary.emit(captured, skipped_dups, msg)
        except Exception as exc:
            logger.error("ScraperWorker crashed", error=str(exc))
            self.finished_summary.emit(captured, skipped_dups, f"Scraper error: {exc}")


class FolderImportWorker(QThread):
    """Import an existing directory of character cards and index hashes, OKLab, and CLIP."""

    progress_updated = Signal(int, int, str)
    log_message = Signal(str)
    finished_summary = Signal(int, str)

    def __init__(
        self,
        folder: Path,
        config: AppConfig,
        db: DatabaseManager,
        clip_engine: CLIPEngine,
    ) -> None:
        super().__init__()
        self.folder = Path(folder)
        self.config = config
        self.db = db
        self.clip_engine = clip_engine

    def run(self) -> None:
        files = sorted(
            p for p in self.folder.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        total = len(files)
        if total == 0:
            self.finished_summary.emit(0, "No PNG/JPG/WEBP images found in selected folder.")
            return

        self.log_message.emit(f"Found {total} image files in {self.folder}")
        imported = 0
        batch_size = 16

        for start in range(0, total, batch_size):
            if self.isInterruptionRequested():
                break
            chunk = files[start : start + batch_size]
            pil_images: List[Image.Image] = []
            valid_paths: List[Path] = []

            for path in chunk:
                try:
                    img = Image.open(path).convert("RGB")
                    pil_images.append(img)
                    valid_paths.append(path)
                except Exception as exc:
                    self.log_message.emit(f"Skipped unreadable image {path.name}: {exc}")

            if not pil_images:
                continue

            embeddings = self.clip_engine.encode_images(
                pil_images, batch_size=batch_size, enable_neural=self.config.enable_clip
            )

            for img, path, emb in zip(pil_images, valid_paths, embeddings):
                name, series = infer_name_and_series(path, self.folder)
                phash = ImageHasher.compute_hash(img)
                hex_palette, oklab_vec = ColorAnalyzer.extract_palette(img)
                char = Character(
                    name=name,
                    series=series,
                    image_path=path,
                    phash=phash,
                    dominant_hex=hex_palette,
                    oklab_feature=oklab_vec,
                    clip_embedding=emb.tolist(),
                )
                self.db.upsert_character(char)
                imported += 1
                img.close()

            done = min(total, start + len(chunk))
            self.progress_updated.emit(done, total, f"Indexed {done}/{total} images...")

        self.finished_summary.emit(imported, f"Imported and indexed {imported} character cards.")


class DemoGeneratorWorker(QThread):
    """Create 12 rich synthetic character cards across color families for instant testing."""

    progress_updated = Signal(int, int, str)
    finished_summary = Signal(int, str)

    _SAMPLES = [
        ("Zero Two", "Darling in the FranXX", (245, 130, 174), (255, 224, 236), (60, 22, 46)),
        ("Rem", "Re:Zero", (92, 168, 255), (218, 238, 255), (24, 34, 68)),
        ("Megumin", "KonoSuba", (215, 58, 52), (255, 194, 108), (38, 18, 20)),
        ("Saber", "Fate/stay night", (236, 198, 92), (72, 118, 212), (245, 240, 220)),
        ("Hatsune Miku", "Vocaloid", (58, 212, 204), (190, 250, 246), (26, 40, 48)),
        ("Makima", "Chainsaw Man", (222, 92, 74), (244, 226, 206), (28, 24, 28)),
        ("Raiden Shogun", "Genshin Impact", (142, 96, 232), (212, 188, 255), (30, 20, 56)),
        ("Frieren", "Sousou no Frieren", (236, 240, 248), (154, 198, 188), (48, 58, 72)),
        ("C.C.", "Code Geass", (118, 214, 96), (222, 246, 198), (28, 42, 26)),
        ("2B", "NieR:Automata", (42, 44, 52), (220, 224, 232), (18, 18, 22)),
        ("Marin Kitagawa", "My Dress-Up Darling", (252, 214, 116), (255, 152, 182), (62, 42, 54)),
        ("Alucard", "Hellsing", (176, 28, 38), (42, 18, 22), (235, 216, 198)),
    ]

    def __init__(self, config: AppConfig, db: DatabaseManager, clip_engine: CLIPEngine) -> None:
        super().__init__()
        self.config = config
        self.db = db
        self.clip_engine = clip_engine

    @staticmethod
    def _render_card(name: str, c1: Tuple[int, int, int], c2: Tuple[int, int, int], c3: Tuple[int, int, int]) -> Image.Image:
        w, h = 225, 350
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        bg_top = np.array(c3, dtype=np.float32)
        bg_bot = np.array(c1, dtype=np.float32)
        grad = (1.0 - t) * bg_top + t * bg_bot
        arr[:, :, :] = np.clip(grad, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

        draw = ImageDraw.Draw(img)
        # Halo / aura ring
        draw.ellipse((32, 46, 193, 207), fill=c2, outline=c1, width=4)
        # Stylized character silhouette
        draw.ellipse((68, 76, 157, 165), fill=c1, outline=(255, 255, 255), width=2)
        draw.polygon([(46, 330), (88, 178), (137, 178), (179, 330)], fill=c3, outline=c2)
        # Accent hair/ribbon strokes
        draw.polygon([(62, 110), (42, 235), (82, 160)], fill=c1)
        draw.polygon([(163, 110), (183, 235), (143, 160)], fill=c1)
        # Footer nameplate bar
        draw.rectangle((10, 304, w - 10, 340), fill=(18, 20, 28), outline=c2, width=2)
        draw.text((20, 316), name[:22], fill=(245, 247, 252))
        return img

    def run(self) -> None:
        demo_dir = Path(self.config.output_folder) / "DemoCards"
        demo_dir.mkdir(parents=True, exist_ok=True)
        total = len(self._SAMPLES)
        images: List[Image.Image] = []
        paths: List[Path] = []

        for idx, (name, series, c1, c2, c3) in enumerate(self._SAMPLES):
            img = self._render_card(name, c1, c2, c3)
            out_path = demo_dir / f"{sanitize_filename(series)} - {sanitize_filename(name)}.png"
            img.save(out_path, format="PNG")
            images.append(img)
            paths.append(out_path)
            self.progress_updated.emit(idx + 1, total * 2, f"Rendered sample card {name}")

        embeddings = self.clip_engine.encode_images(
            images, batch_size=12, enable_neural=self.config.enable_clip
        )
        for idx, ((name, series, _, _, _), img, path, emb) in enumerate(
            zip(self._SAMPLES, images, paths, embeddings)
        ):
            phash = ImageHasher.compute_hash(img)
            hex_palette, oklab_vec = ColorAnalyzer.extract_palette(img)
            char = Character(
                name=name,
                series=series,
                image_path=path,
                phash=phash,
                dominant_hex=hex_palette,
                oklab_feature=oklab_vec,
                clip_embedding=emb.tolist(),
                tags=["demo"],
            )
            self.db.upsert_character(char)
            img.close()
            self.progress_updated.emit(total + idx + 1, total * 2, f"Indexed {name}")

        self.finished_summary.emit(total, f"Generated and indexed {total} demo character cards.")


class SortWorker(QThread):
    """Run AI/TSP sorting and persist updated order without freezing the UI."""

    finished_sorted = Signal(list, str)
    failed = Signal(str)

    def __init__(
        self,
        characters: List[Character],
        method: SortMethod,
        db: DatabaseManager,
        clip_engine: CLIPEngine,
        min_cluster_size: int = 2,
        prompt: str = "",
        enable_neural: bool = True,
    ) -> None:
        super().__init__()
        self.characters = list(characters)
        self.method = method
        self.db = db
        self.clip_engine = clip_engine
        self.min_cluster_size = min_cluster_size
        self.prompt = prompt
        self.enable_neural = enable_neural

    def run(self) -> None:
        try:
            sorted_chars = AdvancedSorter.sort(
                self.characters,
                method=self.method,
                clip_engine=self.clip_engine,
                min_cluster_size=self.min_cluster_size,
                prompt=self.prompt,
                enable_neural=self.enable_neural,
            )
            self.db.update_sort_orders(sorted_chars)
            label = self.method.value.replace("_", " ").title()
            if self.method == SortMethod.NATURAL_LANGUAGE and self.prompt:
                label = f"CLIP Prompt: '{self.prompt}'"
            self.finished_sorted.emit(sorted_chars, f"Sorted {len(sorted_chars)} characters via {label}.")
        except Exception as exc:
            logger.error("SortWorker failed", error=str(exc))
            self.failed.emit(f"Sorting failed: {exc}")
