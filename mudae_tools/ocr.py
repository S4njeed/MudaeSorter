"""RapidOCR (ONNX Runtime) text extraction, Mudae text-list parsing, and fuzzy typo correction."""

from __future__ import annotations

import difflib
import re
import threading
from typing import Iterable, List, Literal, Tuple

import cv2
import numpy as np

from .config import logger

_KAKERA_RE = re.compile(r"(?:[·•|]\s*)?\(?\b\d+\s*(?:ka|kakera)\b\)?", re.IGNORECASE)
_RANK_RE = re.compile(r"(?:^|\s)#\d+\b")
_LEADING_INDEX_RE = re.compile(r"^\s*(?:#\d+|\d+[.)\]])\s*")
_SM_HEADER_RE = re.compile(r"^\s*\$(?:sortmarrypos|sortmarry|smp|sm)\b\s*", re.IGNORECASE)
_SKIP_LINE_RE = re.compile(
    r"^\s*(?:page\s+\d+|total\s+value|harem\s+of|claims?:|kakera:|\d+\s*/\s*\d+)\b",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s+")
_BAD_CHARS_RE = re.compile(r"[^\w\s'\-\.\:\(\)\/&!?,+]")


def clean_mudae_name(raw_text: str) -> str:
    """Strip common Discord/Mudae embed noise while preserving character name punctuation."""
    text = raw_text.replace("\n", " ").strip()
    text = _KAKERA_RE.sub("", text)
    text = _RANK_RE.sub("", text)
    text = _BAD_CHARS_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text).strip(" -|:,·•")
    return text


def parse_mudae_text_list(raw_text: str) -> List[Tuple[str, str]]:
    """Parse pasted Mudae $mm / $mms / $mmk / $mma / $sm text into (name, series) pairs."""
    stripped = raw_text.strip()
    if not stripped:
        return []

    results: List[Tuple[str, str]] = []
    seen: set[str] = set()

    # Handle $sm / $smp commands directly
    if _SM_HEADER_RE.match(stripped) or ("$" in stripped and "\n" not in stripped):
        body = _SM_HEADER_RE.sub("", stripped)
        for token in body.split("$"):
            name = clean_mudae_name(token)
            if name and name.lower() not in seen:
                seen.add(name.lower())
                results.append((name, ""))
        return results

    for line in stripped.splitlines():
        line = line.strip()
        if not line or _SKIP_LINE_RE.match(line):
            continue

        # Remove leading list index ("1. ", "12) ", "#4 ") and trailing kakera
        line = _LEADING_INDEX_RE.sub("", line)
        line = _KAKERA_RE.sub("", line).strip()
        if not line:
            continue

        name_part = line
        series_part = ""
        for sep in (" — ", " – ", " | ", " - "):
            if sep in line:
                left, right = line.split(sep, 1)
                name_part = left.strip()
                series_part = right.strip()
                break

        name = clean_mudae_name(name_part)
        series = clean_mudae_name(series_part) if series_part else ""
        if name and name.lower() not in seen:
            seen.add(name.lower())
            results.append((name, series))

    return results


class FuzzyLexiconCorrector:
    """Snaps noisy OCR readings to known Mudae character and series names."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._names_lower: dict[str, str] = {}
        self._series_lower: dict[str, str] = {}

    def add_entries(self, names: Iterable[str] = (), series: Iterable[str] = ()) -> None:
        with self._lock:
            for item in names:
                clean = item.strip()
                if clean and clean != "Unknown" and not clean.startswith("Character_"):
                    self._names_lower[clean.lower()] = clean
            for item in series:
                clean = item.strip()
                if clean and clean != "Unknown Series":
                    self._series_lower[clean.lower()] = clean

    def clear(self) -> None:
        with self._lock:
            self._names_lower.clear()
            self._series_lower.clear()

    @staticmethod
    def _best_match(raw: str, lexicon: dict[str, str], threshold: float) -> str:
        candidate = raw.strip()
        if not candidate or not lexicon:
            return candidate
        low = candidate.lower()
        if low in lexicon:
            return lexicon[low]

        best_key = ""
        best_ratio = 0.0
        for key in lexicon:
            if abs(len(key) - len(low)) > max(4, int(len(key) * 0.4)):
                continue
            ratio = difflib.SequenceMatcher(None, low, key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key

        if best_key and best_ratio >= threshold:
            return lexicon[best_key]
        return candidate

    def correct_name(self, text: str, threshold: float = 0.80) -> str:
        with self._lock:
            return self._best_match(text, self._names_lower, threshold)

    def correct_series(self, text: str, threshold: float = 0.80) -> str:
        with self._lock:
            return self._best_match(text, self._series_lower, threshold)


class OCRService:
    """Thread-safe RapidOCR engine with preprocessing and optional fuzzy lexicon correction."""

    def __init__(self, min_confidence: float = 0.45) -> None:
        self.min_confidence = min_confidence
        self.corrector = FuzzyLexiconCorrector()
        self._engine = None
        self._lock = threading.Lock()

    def _ensure_engine(self) -> bool:
        if self._engine is not None:
            return True
        with self._lock:
            if self._engine is not None:
                return True
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
                logger.info("RapidOCR ONNX engine initialized")
                return True
            except Exception as exc:
                logger.error("Failed to initialize RapidOCR", error=str(exc))
                return False

    def is_available(self) -> bool:
        """Check whether RapidOCR can be loaded."""
        return self._ensure_engine()

    def extract_text(
        self,
        rgb_image: np.ndarray,
        field_kind: Literal["raw", "name", "series"] = "raw",
    ) -> Tuple[str, float]:
        """Extract cleaned text and average confidence from an RGB numpy crop."""
        if rgb_image is None or rgb_image.size == 0:
            return "", 0.0
        if not self._ensure_engine():
            return "", 0.0

        try:
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if h < 80:
                scale = max(2.0, 96.0 / max(1, h))
                bgr = cv2.resize(
                    bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            with self._lock:
                result, _ = self._engine(bgr)

            if not result:
                return "", 0.0

            texts = []
            confs = []
            for item in result:
                if len(item) >= 3:
                    text, score = str(item[1]), float(item[2])
                    if score >= self.min_confidence and text.strip():
                        texts.append(text.strip())
                        confs.append(score)

            if not texts:
                return "", 0.0

            joined = " ".join(texts)
            cleaned = clean_mudae_name(joined) or joined.strip()
            if field_kind == "name":
                cleaned = self.corrector.correct_name(cleaned)
            elif field_kind == "series":
                cleaned = self.corrector.correct_series(cleaned)

            avg_conf = float(sum(confs) / len(confs))
            return cleaned, avg_conf
        except Exception as exc:
            logger.error("OCR extraction failed", error=str(exc))
            return "", 0.0
