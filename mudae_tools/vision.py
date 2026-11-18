"""Perceptual hashing, OKLab color palette analysis, and CLIP semantic embeddings."""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from .config import logger


class ImageHasher:
    """Multi-algorithm perceptual hashing (dHash + pHash + wHash + ColorHash)."""

    @staticmethod
    def compute_hash(image: Image.Image, hash_size: int = 8) -> str:
        rgb = image.convert("RGB")
        dh = str(imagehash.dhash(rgb, hash_size=hash_size))
        ph = str(imagehash.phash(rgb, hash_size=hash_size))
        wh = str(imagehash.whash(rgb, hash_size=hash_size))
        ch = str(imagehash.colorhash(rgb, binbits=3))
        return f"{dh}_{ph}_{wh}_{ch}"

    @staticmethod
    def normalized_distance(hash1: str, hash2: str) -> float:
        """Return normalized Hamming distance in [0.0, 1.0]; 0.0 means identical."""
        if not hash1 or not hash2:
            return 1.0
        parts1 = hash1.split("_")
        parts2 = hash2.split("_")
        if len(parts1) not in (3, 4) or len(parts1) != len(parts2):
            return 1.0

        dists: List[float] = []
        for idx, (p1, p2) in enumerate(zip(parts1, parts2)):
            try:
                if idx == 3:
                    # colorhash encodes 42 bits (14 HSV bins * 3 bits)
                    b1 = int(p1, 16)
                    b2 = int(p2, 16)
                    diff_bits = (b1 ^ b2).bit_count()
                    max_bits = max(1, len(p1) * 4)
                    dists.append(diff_bits / float(max_bits))
                else:
                    h1 = imagehash.hex_to_hash(p1)
                    h2 = imagehash.hex_to_hash(p2)
                    max_bits = max(1, h1.hash.size)
                    dists.append((h1 - h2) / float(max_bits))
            except Exception:
                continue
        return float(sum(dists) / len(dists)) if dists else 1.0


class ColorAnalyzer:
    """Exact sRGB -> OKLab conversion and center-weighted 5-color palette extraction."""

    @staticmethod
    def srgb_to_oklab(rgb_uint8: np.ndarray) -> np.ndarray:
        """Vectorized conversion from uint8 RGB (..., 3) to perceptual OKLab (..., 3)."""
        srgb = np.asarray(rgb_uint8, dtype=np.float32) / 255.0
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power(np.clip((srgb + 0.055) / 1.055, 0.0, 1.0), 2.4),
        )
        r = linear[..., 0]
        g = linear[..., 1]
        b = linear[..., 2]

        l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_cbrt = np.cbrt(np.maximum(l_, 0.0))
        m_cbrt = np.cbrt(np.maximum(m_, 0.0))
        s_cbrt = np.cbrt(np.maximum(s_, 0.0))

        L = 0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt
        a = 1.9779984951 * l_cbrt - 2.4285922050 * m_cbrt + 0.4505937099 * s_cbrt
        b_ok = 0.0259040371 * l_cbrt + 0.7827717662 * m_cbrt - 0.8086757660 * s_cbrt
        return np.stack([L, a, b_ok], axis=-1).astype(np.float32)

    @staticmethod
    def oklab_to_srgb_hex(oklab: np.ndarray) -> str:
        """Convert single OKLab [L, a, b] vector to #RRGGBB hex string."""
        L, a, b = float(oklab[0]), float(oklab[1]), float(oklab[2])
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b

        l3 = l_ * l_ * l_
        m3 = m_ * m_ * m_
        s3 = s_ * s_ * s_

        r_lin = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
        g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
        b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

        rgb_lin = np.clip(np.array([r_lin, g_lin, b_lin], dtype=np.float32), 0.0, 1.0)
        srgb = np.where(
            rgb_lin <= 0.0031308,
            12.92 * rgb_lin,
            1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
        )
        rgb_u8 = np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
        return f"#{rgb_u8[0]:02X}{rgb_u8[1]:02X}{rgb_u8[2]:02X}"

    @classmethod
    def extract_palette(cls, image: Image.Image, n_colors: int = 5) -> Tuple[List[str], List[float]]:
        """Extract dominant hex palette and an 18-float OKLab descriptor vector."""
        thumb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(thumb, dtype=np.uint8)
        oklab_grid = cls.srgb_to_oklab(arr)

        # Center-weight portrait region (inner 70% gets 2x sampling weight)
        h, w = oklab_grid.shape[:2]
        y0, y1 = int(h * 0.15), int(h * 0.85)
        x0, x1 = int(w * 0.15), int(w * 0.85)
        all_pixels = oklab_grid.reshape(-1, 3)
        center_pixels = oklab_grid[y0:y1, x0:x1].reshape(-1, 3)
        weighted_pixels = np.vstack([all_pixels, center_pixels])

        mean_oklab = weighted_pixels.mean(axis=0)

        unique_count = len(np.unique(np.round(weighted_pixels, 3), axis=0))
        k = min(n_colors, max(1, unique_count))
        kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42, batch_size=512)
        labels = kmeans.fit_predict(weighted_pixels)
        counts = np.bincount(labels, minlength=k)
        order = np.argsort(-counts)

        centers = kmeans.cluster_centers_[order]
        if len(centers) < n_colors:
            pad = np.repeat(centers[-1:], n_colors - len(centers), axis=0)
            centers = np.vstack([centers, pad])

        hex_colors = [cls.oklab_to_srgb_hex(c) for c in centers[:n_colors]]
        feature_vec = np.concatenate([mean_oklab, centers[:n_colors].reshape(-1)]).astype(np.float32)
        return hex_colors, feature_vec.tolist()


class CLIPEngine:
    """Lazy-loaded CLIP / SigLIP vision-language encoder with deterministic fallback."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", use_gpu: bool = True) -> None:
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._load_failed = False

    def set_model_name(self, model_name: str) -> None:
        """Switch vision-language checkpoint and reset cached weights."""
        clean = model_name.strip() or "openai/clip-vit-base-patch32"
        if clean != self.model_name:
            with self._lock:
                self.model_name = clean
                self._model = None
                self._processor = None
                self._load_failed = False

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_neural_active(self) -> bool:
        return self._model is not None and not self._load_failed

    def ensure_loaded(self) -> bool:
        """Load AutoModel/AutoProcessor (SigLIP or CLIP) from local/HF cache."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            try:
                import torch
                from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

                self._device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
                try:
                    self._processor = AutoProcessor.from_pretrained(self.model_name)
                    self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
                except Exception:
                    self._processor = CLIPProcessor.from_pretrained(self.model_name)
                    self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
                self._model.eval()
                logger.info("Vision-language model loaded", model=self.model_name, device=self._device)
                return True
            except Exception as exc:
                self._load_failed = True
                logger.warning(
                    "Vision-language model unavailable; using fast deterministic fallback descriptor",
                    model=self.model_name,
                    error=str(exc),
                )
                return False

    @staticmethod
    def _extract_tensor(output, attr_hint: str):
        import torch

        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, attr_hint) and getattr(output, attr_hint) is not None:
            return getattr(output, attr_hint)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if isinstance(output, (tuple, list)) and len(output) > 0:
            return output[0]
        raise TypeError(f"Unsupported feature output type: {type(output)}")

    @staticmethod
    def fallback_image_descriptor(image: Image.Image, dim: int = 512) -> np.ndarray:
        """Compute a fast 512-dim L2-normalized OKLab + gradient spatial pyramid descriptor."""
        small = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.uint8)
        oklab = ColorAnalyzer.srgb_to_oklab(arr)

        feats: List[np.ndarray] = []
        # 4x4 spatial grid mean + std of OKLab (4*4*6 = 96 dims)
        for gy in range(4):
            for gx in range(4):
                patch = oklab[gy * 16 : (gy + 1) * 16, gx * 16 : (gx + 1) * 16]
                feats.append(patch.mean(axis=(0, 1)))
                feats.append(patch.std(axis=(0, 1)))

        # Color histogram in OKLab (8 bins per channel -> 8*8*4? Keep compact: 6x6x6 = 216 bins)
        L_bin = np.clip((oklab[..., 0] * 6).astype(int), 0, 5)
        a_bin = np.clip(((oklab[..., 1] + 0.35) / 0.70 * 6).astype(int), 0, 5)
        b_bin = np.clip(((oklab[..., 2] + 0.35) / 0.70 * 6).astype(int), 0, 5)
        idx = (L_bin * 36 + a_bin * 6 + b_bin).ravel()
        hist = np.bincount(idx, minlength=216).astype(np.float32)
        hist /= max(1.0, float(hist.sum()))
        feats.append(hist)

        # Edge orientation & magnitude summary across 4x4 grid (4*4*4 = 64 dims)
        gray = oklab[..., 0]
        dx = np.diff(gray, axis=1, prepend=gray[:, :1])
        dy = np.diff(gray, axis=0, prepend=gray[:1, :])
        mag = np.sqrt(dx * dx + dy * dy)
        for gy in range(4):
            for gx in range(4):
                mg = mag[gy * 16 : (gy + 1) * 16, gx * 16 : (gx + 1) * 16]
                gx_p = np.abs(dx[gy * 16 : (gy + 1) * 16, gx * 16 : (gx + 1) * 16])
                gy_p = np.abs(dy[gy * 16 : (gy + 1) * 16, gx * 16 : (gx + 1) * 16])
                feats.append(np.array([mg.mean(), mg.std(), gx_p.mean(), gy_p.mean()], dtype=np.float32))

        vec = np.concatenate(feats).astype(np.float32)
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        else:
            vec = vec[:dim]
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-8)

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 16,
        enable_neural: bool = True,
    ) -> np.ndarray:
        """Encode a list of PIL images into L2-normalized (N, 512) float32 embeddings."""
        if not images:
            return np.zeros((0, 512), dtype=np.float32)

        if enable_neural and self.ensure_loaded():
            import torch

            all_vecs: List[np.ndarray] = []
            for start in range(0, len(images), batch_size):
                batch = [img.convert("RGB") for img in images[start : start + batch_size]]
                inputs = self._processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self._device)
                with torch.no_grad():
                    out = self._model.get_image_features(pixel_values=pixel_values)
                    emb = self._extract_tensor(out, "image_embeds")
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    all_vecs.append(emb.detach().cpu().numpy().astype(np.float32))
            return np.vstack(all_vecs)

        return np.vstack([self.fallback_image_descriptor(img) for img in images]).astype(np.float32)

    def encode_text(self, query: str, enable_neural: bool = True) -> Optional[np.ndarray]:
        """Encode natural-language query into L2-normalized (1, D) float32 vector."""
        if not query.strip():
            return None
        if enable_neural and self.ensure_loaded():
            import torch

            pad_mode = "max_length" if "siglip" in self.model_name.lower() else True
            inputs = self._processor(text=[query.strip()], return_tensors="pt", padding=pad_mode, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.get_text_features(**inputs)
                emb = self._extract_tensor(out, "text_embeds")
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                return emb.detach().cpu().numpy().astype(np.float32)
        return None
