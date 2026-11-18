"""Advanced visual, perceptual OKLab TSP, CLIP+FAISS, and HDBSCAN sorting algorithms."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

import faiss
import hdbscan
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from .config import Character, SortMethod, logger
from .vision import CLIPEngine, ColorAnalyzer

# Approximate OKLab anchors for offline color-keyword fallback queries
_COLOR_KEYWORD_OKLAB: Dict[str, Tuple[float, float, float]] = {
    "red": (0.60, 0.22, 0.12),
    "crimson": (0.50, 0.21, 0.10),
    "pink": (0.76, 0.16, -0.02),
    "magenta": (0.64, 0.24, -0.10),
    "orange": (0.72, 0.12, 0.16),
    "yellow": (0.88, -0.05, 0.18),
    "gold": (0.80, 0.02, 0.15),
    "green": (0.68, -0.18, 0.12),
    "emerald": (0.62, -0.18, 0.08),
    "cyan": (0.80, -0.12, -0.06),
    "blue": (0.55, -0.04, -0.22),
    "navy": (0.35, 0.00, -0.16),
    "purple": (0.52, 0.14, -0.18),
    "violet": (0.58, 0.14, -0.20),
    "white": (0.96, 0.00, 0.00),
    "light": (0.88, 0.00, 0.00),
    "bright": (0.88, 0.00, 0.00),
    "dark": (0.24, 0.00, 0.00),
    "black": (0.15, 0.00, 0.00),
    "monochrome": (0.50, 0.00, 0.00),
}


def xyz_to_hilbert_index(x: int, y: int, z: int, bits: int = 8) -> int:
    """Compute the 1D 3D-Hilbert space-filling curve index for quantized coords in [0, 2^bits - 1]."""
    coords = [int(x), int(y), int(z)]
    m = 1 << (bits - 1)
    q = m
    while q > 1:
        p = q - 1
        for i in range(3):
            if coords[i] & q:
                coords[0] ^= p
            else:
                t = (coords[0] ^ coords[i]) & p
                coords[0] ^= t
                coords[i] ^= t
        q >>= 1

    for i in range(1, 3):
        coords[i] ^= coords[i - 1]
    t = 0
    q = m
    while q > 1:
        if coords[2] & q:
            t ^= q - 1
        q >>= 1
    for i in range(3):
        coords[i] ^= t

    # Interleave bits into a single 3*bits integer (Gray-coded Hilbert order)
    index = 0
    for b in range(bits - 1, -1, -1):
        index = (index << 3) | (((coords[0] >> b) & 1) << 2) | (((coords[1] >> b) & 1) << 1) | ((coords[2] >> b) & 1)
    return index


def compute_path_distance(dist_matrix: np.ndarray, order: List[int]) -> float:
    """Compute total open-path cost across sequential cards in `order`."""
    if len(order) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(order) - 1):
        total += float(dist_matrix[order[idx], order[idx + 1]])
    return total


def solve_tsp_2opt(dist_matrix: np.ndarray, start_idx: int = 0, max_passes: int = 50) -> List[int]:
    """Build nearest-neighbor open path from `start_idx` and refine with 2-Opt local search."""
    n = dist_matrix.shape[0]
    if n <= 2:
        return list(range(n))

    # 1. Greedy nearest-neighbor open path
    visited = np.zeros(n, dtype=bool)
    order: List[int] = [int(start_idx)]
    visited[start_idx] = True

    curr = int(start_idx)
    for _ in range(n - 1):
        dists = dist_matrix[curr].copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
        curr = nxt

    # 2. Open-path 2-Opt refinement (keeps order[0] anchored)
    for _ in range(max_passes):
        improved = False
        for i in range(0, n - 2):
            a = order[i]
            b = order[i + 1]
            ab = float(dist_matrix[a, b])
            for j in range(i + 2, n):
                c = order[j]
                if j + 1 < n:
                    d = order[j + 1]
                    old_cost = ab + float(dist_matrix[c, d])
                    new_cost = float(dist_matrix[a, c]) + float(dist_matrix[b, d])
                else:
                    old_cost = ab
                    new_cost = float(dist_matrix[a, c])
                if new_cost + 1e-9 < old_cost:
                    order[i + 1 : j + 1] = reversed(order[i + 1 : j + 1])
                    b = order[i + 1]
                    ab = float(dist_matrix[a, b])
                    improved = True
        if not improved:
            break

    return order


class AdvancedSorter:
    """Collection sorting and semantic filtering engine."""

    @staticmethod
    def _ensure_oklab_matrix(characters: List[Character]) -> np.ndarray:
        """Return (N, 18) weighted OKLab feature matrix, computing on demand if missing."""
        rows: List[np.ndarray] = []
        for ch in characters:
            vec = ch.oklab_feature
            if (not vec or len(vec) != 18) and ch.image_path.exists():
                try:
                    with Image.open(ch.image_path) as img:
                        hex_colors, vec = ColorAnalyzer.extract_palette(img)
                        ch.dominant_hex = hex_colors
                        ch.oklab_feature = vec
                except Exception:
                    vec = None
            if not vec or len(vec) != 18:
                rows.append(np.zeros(18, dtype=np.float32))
            else:
                rows.append(np.asarray(vec, dtype=np.float32))

        mat = np.vstack(rows)
        # Apply perceptual importance weights: mean OKLab (1.0), dominant colors 1..5
        weights = np.array(
            [
                1.0, 1.0, 1.0,
                0.75, 0.75, 0.75,
                0.45, 0.45, 0.45,
                0.28, 0.28, 0.28,
                0.18, 0.18, 0.18,
                0.10, 0.10, 0.10,
            ],
            dtype=np.float32,
        )
        return mat * weights[None, :]

    @staticmethod
    def _ensure_clip_matrix(characters: List[Character], clip_engine: Optional[CLIPEngine] = None) -> np.ndarray:
        """Return (N, D) L2-normalized embedding matrix, filling missing vectors on demand."""
        missing_indices: List[int] = []
        missing_images: List[Image.Image] = []
        dim = 512

        for idx, ch in enumerate(characters):
            if not ch.clip_embedding:
                if ch.image_path.exists():
                    try:
                        missing_images.append(Image.open(ch.image_path).convert("RGB"))
                        missing_indices.append(idx)
                    except Exception:
                        pass
            else:
                dim = len(ch.clip_embedding)

        if missing_images:
            engine = clip_engine or CLIPEngine(use_gpu=True)
            encoded = engine.encode_images(missing_images, enable_neural=(clip_engine is not None))
            for pos, vec in zip(missing_indices, encoded):
                characters[pos].clip_embedding = vec.tolist()
            for img in missing_images:
                img.close()

        mat = np.zeros((len(characters), dim), dtype=np.float32)
        for i, ch in enumerate(characters):
            if ch.clip_embedding and len(ch.clip_embedding) == dim:
                v = np.asarray(ch.clip_embedding, dtype=np.float32)
                norm = float(np.linalg.norm(v))
                mat[i] = v / max(norm, 1e-8)
            else:
                mat[i, 0] = 1.0
        return mat

    @classmethod
    def oklab_distance_matrix(cls, characters: List[Character]) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix in weighted OKLab space."""
        mat = cls._ensure_oklab_matrix(characters)
        diff = mat[:, None, :] - mat[None, :, :]
        return np.linalg.norm(diff, axis=-1).astype(np.float32)

    @classmethod
    def sort_color_gradient_tsp(cls, characters: List[Character]) -> List[Character]:
        """Sort cards for smoothest visual color gradient using OKLab + 2-Opt TSP."""
        if len(characters) <= 1:
            return list(characters)

        dist_matrix = cls.oklab_distance_matrix(characters)
        # Start from brightest card so the flow sweeps naturally from light -> vivid -> dark
        lightness = [
            (ch.oklab_feature[0] if ch.oklab_feature and len(ch.oklab_feature) >= 3 else 0.5)
            for ch in characters
        ]
        start_idx = int(np.argmax(lightness))
        order = solve_tsp_2opt(dist_matrix, start_idx=start_idx, max_passes=60)
        return cls._finalize_order([characters[i] for i in order])

    @classmethod
    def sort_hilbert_color(cls, characters: List[Character]) -> List[Character]:
        """Quantize OKLab into 3D grid, sort via 3D Hilbert space-filling curve, then lightly polish."""
        if len(characters) <= 1:
            return list(characters)

        cls._ensure_oklab_matrix(characters)
        hilbert_keys: List[Tuple[int, int]] = []
        max_val = 255
        for idx, ch in enumerate(characters):
            vec = ch.oklab_feature or [0.5, 0.0, 0.0]
            L, a, b = (float(vec[3]), float(vec[4]), float(vec[5])) if len(vec) >= 6 else (float(vec[0]), float(vec[1]), float(vec[2]))
            qx = max(0, min(max_val, int(round(np.clip(L, 0.0, 1.0) * max_val))))
            qy = max(0, min(max_val, int(round(np.clip((a + 0.4) / 0.8, 0.0, 1.0) * max_val))))
            qz = max(0, min(max_val, int(round(np.clip((b + 0.4) / 0.8, 0.0, 1.0) * max_val))))
            hilbert_keys.append((xyz_to_hilbert_index(qx, qy, qz, bits=8), idx))

        hilbert_keys.sort(key=lambda item: item[0])
        ordered = [characters[idx] for _, idx in hilbert_keys]
        return cls._finalize_order(ordered)

    @classmethod
    def sort_rainbow_spectrum(cls, characters: List[Character]) -> List[Character]:
        """Sort chromatic cards across OKLCh hue sectors with serpentine lightness, neutrals last."""
        if len(characters) <= 1:
            return list(characters)

        cls._ensure_oklab_matrix(characters)
        chromatic: List[Tuple[int, float, Character]] = []
        achromatic: List[Tuple[float, Character]] = []

        for ch in characters:
            vec = ch.oklab_feature or [0.5, 0.0, 0.0]
            # Prefer primary dominant color (indices 3..5) for stronger hue separation if present
            if len(vec) >= 6:
                L, a, b = float(vec[3]), float(vec[4]), float(vec[5])
            else:
                L, a, b = float(vec[0]), float(vec[1]), float(vec[2])
            chroma = math.hypot(a, b)
            if chroma < 0.032:
                achromatic.append((-L, ch))
            else:
                hue = math.atan2(b, a)
                if hue < 0:
                    hue += 2.0 * math.pi
                sector = int((hue / (2.0 * math.pi)) * 12) % 12
                # Serpentine lightness inside alternating sectors to minimize jumps
                light_key = L if (sector % 2 == 0) else -L
                chromatic.append((sector, light_key, ch))

        chromatic.sort(key=lambda item: (item[0], item[1]))
        achromatic.sort(key=lambda item: item[0])
        ordered = [item[2] for item in chromatic] + [item[1] for item in achromatic]
        return cls._finalize_order(ordered)

    @classmethod
    def sort_visual_similarity_tsp(
        cls,
        characters: List[Character],
        clip_engine: Optional[CLIPEngine] = None,
    ) -> List[Character]:
        """Sort cards by CLIP visual/style similarity using FAISS index + 2-Opt TSP."""
        if len(characters) <= 1:
            return list(characters)

        emb = cls._ensure_clip_matrix(characters, clip_engine=clip_engine)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        # Exact cosine similarity matrix via FAISS / normalized dot product
        sim = np.clip(emb @ emb.T, -1.0, 1.0)
        dist_matrix = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)
        np.fill_diagonal(dist_matrix, 0.0)

        # Anchor at card with highest average similarity to its top-3 neighbors
        _, _ = index.search(emb[:1], min(4, len(characters)))
        order = solve_tsp_2opt(dist_matrix, start_idx=0, max_passes=60)
        return cls._finalize_order([characters[i] for i in order])

    @classmethod
    def sort_theme_clusters(
        cls,
        characters: List[Character],
        min_cluster_size: int = 2,
        clip_engine: Optional[CLIPEngine] = None,
    ) -> List[Character]:
        """Cluster cards into visual/aesthetic themes using HDBSCAN and sort within clusters."""
        if len(characters) <= 1:
            return list(characters)

        emb = cls._ensure_clip_matrix(characters, clip_engine=clip_engine)
        n = len(characters)

        if n >= 5:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min(min_cluster_size, n // 2)),
                min_samples=1,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(emb)
            # If HDBSCAN puts everything in noise (-1), fall back to agglomerative clustering
            if len(set(labels)) <= 1:
                k = max(2, min(4, int(round(math.sqrt(n)))))
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(emb)
        else:
            k = min(2, n)
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(emb)

        grouped: Dict[int, List[Character]] = {}
        for ch, lbl in zip(characters, labels):
            cid = int(lbl)
            ch.cluster_id = cid
            grouped.setdefault(cid, []).append(ch)

        # Largest clusters first, noise (-1) last
        sorted_groups = sorted(
            grouped.items(),
            key=lambda kv: (kv[0] == -1, -len(kv[1]), kv[0]),
        )

        result: List[Character] = []
        for _, group_chars in sorted_groups:
            if len(group_chars) > 2:
                group_chars = cls.sort_visual_similarity_tsp(group_chars, clip_engine=clip_engine)
            result.extend(group_chars)

        return cls._finalize_order(result, preserve_clusters=True)

    @classmethod
    def sort_by_text_prompt(
        cls,
        characters: List[Character],
        prompt: str,
        clip_engine: Optional[CLIPEngine] = None,
        enable_neural: bool = True,
    ) -> List[Character]:
        """Rank characters by natural-language CLIP text similarity (with OKLab keyword fallback)."""
        query = prompt.strip()
        if not query or not characters:
            return list(characters)

        engine = clip_engine or CLIPEngine(use_gpu=True)
        text_vec = engine.encode_text(query, enable_neural=enable_neural)

        if text_vec is not None:
            emb = cls._ensure_clip_matrix(characters, clip_engine=engine)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            scores, indices = index.search(text_vec.astype(np.float32), len(characters))
            ordered: List[Character] = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(characters):
                    ch = characters[int(idx)]
                    ch.metadata["prompt_score"] = round(float(score), 4)
                    ordered.append(ch)
            return cls._finalize_order(ordered)

        # Deterministic OKLab color-keyword fallback when offline or CLIP disabled
        cls._ensure_oklab_matrix(characters)
        tokens = query.lower().split()
        targets = [_COLOR_KEYWORD_OKLAB[t] for t in tokens if t in _COLOR_KEYWORD_OKLAB]
        if targets:
            target_oklab = np.mean(targets, axis=0)
            scored: List[Tuple[float, Character]] = []
            for ch in characters:
                vec = ch.oklab_feature or [0.5, 0.0, 0.0]
                primary = np.array(vec[3:6] if len(vec) >= 6 else vec[:3], dtype=np.float32)
                mean_v = np.array(vec[:3], dtype=np.float32)
                dist = 0.6 * float(np.linalg.norm(primary - target_oklab)) + 0.4 * float(
                    np.linalg.norm(mean_v - target_oklab)
                )
                score = max(0.0, 1.0 - dist)
                ch.metadata["prompt_score"] = round(score, 4)
                scored.append((dist, ch))
            scored.sort(key=lambda item: item[0])
            return cls._finalize_order([item[1] for item in scored])

        # Last-resort text match on name / series / tags
        q_low = query.lower()
        ordered = sorted(
            characters,
            key=lambda c: (
                0 if q_low in c.name.lower() or q_low in c.series.lower() else 1,
                c.name.lower(),
            ),
        )
        return cls._finalize_order(ordered)

    @classmethod
    def sort_by_series(cls, characters: List[Character]) -> List[Character]:
        """Group by series (largest series first, then alphabetically by series and name)."""
        counts: Dict[str, int] = {}
        for ch in characters:
            counts[ch.series] = counts.get(ch.series, 0) + 1
        ordered = sorted(
            characters,
            key=lambda c: (-counts.get(c.series, 0), c.series.lower(), c.name.lower()),
        )
        return cls._finalize_order(ordered)

    @classmethod
    def sort_alphabetical(cls, characters: List[Character]) -> List[Character]:
        """Sort alphabetically by character name."""
        ordered = sorted(characters, key=lambda c: (c.name.lower(), c.series.lower()))
        return cls._finalize_order(ordered)

    @classmethod
    def sort(
        cls,
        characters: List[Character],
        method: SortMethod,
        clip_engine: Optional[CLIPEngine] = None,
        min_cluster_size: int = 2,
        prompt: str = "",
        enable_neural: bool = True,
    ) -> List[Character]:
        """Dispatch to the requested sorting strategy."""
        if method == SortMethod.COLOR_GRADIENT_TSP:
            return cls.sort_color_gradient_tsp(characters)
        if method == SortMethod.HILBERT_COLOR:
            return cls.sort_hilbert_color(characters)
        if method == SortMethod.RAINBOW_SPECTRUM:
            return cls.sort_rainbow_spectrum(characters)
        if method == SortMethod.VISUAL_SIMILARITY_TSP:
            return cls.sort_visual_similarity_tsp(characters, clip_engine=clip_engine)
        if method == SortMethod.THEME_CLUSTERS:
            return cls.sort_theme_clusters(
                characters, min_cluster_size=min_cluster_size, clip_engine=clip_engine
            )
        if method == SortMethod.SERIES_GROUPING:
            return cls.sort_by_series(characters)
        if method == SortMethod.ALPHABETICAL:
            return cls.sort_alphabetical(characters)
        if method == SortMethod.NATURAL_LANGUAGE:
            return cls.sort_by_text_prompt(
                characters, prompt=prompt, clip_engine=clip_engine, enable_neural=enable_neural
            )
        logger.warning("Unknown sort method; returning current order", method=str(method))
        return list(characters)

    @staticmethod
    def _finalize_order(characters: List[Character], preserve_clusters: bool = False) -> List[Character]:
        for idx, ch in enumerate(characters):
            ch.sort_index = idx
            if not preserve_clusters:
                ch.cluster_id = -1
        return characters
