"""Mudae $sm ($sortmarry) command chunking and sorted folder image exporter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal

from PIL import Image

from .config import Character, ImageFormat, logger

_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str, fallback: str = "character") -> str:
    """Create a Windows-safe filename fragment."""
    cleaned = _UNSAFE_FILENAME_RE.sub("_", name).strip(" .")
    return cleaned or fallback


def generate_sm_commands(
    characters: List[Character],
    prefix: str = "$",
    command: str = "sm",
    max_length: int = 1950,
    use_pos_chaining: bool = True,
) -> List[str]:
    """Build Discord-safe Mudae $sm commands chunked below Discord's 2000-character limit.

    Duplicates are deduplicated while preserving first-seen sorted order.
    When `use_pos_chaining` is enabled and multiple batches are needed:
      Batch 1: `$sm Char1 $Char2 ... $CharK`
      Batch 2: `$smp CharK $Char(K+1) $Char(K+2) ...`
    So pasting batches sequentially in Discord orders the full harem without overwriting
    earlier batches.
    """
    names: List[str] = []
    seen: set[str] = set()
    for ch in characters:
        clean = ch.name.replace("$", "").strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            names.append(clean)
    if not names:
        return []

    prefix = prefix or "$"
    command = (command or "sm").strip()
    pos_command = f"{command}p" if command == "sm" else f"{command} pos"
    max_length = max(120, min(2000, int(max_length)))

    batches: List[str] = []
    idx = 0
    prev_anchor: str | None = None

    while idx < len(names):
        if not batches or not use_pos_chaining or prev_anchor is None:
            header = f"{prefix}{command} "
            first_name = names[idx]
            current = header + first_name
            last_added = first_name
            idx += 1
        else:
            header = f"{prefix}{pos_command} {prev_anchor}"
            first_name = names[idx]
            candidate = f"{header} ${first_name}"
            if len(candidate) > max_length:
                header = f"{prefix}{command} "
                current = header + first_name
            else:
                current = candidate
            last_added = first_name
            idx += 1

        while idx < len(names):
            nxt = names[idx]
            addition = f" ${nxt}"
            if len(current) + len(addition) <= max_length:
                current += addition
                last_added = nxt
                idx += 1
            else:
                break

        batches.append(current)
        prev_anchor = last_added

    return batches


def generate_mma_text(characters: List[Character]) -> str:
    """Export a clean plain-text list matching Mudae's character - series summary."""
    lines = [f"{i + 1}. {ch.name} — {ch.series}" for i, ch in enumerate(characters)]
    return "\n".join(lines)


def export_sorted_folder(
    characters: List[Character],
    dest_dir: Path,
    group_by: Literal["flat", "series", "cluster"] = "flat",
    image_format: ImageFormat = ImageFormat.PNG,
    jpeg_quality: int = 95,
) -> int:
    """Export sorted character images with sequential rank prefixes into `dest_dir`."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if image_format == ImageFormat.JPEG else image_format.value.lower()

    exported = 0
    for idx, ch in enumerate(characters, start=1):
        if not ch.image_path.exists():
            logger.warning("Skipping missing image during export", path=str(ch.image_path))
            continue

        if group_by == "series":
            subdir = dest_dir / sanitize_filename(ch.series, "Unknown_Series")
        elif group_by == "cluster":
            label = f"Cluster_{ch.cluster_id}" if ch.cluster_id >= 0 else "Unclustered"
            subdir = dest_dir / label
        else:
            subdir = dest_dir

        subdir.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(ch.name, f"card_{idx}")
        out_path = subdir / f"{idx:04d}_{safe_name}.{ext}"

        try:
            with Image.open(ch.image_path) as img:
                save_img = img.convert("RGB") if image_format == ImageFormat.JPEG else img
                save_kwargs = {"quality": jpeg_quality} if image_format in (ImageFormat.JPEG, ImageFormat.WEBP) else {}
                save_img.save(out_path, format=image_format.value, **save_kwargs)
            exported += 1
        except Exception as exc:
            logger.error("Failed to export character image", name=ch.name, error=str(exc))

    return exported
