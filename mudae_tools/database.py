"""Thread-safe SQLite storage with WAL mode, msgpack vectors, and FTS5 triggers."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import msgpack

from .config import DATABASE_FILE, Character, logger


class DatabaseManager:
    """High-performance SQLite manager for character cards, embeddings, and sort order."""

    def __init__(self, db_path: Path = DATABASE_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            with conn:
                yield conn
        finally:
            conn.close()

    _V3_SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS characters (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            series TEXT NOT NULL,
            image_path TEXT NOT NULL UNIQUE,
            phash TEXT NOT NULL DEFAULT '',
            dominant_hex TEXT NOT NULL DEFAULT '[]',
            oklab_feature BLOB,
            clip_embedding BLOB,
            ocr_confidence REAL NOT NULL DEFAULT 0.0,
            cluster_id INTEGER NOT NULL DEFAULT -1,
            sort_index INTEGER NOT NULL DEFAULT 0,
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            existing_tables = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "characters" in existing_tables:
                cols = {
                    r["name"]
                    for r in conn.execute("PRAGMA table_info(characters)").fetchall()
                }
                if "series" not in cols or "phash" not in cols or "oklab_feature" not in cols:
                    self._migrate_legacy_v2(conn, cols)

            conn.execute(self._V3_SCHEMA_SQL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chars_name ON characters(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chars_series ON characters(series)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chars_phash ON characters(phash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chars_sort ON characters(sort_index)")

            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS characters_fts USING fts5(
                    name, series, tags, content=characters, content_rowid=rowid
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS characters_fts_ai AFTER INSERT ON characters BEGIN
                    INSERT INTO characters_fts(rowid, name, series, tags)
                    VALUES (new.rowid, new.name, new.series, new.tags);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS characters_fts_ad AFTER DELETE ON characters BEGIN
                    INSERT INTO characters_fts(characters_fts, rowid, name, series, tags)
                    VALUES ('delete', old.rowid, old.name, old.series, old.tags);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS characters_fts_au AFTER UPDATE ON characters BEGIN
                    INSERT INTO characters_fts(characters_fts, rowid, name, series, tags)
                    VALUES ('delete', old.rowid, old.name, old.series, old.tags);
                    INSERT INTO characters_fts(rowid, name, series, tags)
                    VALUES (new.rowid, new.name, new.series, new.tags);
                END
                """
            )

    def _migrate_legacy_v2(self, conn: sqlite3.Connection, cols: set[str]) -> None:
        """Migrate an existing v2.0 schema (show_name/hash_value) non-destructively to v3.0."""
        logger.info("Migrating legacy characters table to v3.0 schema", columns=sorted(cols))
        for trig in ("characters_fts_insert", "characters_fts_ai", "characters_fts_ad", "characters_fts_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        conn.execute("DROP TABLE IF EXISTS characters_fts")
        conn.execute("DROP TABLE IF EXISTS characters_legacy_v2")
        conn.execute("ALTER TABLE characters RENAME TO characters_legacy_v2")
        conn.execute(self._V3_SCHEMA_SQL)

        series_col = "series" if "series" in cols else ("show_name" if "show_name" in cols else "'Unknown Series'")
        phash_col = "phash" if "phash" in cols else ("hash_value" if "hash_value" in cols else "''")
        conf_col = "ocr_confidence" if "ocr_confidence" in cols else ("confidence" if "confidence" in cols else "0.0")
        tags_col = "tags" if "tags" in cols else "'[]'"
        meta_col = "metadata" if "metadata" in cols else "'{}'"
        created_col = "created_at" if "created_at" in cols else "unixepoch()"
        updated_col = "updated_at" if "updated_at" in cols else "unixepoch()"

        conn.execute(
            f"""
            INSERT OR IGNORE INTO characters (
                id, name, series, image_path, phash, ocr_confidence,
                tags, metadata, created_at, updated_at
            )
            SELECT
                id,
                name,
                COALESCE({series_col}, 'Unknown Series'),
                image_path,
                COALESCE({phash_col}, ''),
                COALESCE({conf_col}, 0.0),
                COALESCE({tags_col}, '[]'),
                COALESCE({meta_col}, '{{}}'),
                COALESCE({created_col}, unixepoch()),
                COALESCE({updated_col}, unixepoch())
            FROM characters_legacy_v2
            """
        )
        conn.execute("DROP TABLE IF EXISTS characters_legacy_v2")

    def _pack_floats(self, values: Optional[List[float]]) -> Optional[bytes]:
        if values is None:
            return None
        return msgpack.packb([float(v) for v in values], use_bin_type=True)

    def _unpack_floats(self, blob: Optional[bytes]) -> Optional[List[float]]:
        if not blob:
            return None
        data = msgpack.unpackb(blob, raw=False)
        return [float(v) for v in data]

    def _row_to_character(self, row: sqlite3.Row) -> Character:
        return Character(
            id=row["id"],
            name=row["name"],
            series=row["series"],
            image_path=Path(row["image_path"]),
            phash=row["phash"],
            dominant_hex=json.loads(row["dominant_hex"] or "[]"),
            oklab_feature=self._unpack_floats(row["oklab_feature"]),
            clip_embedding=self._unpack_floats(row["clip_embedding"]),
            ocr_confidence=float(row["ocr_confidence"]),
            cluster_id=int(row["cluster_id"]),
            sort_index=int(row["sort_index"]),
            tags=json.loads(row["tags"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def upsert_character(self, character: Character) -> bool:
        """Insert or update a character keyed by image_path (preserving existing id if matched)."""
        now = time.time()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id, sort_index, created_at FROM characters WHERE image_path = ?",
                (str(character.image_path),),
            ).fetchone()

            if existing:
                char_id = existing["id"]
                sort_index = character.sort_index if character.sort_index else existing["sort_index"]
                created_at = existing["created_at"]
                conn.execute(
                    """
                    UPDATE characters
                    SET name = ?, series = ?, phash = ?, dominant_hex = ?,
                        oklab_feature = ?, clip_embedding = ?, ocr_confidence = ?,
                        cluster_id = ?, sort_index = ?, tags = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        character.name,
                        character.series,
                        character.phash,
                        json.dumps(character.dominant_hex),
                        self._pack_floats(character.oklab_feature),
                        self._pack_floats(character.clip_embedding),
                        float(character.ocr_confidence),
                        int(character.cluster_id),
                        int(sort_index),
                        json.dumps(character.tags),
                        json.dumps(character.metadata),
                        now,
                        char_id,
                    ),
                )
                character.id = char_id
                character.sort_index = sort_index
                character.created_at = created_at
                character.updated_at = now
                return True

            max_row = conn.execute("SELECT COALESCE(MAX(sort_index), -1) AS m FROM characters").fetchone()
            next_sort = character.sort_index if character.sort_index > 0 else int(max_row["m"]) + 1
            conn.execute(
                """
                INSERT INTO characters (
                    id, name, series, image_path, phash, dominant_hex,
                    oklab_feature, clip_embedding, ocr_confidence,
                    cluster_id, sort_index, tags, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character.id,
                    character.name,
                    character.series,
                    str(character.image_path),
                    character.phash,
                    json.dumps(character.dominant_hex),
                    self._pack_floats(character.oklab_feature),
                    self._pack_floats(character.clip_embedding),
                    float(character.ocr_confidence),
                    int(character.cluster_id),
                    int(next_sort),
                    json.dumps(character.tags),
                    json.dumps(character.metadata),
                    character.created_at,
                    now,
                ),
            )
            character.sort_index = next_sort
            character.updated_at = now
            return True

    def bulk_upsert_characters(self, characters: List[Character]) -> int:
        """Persist multiple characters in sequence."""
        count = 0
        for char in characters:
            if self.upsert_character(char):
                count += 1
        return count

    def get_all_characters(self) -> List[Character]:
        """Return all characters ordered by sort_index then created_at."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM characters ORDER BY sort_index ASC, created_at ASC"
            ).fetchall()
            return [self._row_to_character(r) for r in rows]

    def search_characters(self, query: str) -> List[Character]:
        """Full-text + substring search across name, series, and tags."""
        clean = query.strip()
        if not clean:
            return self.get_all_characters()

        with self._lock, self._connect() as conn:
            like_term = f"%{clean}%"
            rows = conn.execute(
                """
                SELECT * FROM characters
                WHERE name LIKE ? OR series LIKE ? OR tags LIKE ?
                ORDER BY sort_index ASC, created_at ASC
                """,
                (like_term, like_term, like_term),
            ).fetchall()
            return [self._row_to_character(r) for r in rows]

    def update_sort_orders(self, characters: List[Character]) -> None:
        """Persist updated sort_index and cluster_id for a sorted list."""
        now = time.time()
        with self._lock, self._connect() as conn:
            for idx, char in enumerate(characters):
                char.sort_index = idx
                char.updated_at = now
                conn.execute(
                    "UPDATE characters SET sort_index = ?, cluster_id = ?, metadata = ?, updated_at = ? WHERE id = ?",
                    (idx, int(char.cluster_id), json.dumps(char.metadata), now, char.id),
                )

    def update_character_fields(
        self,
        char_id: str,
        name: Optional[str] = None,
        series: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update editable character metadata."""
        updates: List[str] = []
        params: List[Any] = []
        if name is not None:
            updates.append("name = ?")
            params.append(name.strip() or "Unnamed")
        if series is not None:
            updates.append("series = ?")
            params.append(series.strip() or "Unknown Series")
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(char_id)

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE characters SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            return cur.rowcount > 0

    def delete_character(self, char_id: str) -> bool:
        """Remove a character record by id."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM characters WHERE id = ?", (char_id,))
            return cur.rowcount > 0

    def clear_all(self) -> int:
        """Delete all character rows from the database."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM characters")
            return cur.rowcount

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Compute collection statistics for the Analytics tab."""
        chars = self.get_all_characters()
        total = len(chars)
        series_counts: Dict[str, int] = {}
        cluster_counts: Dict[int, int] = {}
        clip_ready = 0
        oklab_ready = 0
        conf_sum = 0.0
        conf_count = 0

        for ch in chars:
            series_counts[ch.series] = series_counts.get(ch.series, 0) + 1
            cluster_counts[ch.cluster_id] = cluster_counts.get(ch.cluster_id, 0) + 1
            if ch.clip_embedding:
                clip_ready += 1
            if ch.oklab_feature:
                oklab_ready += 1
            if ch.ocr_confidence > 0:
                conf_sum += ch.ocr_confidence
                conf_count += 1

        top_series: List[Tuple[str, int]] = sorted(
            series_counts.items(), key=lambda x: (-x[1], x[0].lower())
        )
        return {
            "total_characters": total,
            "unique_series": len(series_counts),
            "clip_indexed": clip_ready,
            "oklab_indexed": oklab_ready,
            "avg_ocr_confidence": (conf_sum / conf_count) if conf_count else 0.0,
            "top_series": top_series,
            "cluster_counts": cluster_counts,
        }
