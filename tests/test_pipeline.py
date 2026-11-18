"""End-to-end verification tests for Mudae Power Tools Pro v3.0."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mudae_tools.capture import bgra_to_rgb
from mudae_tools.config import AppConfig, Character, ImageFormat, SortMethod
from mudae_tools.database import DatabaseManager
from mudae_tools.exporter import export_sorted_folder, generate_sm_commands
from mudae_tools.ocr import FuzzyLexiconCorrector, OCRService, clean_mudae_name, parse_mudae_text_list
from mudae_tools.sorter import AdvancedSorter, compute_path_distance
from mudae_tools.vision import CLIPEngine, ColorAnalyzer, ImageHasher


def _make_card_image(rgb: tuple[int, int, int], accent: tuple[int, int, int], size=(128, 192)) -> Image.Image:
    img = Image.new("RGB", size, rgb)
    draw = ImageDraw.Draw(img)
    draw.ellipse((24, 32, size[0] - 24, size[1] - 48), fill=accent)
    return img


class TestMudaePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = DatabaseManager(self.root / "test.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bgra_to_rgb_and_phash_duplicate_detection(self) -> None:
        # Pure Red in BGRA has B=0, G=0, R=255, A=255
        bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        bgra[:, :, 2] = 255
        bgra[:, :, 3] = 255
        rgb = bgra_to_rgb(bgra)
        self.assertEqual(tuple(rgb[0, 0]), (255, 0, 0))

        img1 = _make_card_image((220, 40, 60), (255, 210, 210))
        img2 = img1.copy()
        img3 = Image.new("RGB", (128, 192), (20, 60, 220))
        d3 = ImageDraw.Draw(img3)
        d3.polygon([(16, 170), (64, 24), (112, 170)], fill=(140, 220, 255))
        d3.rectangle((20, 20, 108, 44), fill=(255, 255, 255))

        h1 = ImageHasher.compute_hash(img1)
        h2 = ImageHasher.compute_hash(img2)
        h3 = ImageHasher.compute_hash(img3)
        self.assertLess(ImageHasher.normalized_distance(h1, h2), 0.01)
        self.assertGreater(ImageHasher.normalized_distance(h1, h3), 0.1)

    def test_database_crud_and_fts_triggers(self) -> None:
        p1 = self.root / "rem.png"
        _make_card_image((90, 160, 250), (230, 240, 255)).save(p1)
        c1 = Character(name="Rem", series="Re:Zero", image_path=p1, phash="a_b_c", tags=["maid", "blue"])
        self.assertTrue(self.db.upsert_character(c1))

        self.assertEqual(len(self.db.search_characters("Rem")), 1)
        self.assertTrue(self.db.update_character_fields(c1.id, name="Ram", series="Re:Zero", tags=["maid", "pink"]))
        self.assertEqual(self.db.search_characters("Ram")[0].name, "Ram")
        self.assertEqual(len(self.db.search_characters("Rem")), 0)

        self.assertTrue(self.db.delete_character(c1.id))
        self.assertEqual(len(self.db.get_all_characters()), 0)

    def test_legacy_v2_database_migration(self) -> None:
        import sqlite3

        legacy_db_path = self.root / "legacy_v2.db"
        conn = sqlite3.connect(str(legacy_db_path))
        conn.execute(
            """
            CREATE TABLE characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                show_name TEXT NOT NULL,
                image_path TEXT NOT NULL UNIQUE,
                hash_value TEXT NOT NULL,
                embedding BLOB,
                confidence REAL DEFAULT 0.0,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT 1.0,
                updated_at REAL DEFAULT 1.0
            )
            """
        )
        conn.execute(
            "INSERT INTO characters (id, name, show_name, image_path, hash_value, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            ("id-1", "Saber", "Fate/stay night", str(self.root / "saber.png"), "abc", 0.92),
        )
        conn.commit()
        conn.close()

        migrated = DatabaseManager(legacy_db_path)
        chars = migrated.get_all_characters()
        self.assertEqual(len(chars), 1)
        self.assertEqual(chars[0].name, "Saber")
        self.assertEqual(chars[0].series, "Fate/stay night")
        self.assertEqual(chars[0].phash, "abc")
        self.assertAlmostEqual(chars[0].ocr_confidence, 0.92, places=2)

    def test_ocr_clean_list_parser_and_fuzzy_corrector(self) -> None:
        cleaned = clean_mudae_name("Zero Two #12 500 ka")
        self.assertEqual(cleaned, "Zero Two")

        raw_list = """
        Page 1 / 4
        1. Zero Two - Darling in the FranXX · 520 ka
        2. Rem — Re:Zero (490 ka)
        3. Megumin | KonoSuba
        """
        parsed = parse_mudae_text_list(raw_list)
        self.assertEqual(
            parsed,
            [
                ("Zero Two", "Darling in the FranXX"),
                ("Rem", "Re:Zero"),
                ("Megumin", "KonoSuba"),
            ],
        )

        sm_parsed = parse_mudae_text_list("$sm Zero Two $ Rem $ Megumin")
        self.assertEqual([n for n, _ in sm_parsed], ["Zero Two", "Rem", "Megumin"])

        corrector = FuzzyLexiconCorrector()
        corrector.add_entries(names=[n for n, _ in parsed], series=[s for _, s in parsed])
        self.assertEqual(corrector.correct_name("Zer0 Two"), "Zero Two")
        self.assertEqual(corrector.correct_name("Megurnin"), "Megumin")

        banner = Image.new("RGB", (320, 64), (49, 51, 56))
        draw = ImageDraw.Draw(banner)
        draw.text((20, 22), "Zero Two", fill=(255, 255, 255))
        ocr = OCRService()
        self.assertTrue(ocr.is_available())
        text, conf = ocr.extract_text(np.asarray(banner, dtype=np.uint8))
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(conf, 0.0)

    def test_oklab_tsp_hilbert_rainbow_and_clip_clustering(self) -> None:
        palette_specs = [
            ("RedCard", "Fire", (230, 30, 40), (255, 140, 120)),
            ("BlueCard", "Water", (30, 80, 235), (140, 200, 255)),
            ("OrangeCard", "Fire", (245, 130, 40), (255, 210, 140)),
            ("CyanCard", "Water", (40, 210, 220), (190, 250, 255)),
            ("PinkCard", "Sakura", (245, 120, 180), (255, 215, 235)),
            ("DarkCard", "Shadow", (25, 25, 32), (90, 90, 105)),
            ("GreenCard", "Forest", (50, 190, 80), (190, 245, 180)),
            ("WhiteCard", "Light", (245, 245, 248), (210, 220, 230)),
        ]

        chars: list[Character] = []
        clip = CLIPEngine(use_gpu=False)
        imgs: list[Image.Image] = []
        for idx, (name, series, bg, fg) in enumerate(palette_specs):
            p = self.root / f"{name}.png"
            img = _make_card_image(bg, fg)
            img.save(p)
            imgs.append(img)
            hx, ok_vec = ColorAnalyzer.extract_palette(img)
            self.assertEqual(len(hx), 5)
            self.assertEqual(len(ok_vec), 18)
            chars.append(
                Character(
                    name=name,
                    series=series,
                    image_path=p,
                    dominant_hex=hx,
                    oklab_feature=ok_vec,
                    sort_index=idx,
                )
            )

        embs = clip.encode_images(imgs, batch_size=8, enable_neural=False)
        self.assertEqual(embs.shape, (len(chars), 512))
        for ch, emb in zip(chars, embs):
            ch.clip_embedding = emb.tolist()
            self.db.upsert_character(ch)

        dist_mat = AdvancedSorter.oklab_distance_matrix(chars)
        unsorted_cost = compute_path_distance(dist_mat, list(range(len(chars))))

        tsp_sorted = AdvancedSorter.sort(chars, SortMethod.COLOR_GRADIENT_TSP)
        id_to_idx = {ch.id: i for i, ch in enumerate(chars)}
        tsp_order = [id_to_idx[ch.id] for ch in tsp_sorted]
        tsp_cost = compute_path_distance(dist_mat, tsp_order)
        self.assertLessEqual(tsp_cost, unsorted_cost + 1e-6)

        hilbert = AdvancedSorter.sort(chars, SortMethod.HILBERT_COLOR)
        self.assertEqual(len(hilbert), len(chars))

        rainbow = AdvancedSorter.sort(chars, SortMethod.RAINBOW_SPECTRUM)
        self.assertEqual(len(rainbow), len(chars))

        sim_sorted = AdvancedSorter.sort(chars, SortMethod.VISUAL_SIMILARITY_TSP, clip_engine=clip)
        self.assertEqual(len(sim_sorted), len(chars))

        clustered = AdvancedSorter.sort(chars, SortMethod.THEME_CLUSTERS, min_cluster_size=2, clip_engine=clip)
        self.assertEqual(len(clustered), len(chars))

        prompted = AdvancedSorter.sort(
            chars, SortMethod.NATURAL_LANGUAGE, clip_engine=clip, prompt="red", enable_neural=False
        )
        self.assertEqual(prompted[0].name, "RedCard")

    def test_sm_smp_command_chunking_dedup_and_folder_export(self) -> None:
        p = self.root / "sample.png"
        _make_card_image((120, 80, 200), (230, 210, 255)).save(p)

        many_chars = [
            Character(name=f"Character_{i:03d}_LongName", series="TestSeries", image_path=p)
            for i in range(180)
        ]
        # Add duplicate name to verify deduplication preserves first-seen order
        many_chars.append(Character(name="Character_000_LongName", series="TestSeries", image_path=p))

        batches = generate_sm_commands(
            many_chars, prefix="$", command="sm", max_length=1950, use_pos_chaining=True
        )
        self.assertGreater(len(batches), 1)
        for idx, batch in enumerate(batches):
            self.assertLessEqual(len(batch), 1950)
            if idx == 0:
                self.assertTrue(batch.startswith("$sm "))
            else:
                self.assertTrue(batch.startswith("$smp "))

        # Confirm Character_000_LongName appears only once at the start of batch 0
        joined_tokens = " ".join(batches)
        self.assertEqual(joined_tokens.count("Character_000_LongName"), 1)

        export_dir = self.root / "exported"
        count = export_sorted_folder(many_chars[:5], export_dir, group_by="series", image_format=ImageFormat.PNG)
        self.assertEqual(count, 5)
        self.assertTrue((export_dir / "TestSeries" / "0001_Character_000_LongName.png").exists())

    def test_offscreen_pyside6_ui(self) -> None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        from mudae_tools.ui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        cfg = AppConfig(output_folder=self.root / "out", enable_clip=False)
        win = MainWindow(config=cfg, db=self.db)
        self.assertEqual(win.stack.count(), 4)
        self.assertGreaterEqual(win.vision_model_combo.count(), 2)
        for tab_idx in range(4):
            win._switch_tab(tab_idx)
        win.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
