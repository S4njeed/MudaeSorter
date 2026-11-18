# Mudae Power Tools Pro v3.0

A complete, production-grade desktop application that captures, indexes, visually sorts, and exports Discord **Mudae** character collections (harems) into smooth color-gradient or AI theme-clustered layouts and ready-to-paste `$sm` / `$smp` Discord commands.

---

## 1. Tech Stack & Research Rationale

| Layer | Selected Stack | Why It Wins Over Alternatives |
| :--- | :--- | :--- |
| **Desktop UI** | **PySide6 (Qt6)** | Native scrollable `QGridLayout` card gallery, true translucent multi-monitor overlay (`RegionSelectorOverlay`), native drag-and-drop reordering (`QDrag`), and clean `QThread` signals without Tkinter thread freezes. |
| **Screen Capture & Input** | **MSS + Win32 DPI Awareness + PyAutoGUI** | `mss` captures raw BGRA buffers in milliseconds; explicit `bgra_to_rgb()` eliminates red/blue channel inversion; `SetProcessDpiAwareness(2)` + Win32 `GetCursorPos` eliminates Windows 125%/150% DPI coordinate drift. |
| **OCR & Typo Repair** | **RapidOCR (`onnxruntime`) + `FuzzyLexiconCorrector`** | Runs PaddleOCR PP-OCR weights on ONNX Runtime in ~30–80 ms on CPU without heavy PaddlePaddle/PyTorch EasyOCR overhead. 2x bicubic upscaling boosts accuracy on small Discord text, while `FuzzyLexiconCorrector` snaps OCR typos (`Zer0 Two` → `Zero Two`) against imported Mudae lists. |
| **Perceptual Deduplication** | **ImageHash (`dhash` + `phash` + `whash` + `colorhash`)** | Combining structural grayscale hashes with HSV `colorhash` prevents duplicate Discord frames during lag while still separating distinct cards that share a silhouette. |
| **Color Science & Path Optimization** | **OKLab + MiniBatchKMeans + 2-Opt TSP & 3D Hilbert Curve** | **OKLab** matches human lightness/chroma perception (unlike RGB/HSL). Center-weighted 5-color palettes feed either an open-path **2-Opt Traveling Salesperson (TSP)** solver or a **3D Hilbert curve** traversal for seamless visual color gradients. |
| **Vision-Language AI** | **Google SigLIP (`google/siglip-base-patch16-224`) / OpenAI CLIP (`openai/clip-vit-base-patch32`) + FAISS + HDBSCAN** | SigLIP’s pairwise sigmoid loss achieves stronger zero-shot anime/illustration retrieval and natural-language prompt matching than softmax CLIP, indexed via `faiss.IndexFlatIP` and automatically grouped into aesthetic themes via `HDBSCAN`. |
| **Storage** | **SQLite (WAL) + FTS5 + Msgpack** | Zero-setup local database with `INSERT`/`UPDATE`/`DELETE` FTS5 triggers, compact binary embedding storage via `msgpack`, and automatic connection closing on every query. |
| **Discord Export** | **Chunked `$sm` + `$smp` (`$sortmarrypos`)** | Automatically deduplicates character names and splits long harems under Discord’s 2,000-character limit using `$sm` for Batch 1 and `$smp <LastCharOfPrevBatch>` for subsequent batches. |

---

## 2. Project Structure

```text
MudaePowerTools/
├── main.py                     # DPI-aware desktop entry point
├── run.bat / run.ps1           # One-click Windows launchers using .venv
├── requirements.txt            # Isolated pinned dependencies (numpy<2.0)
├── mudae_tools/
│   ├── __init__.py             # Windows PySide6/shiboken6 DLL search isolation
│   ├── config.py               # Pydantic v2 AppConfig, Character, SortMethod
│   ├── database.py             # Thread-safe SQLite + WAL + FTS5 + Msgpack
│   ├── capture.py              # DPI-aware MSS BGRA->RGB capture & safe clicking
│   ├── ocr.py                  # RapidOCR ONNX, Mudae list parser, fuzzy corrector
│   ├── vision.py               # OKLab palette extractor, ImageHasher, SigLIP/CLIP
│   ├── sorter.py               # 2-Opt TSP, 3D Hilbert curve, OKLCh, FAISS, HDBSCAN
│   ├── exporter.py             # $sm/$smp chunking & sorted folder image export
│   └── ui/
│       ├── overlay.py          # Multi-monitor translucent region selector
│       ├── widgets.py          # Card tile + drag-and-drop HaremGalleryWidget
│       ├── workers.py          # QThread workers (Scraper, Folder Import, Demo, Sort)
│       └── main_window.py      # 4-tab PySide6 desktop workspace
└── tests/
    └── test_pipeline.py        # End-to-end automated verification suite
```

---

## 3. Setup & Dependencies

Using an isolated `.venv` prevents `numpy 2.x` / `torchvision` / system `VCRUNTIME140.dll` conflicts:

```powershell
cd "C:\Users\sanje\.gemini\antigravity\scratch\MudaePowerTools"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

*(Optional GPU Acceleration)* If you want CUDA acceleration for SigLIP/CLIP on an NVIDIA GPU (such as an RTX 4070), install CUDA PyTorch inside `.venv`:

```powershell
.\.venv\Scripts\pip.exe install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

Launch the app from source:

```powershell
.\run.bat
```

### Building Standalone Single-File Binaries (Windows / macOS / Linux)

You can compile a single-file executable for your current OS with:

```powershell
.\.venv\Scripts\python.exe build_binary.py
```

This creates `dist/MudaePowerTools-Windows-x64.exe` (or `MudaePowerTools-macOS-arm64` / `MudaePowerTools-Linux-x64` on macOS/Linux) bundling PySide6, RapidOCR ONNX models, OKLab 2-Opt TSP / 3D Hilbert, FAISS, and HDBSCAN. Pass `--with-torch` if you want to embed PyTorch/Transformers inside the single binary as well.

Cross-platform builds for **Windows (`x64`)**, **macOS (`arm64`)**, and **Linux (`x64`)** also run automatically in GitHub Actions via [.github/workflows/build-binaries.yml](.github/workflows/build-binaries.yml) and attach binaries to GitHub Releases on version tags (`v*`).

Run verification tests:

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## 4. Quick-Start & Troubleshooting Guide

### A. Capturing Directly from Discord
1. Open Discord to your harem (`$mm i` or `$im <character>` / `$mmi` embed view with next/previous arrow buttons).
2. **(Recommended Tip)** Run `$mma` or `$mms` in Discord, copy the character list text, and click **📋 Paste Text List...** in Tab 1 first. This teaches the OCR fuzzy corrector every exact character & series name in your harem.
3. Click each region selector in **Tab 1 (Scraper & Import)**:
   - **Select Card Image Region**: Drag tightly around the character artwork image.
   - **Select Character Name OCR**: Drag around the character's name header.
   - **Select Series Name OCR**: Drag around the anime/game series subtitle.
   - **Select Next Button**: Click (or drag around) Discord's `▶` pagination button.
4. Set **Pages to capture** and **Delay** (`1.0–1.5s` works reliably with Discord rate limits), then click **▶ Start Scraping**.
   - Move the mouse to the top-left corner of your primary monitor at any time to trigger PyAutoGUI's emergency failsafe stop, or press **■ Stop**.

### B. Sorting & Fine-Tuning in the Harem Grid
1. Switch to **Tab 2 (AI Sorter & Grid)**.
2. Choose **5 cols (Mudae Page)** to preview exact Mudae Discord page boundaries (10 cards per `$mm i` page or 15 per page depending on layout).
3. Run **OKLab Gradient (2-Opt)** for the smoothest color flow, **3D Hilbert Curve** for fast locality-preserving grouping, **Vision Similarity (TSP)** for art-style continuity, **Theme Clusters (HDBSCAN)** for automatic aesthetic sections, or enter a natural-language prompt (e.g., `"pink hair"`, `"dark armor"`) in the search bar.
4. Fine-tune manually by **clicking two cards to swap**, **dragging & dropping**, using **◀ / ▶**, or **double-clicking** any card to edit its name/series.

### C. Exporting `$sm` / `$smp` Commands to Discord
1. Open **Tab 3 (Mudae $sm & Export)**.
2. Copy **Batch #1** (`$sm Char1 $Char2 ...`) and paste it into your Discord bot channel.
3. If your harem spans multiple batches, paste **Batch #2**, **Batch #3**, etc. in order—each subsequent batch starts with `$smp <LastCharOfPrevBatch> $...` (`$sortmarrypos`) so Mudae inserts the next batch immediately after the previous batch without overwriting your earlier sort.

### Troubleshooting
* **OCR skips or misreads stylized fonts**: Use **📋 Paste Text List...** in Tab 1 to populate the fuzzy lexicon, or double-click any card in Tab 2 to edit its name directly.
* **Duplicate frames skipped repeatedly**: Discord may take slightly longer than your delay to flip pages—increase **Delay (sec)** to `1.5s` or lower **Duplicate Frame Threshold** in Tab 4.
* **First neural SigLIP/CLIP sort pauses briefly**: `transformers` downloads the selected Hugging Face weights (`openai/clip-vit-base-patch32` or `google/siglip-base-patch16-224`) once and caches them locally under `~/.cache/huggingface`. You can also disable neural weights in Tab 4 for instant offline OKLab sorting.
