"""Cross-platform PyInstaller single-file binary builder for Mudae Power Tools Pro."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path


def target_binary_name() -> str:
    sys_name = platform.system()
    os_label = {
        "Windows": "Windows",
        "Darwin": "macOS",
        "Linux": "Linux",
    }.get(sys_name, sys_name)

    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        arch_label = "x64"
    elif machine in ("arm64", "aarch64"):
        arch_label = "arm64"
    else:
        arch_label = machine or "unknown"
    return f"MudaePowerTools-{os_label}-{arch_label}"


def _write_qt_runtime_hook(build_dir: Path) -> Path:
    """Create a PyInstaller runtime hook that preloads system ICU and Qt6/MSVC DLLs on Windows."""
    build_dir.mkdir(parents=True, exist_ok=True)
    hook_path = build_dir / "rthook_pyside6_dlls.py"
    hook_path.write_text(
        '''"""Preload Windows system ICU + PySide6/shiboken6 DLLs inside frozen bundles."""
import ctypes
import os
import sys
from pathlib import Path

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _k32 = ctypes.windll.kernel32
    if "--smoke-test" in sys.argv:
        _k32.SetErrorMode(0x0001 | 0x0002)

    _sys_icu = Path(os.environ.get("SystemRoot", r"C:\\Windows")) / "System32" / "icuuc.dll"
    if _sys_icu.is_file():
        try:
            ctypes.WinDLL(str(_sys_icu))
        except Exception:
            pass

    _base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    _ps = _base / "PySide6" if (_base / "PySide6").is_dir() else _base
    _shib = _base / "shiboken6" if (_base / "shiboken6").is_dir() else _base
    if (_ps / "plugins").is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(_ps / "plugins"))
    for _d in dict.fromkeys((_ps, _shib)):
        if _d.is_dir():
            try:
                os.add_dll_directory(str(_d))
            except Exception:
                pass
    _k32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    _k32.LoadLibraryExW.restype = ctypes.c_void_p
    for _dll in (
        _ps / "msvcp140.dll",
        _ps / "msvcp140_1.dll",
        _ps / "msvcp140_2.dll",
        _ps / "vcruntime140_1.dll",
        _shib / "shiboken6.abi3.dll",
        _ps / "pyside6.abi3.dll",
        _ps / "Qt6Core.dll",
        _ps / "Qt6Gui.dll",
        _ps / "Qt6Widgets.dll",
    ):
        if _dll.is_file():
            _k32.LoadLibraryExW(str(_dll), None, 0x00000008)
''',
        encoding="utf-8",
    )
    return hook_path


def _write_spec_file(
    root: Path,
    name: str,
    rthook: Path,
    with_torch: bool,
) -> Path:
    """Generate a cross-platform .spec file that strips conflicting Anaconda ICU DLLs on Windows."""
    import PySide6

    pyside_dir = Path(PySide6.__file__).resolve().parent
    spec_path = root / f"{name}.spec"

    excludes = [
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.QtPdf",
        "PySide6.QtCharts",
        "PySide6.QtMultimedia",
        "PySide6.QtDataVisualization",
        "tkinter",
        "matplotlib",
    ]
    if not with_torch:
        excludes.extend(["torch", "torchvision", "torchaudio", "transformers", "huggingface_hub", "tokenizers"])

    extra_datas: list[tuple[str, str]] = []
    plugins_root = pyside_dir / "plugins"
    for plugin_sub in ("platforms", "styles", "imageformats"):
        p_dir = plugins_root / plugin_sub
        if p_dir.is_dir():
            extra_datas.append((str(p_dir), f"PySide6/plugins/{plugin_sub}"))

    extra_binaries: list[tuple[str, str]] = []
    if platform.system() == "Windows":
        for dll_name in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            dll_src = pyside_dir / dll_name
            if dll_src.is_file():
                extra_binaries.append((str(dll_src), "PySide6"))
                extra_binaries.append((str(dll_src), "."))

    bundle_block = ""
    if platform.system() == "Darwin":
        bundle_block = f"""
app = BUNDLE(
    exe,
    name={repr(name + '.app')},
    icon=None,
    bundle_identifier='com.s4njeed.mudaepowertools',
)
"""

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas = {repr(extra_datas)}
binaries = {repr(extra_binaries)}
hiddenimports = [
    'sklearn.neighbors._partition_nodes',
    'scipy.special._cdflib',
    'scipy.spatial.transform._rotation_groups',
    'msgpack',
    'structlog',
]

datas += collect_data_files('rapidocr_onnxruntime')
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('faiss')
hiddenimports += collect_submodules('faiss')
hiddenimports += collect_submodules('hdbscan')
{"datas += collect_data_files('transformers')" if with_torch else ""}

a = Analysis(
    [{repr(str(root / 'main.py'))}],
    pathex=[{repr(str(root))}],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[{repr(str(rthook))}],
    excludes={repr(excludes)},
    noarchive=False,
    optimize=0,
)

# Remove Anaconda ICU DLLs (icuuc.dll, icudt*.dll, etc.) that shadow Windows System32\\icuuc.dll
if sys.platform == 'win32':
    a.binaries = [
        b for b in a.binaries
        if not (
            Path(b[0]).name.lower().startswith('icu')
            and 'anaconda' in str(b[1]).lower()
        )
    ]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name={repr(name)},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
{bundle_block}
"""
    spec_path.write_text(spec_content, encoding="utf-8")
    return spec_path


def build(with_torch: bool = False) -> Path:
    import PyInstaller.__main__

    root = Path(__file__).resolve().parent
    name = target_binary_name()
    rthook = _write_qt_runtime_hook(root / "build")
    spec_path = _write_spec_file(root, name, rthook, with_torch=with_torch)

    print(f"Building standalone binary '{name}' (with_torch={with_torch})...")
    PyInstaller.__main__.run(
        [
            str(spec_path),
            "--noconfirm",
            "--clean",
            "--distpath",
            str(root / "dist"),
            "--workpath",
            str(root / "build"),
        ]
    )

    ext = ".exe" if platform.system() == "Windows" else ""
    out_path = root / "dist" / f"{name}{ext}"
    if not out_path.exists():
        raise FileNotFoundError(f"Expected built binary not found: {out_path}")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Built: {out_path} ({size_mb:.1f} MB)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build standalone Mudae Power Tools Pro binary.")
    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Bundle PyTorch & Transformers (~500+ MB larger) instead of lightweight OKLab+FAISS+RapidOCR build.",
    )
    opts = parser.parse_args()
    build(with_torch=opts.with_torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
