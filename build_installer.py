"""Cross-platform native installer builder for Mudae Power Tools Pro.

Produces:
  - Windows: .msi installer (Start Menu & Desktop shortcuts, Add/Remove Programs, CAB payload)
  - macOS:   .dmg drag-to-Applications disk image + .pkg native installer package
  - Linux:   .deb package (Debian/Ubuntu) + universal .tar.gz installer script bundle
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

from mudae_tools import APP_NAME, APP_VERSION

APP_IDENTIFIER = "com.s4njeed.mudaepowertools"
UPGRADE_GUID = "{8B4F2C10-5E3A-4A77-9D1C-6B8E1A42F301}"
AUTHOR = "Sanjeed"
HOMEPAGE = "https://github.com/S4njeed/MudaeSorter"


def detect_os_arch() -> tuple[str, str]:
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
        arch_label = machine or "x64"
    return os_label, arch_label


def generate_app_icon_png(dest_path: Path, size: int = 256) -> Path:
    """Render a crisp PNG app icon for desktop shortcuts and package assets."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (size, size), (19, 21, 28, 255))
    draw = ImageDraw.Draw(img)
    pad = int(size * 0.08)
    draw.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=int(size * 0.22),
        fill=(88, 101, 242, 255),
    )
    inner = int(size * 0.22)
    draw.ellipse((inner, inner, size - inner, size - inner), fill=(255, 255, 255, 240))
    core = int(size * 0.34)
    draw.ellipse((core, core, size - core, size - core), fill=(87, 242, 135, 255))
    img.save(dest_path, format="PNG")
    return dest_path


def generate_app_icon_ico(dest_path: Path) -> Path:
    """Render a multi-resolution Windows .ico icon for MSI shortcuts and Add/Remove Programs."""
    png_tmp = dest_path.with_suffix(".png")
    generate_app_icon_png(png_tmp, size=256)
    with Image.open(png_tmp) as img:
        img.save(dest_path, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
    png_tmp.unlink(missing_ok=True)
    return dest_path


def build_windows_msi(
    binary_path: Path,
    dist_dir: Path,
    version: str = APP_VERSION,
    arch: str = "x64",
) -> Path:
    """Build a standalone Windows .msi installer wrapping `binary_path`."""
    import msilib
    import msilib.schema
    import msilib.sequence

    if not binary_path.is_file():
        raise FileNotFoundError(f"Windows executable not found: {binary_path}")

    dist_dir.mkdir(parents=True, exist_ok=True)
    msi_path = dist_dir / f"MudaePowerTools-{version}-Windows-{arch}.msi"
    if msi_path.exists():
        msi_path.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged_exe = stage_dir / "MudaePowerTools.exe"
        shutil.copy2(binary_path, staged_exe)
        ico_path = generate_app_icon_ico(Path(tmp) / "MudaePowerTools.ico")

        # msilib maintains module-level _directories; clear so multiple calls in one process stay clean.
        msilib._directories.clear()
        msilib.current_feature = None

        product_code = "{" + str(uuid.uuid4()).upper() + "}"
        db = msilib.init_database(
            str(msi_path),
            msilib.schema,
            APP_NAME,
            product_code,
            version,
            AUTHOR,
        )

        # Populate standard execution sequences; omit missing UI Dialogs so silent/default UI works cleanly.
        exec_seq = [
            row for row in msilib.sequence.InstallExecuteSequence
            if row[0] not in ("RemoveExistingProducts",)
        ]
        exec_seq.append(("RemoveExistingProducts", None, 6550))
        msilib.add_data(db, "InstallExecuteSequence", exec_seq)
        msilib.add_data(db, "AdminExecuteSequence", msilib.sequence.AdminExecuteSequence)
        msilib.add_data(db, "AdminUISequence", [
            ("CostInitialize", None, 800),
            ("FileCost", None, 900),
            ("CostFinalize", None, 1000),
            ("ExecuteAction", None, 1300),
        ])
        msilib.add_data(db, "AdvtExecuteSequence", msilib.sequence.AdvtExecuteSequence)
        msilib.add_data(db, "InstallUISequence", [
            ("LaunchConditions", None, 100),
            ("FindRelatedProducts", None, 200),
            ("AppSearch", None, 400),
            ("CostInitialize", None, 800),
            ("FileCost", None, 900),
            ("CostFinalize", None, 1000),
            ("ExecuteAction", None, 1300),
        ])

        msilib.add_data(
            db,
            "Property",
            [
                ("UpgradeCode", UPGRADE_GUID),
                ("ALLUSERS", "2"),
                ("MSIINSTALLPERUSER", "1"),
                ("ARPNOMODIFY", "1"),
                ("ARPCOMMENTS", "AI Harem Scraper, OKLab Color Gradient & Mudae $sm Sorter"),
                ("ARPCONTACT", AUTHOR),
                ("ARPURLINFOABOUT", HOMEPAGE),
                ("ARPPRODUCTICON", "MudaePowerTools.ico"),
            ],
        )

        msilib.add_data(
            db,
            "Icon",
            [
                ("MudaePowerTools.ico", msilib.Binary(str(ico_path))),
            ],
        )

        msilib.add_data(
            db,
            "Upgrade",
            [
                (UPGRADE_GUID, None, version, None, 1, None, "OLD_PRODUCT_FOUND"),
            ],
        )

        cab = msilib.CAB("mudae")
        feature = msilib.Feature(
            db,
            "MainFeature",
            APP_NAME,
            "Mudae Power Tools Pro Application & Local ONNX OCR Engine",
            1,
            directory="INSTALLDIR",
        )
        feature.set_current()

        root_dir = msilib.Directory(db, cab, None, str(stage_dir), "TARGETDIR", "SourceDir")
        pf_folder = "ProgramFiles64Folder" if msilib.AMD64 else "ProgramFilesFolder"
        pf_dir = msilib.Directory(db, cab, root_dir, ".", pf_folder, ".")
        install_dir = msilib.Directory(
            db,
            cab,
            pf_dir,
            ".",
            "INSTALLDIR",
            "MUDAET~1|Mudae Power Tools Pro",
        )
        msilib.add_data(
            db,
            "Directory",
            [
                ("ProgramMenuFolder", "TARGETDIR", "."),
                ("DesktopFolder", "TARGETDIR", "."),
            ],
        )

        install_dir.start_component(
            component="MudaePowerToolsMain",
            feature=feature,
            flags=0,
            keyfile="MudaePowerTools.exe",
        )
        exe_file_id = install_dir.add_file("MudaePowerTools.exe")

        msilib.add_data(
            db,
            "Shortcut",
            [
                (
                    "StartMenuShortcut",
                    "ProgramMenuFolder",
                    "MUDAE~1|Mudae Power Tools Pro",
                    "MudaePowerToolsMain",
                    f"[#{exe_file_id}]",
                    None,
                    "Launch Mudae Power Tools Pro",
                    None,
                    "MudaePowerTools.ico",
                    0,
                    1,
                    "INSTALLDIR",
                ),
                (
                    "DesktopShortcut",
                    "DesktopFolder",
                    "MUDAE~1|Mudae Power Tools Pro",
                    "MudaePowerToolsMain",
                    f"[#{exe_file_id}]",
                    None,
                    "Launch Mudae Power Tools Pro",
                    None,
                    "MudaePowerTools.ico",
                    0,
                    1,
                    "INSTALLDIR",
                ),
            ],
        )

        cab.commit(db)
        db.Commit()
        db.Close()

    return msi_path


def stage_macos_app_bundle(
    binary_path: Path,
    stage_root: Path,
    version: str = APP_VERSION,
) -> Path:
    """Create or copy a clean `MudaePowerTools.app` bundle inside `stage_root`."""
    app_dir = stage_root / "MudaePowerTools.app"
    existing_app = binary_path.parent / f"{binary_path.name}.app"
    if existing_app.is_dir():
        shutil.copytree(existing_app, app_dir, symlinks=True)
        return app_dir

    macos_dir = app_dir / "Contents" / "MacOS"
    res_dir = app_dir / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    target_bin = macos_dir / "MudaePowerTools"
    shutil.copy2(binary_path, target_bin)
    target_bin.chmod(target_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    generate_app_icon_png(res_dir / "MudaePowerTools.png", size=256)

    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>{APP_NAME}</string>
    <key>CFBundleExecutable</key>
    <string>MudaePowerTools</string>
    <key>CFBundleIdentifier</key>
    <string>{APP_IDENTIFIER}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>{APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    (app_dir / "Contents" / "Info.plist").write_text(info_plist, encoding="utf-8")
    return app_dir


def build_macos_installers(
    binary_path: Path,
    dist_dir: Path,
    version: str = APP_VERSION,
    arch: str = "arm64",
) -> List[Path]:
    """Build `.dmg` and `.pkg` installers on macOS using `hdiutil` and `pkgbuild`."""
    if not binary_path.is_file():
        raise FileNotFoundError(f"macOS binary not found: {binary_path}")

    dist_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dmg_stage = tmp_path / "dmg_stage"
        dmg_stage.mkdir(parents=True, exist_ok=True)
        app_bundle = stage_macos_app_bundle(binary_path, dmg_stage, version=version)

        apps_link = dmg_stage / "Applications"
        if not apps_link.exists():
            os.symlink("/Applications", apps_link)

        dmg_path = dist_dir / f"MudaePowerTools-{version}-macOS-{arch}.dmg"
        if dmg_path.exists():
            dmg_path.unlink()
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                APP_NAME,
                "-srcfolder",
                str(dmg_stage),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            check=True,
        )
        outputs.append(dmg_path)

        pkg_stage = tmp_path / "pkg_stage"
        pkg_stage.mkdir(parents=True, exist_ok=True)
        shutil.copytree(app_bundle, pkg_stage / "MudaePowerTools.app", symlinks=True)

        pkg_path = dist_dir / f"MudaePowerTools-{version}-macOS-{arch}.pkg"
        if pkg_path.exists():
            pkg_path.unlink()
        subprocess.run(
            [
                "pkgbuild",
                "--root",
                str(pkg_stage),
                "--identifier",
                APP_IDENTIFIER,
                "--version",
                version,
                "--install-location",
                "/Applications",
                str(pkg_path),
            ],
            check=True,
        )
        outputs.append(pkg_path)

    return outputs


def stage_linux_package_tree(
    binary_path: Path,
    stage_root: Path,
    version: str = APP_VERSION,
    arch: str = "x64",
) -> tuple[Path, Path]:
    """Prepare Debian `.deb` tree and universal `.tar.gz` installer staging."""
    deb_arch = "amd64" if arch in ("x64", "amd64", "x86_64") else "arm64"
    deb_root = stage_root / "deb_root"
    opt_dir = deb_root / "opt" / "mudaepowertools"
    bin_dir = deb_root / "usr" / "bin"
    apps_dir = deb_root / "usr" / "share" / "applications"
    pix_dir = deb_root / "usr" / "share" / "pixmaps"
    debian_dir = deb_root / "DEBIAN"

    for d in (opt_dir, bin_dir, apps_dir, pix_dir, debian_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_bin = opt_dir / "MudaePowerTools"
    shutil.copy2(binary_path, target_bin)
    target_bin.chmod(0o755)

    launcher = bin_dir / "mudaepowertools"
    launcher.write_text("#!/bin/sh\nexec /opt/mudaepowertools/MudaePowerTools \"$@\"\n", encoding="utf-8")
    launcher.chmod(0o755)

    icon_path = pix_dir / "mudaepowertools.png"
    generate_app_icon_png(icon_path, size=256)

    desktop_entry = f"""[Desktop Entry]
Name={APP_NAME}
Comment=AI Mudae Harem Scraper, OKLab Color Gradient & $sm Sorter
Exec=/usr/bin/mudaepowertools
Icon=mudaepowertools
Terminal=false
Type=Application
Categories=Utility;Graphics;
"""
    (apps_dir / "mudaepowertools.desktop").write_text(desktop_entry, encoding="utf-8")

    installed_kb = max(1024, binary_path.stat().st_size // 1024)
    control_text = f"""Package: mudaepowertools
Version: {version}
Section: utils
Priority: optional
Architecture: {deb_arch}
Installed-Size: {installed_kb}
Maintainer: {AUTHOR} <noreply@github.com>
Depends: libxcb-cursor0 | libxcb-icccm4, libgl1
Description: {APP_NAME}
 Standalone desktop app for capturing, visually sorting (OKLab 2-Opt TSP,
 3D Hilbert, HDBSCAN/FAISS), and exporting Discord Mudae $sm/$smp commands.
"""
    (debian_dir / "control").write_text(control_text, encoding="utf-8")

    # Universal tarball installer tree
    tar_root = stage_root / "tar_root"
    tar_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary_path, tar_root / "MudaePowerTools")
    (tar_root / "MudaePowerTools").chmod(0o755)
    shutil.copy2(icon_path, tar_root / "mudaepowertools.png")
    (tar_root / "mudaepowertools.desktop").write_text(desktop_entry, encoding="utf-8")

    install_sh = """#!/usr/bin/env bash
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
APP_DIR="$PREFIX/share/applications"
ICON_DIR="$PREFIX/share/pixmaps"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$BIN_DIR" "$APP_DIR" "$ICON_DIR"
install -m 755 "$SCRIPT_DIR/MudaePowerTools" "$BIN_DIR/mudaepowertools"
install -m 644 "$SCRIPT_DIR/mudaepowertools.png" "$ICON_DIR/mudaepowertools.png"
sed "s|Exec=/usr/bin/mudaepowertools|Exec=$BIN_DIR/mudaepowertools|g" \
    "$SCRIPT_DIR/mudaepowertools.desktop" > "$APP_DIR/mudaepowertools.desktop"
chmod 644 "$APP_DIR/mudaepowertools.desktop"

echo "Installed Mudae Power Tools Pro to $BIN_DIR/mudaepowertools"
"""
    uninstall_sh = """#!/usr/bin/env bash
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local}"
rm -f "$PREFIX/bin/mudaepowertools" \
      "$PREFIX/share/applications/mudaepowertools.desktop" \
      "$PREFIX/share/pixmaps/mudaepowertools.png"
echo "Uninstalled Mudae Power Tools Pro from $PREFIX"
"""
    (tar_root / "install.sh").write_text(install_sh, encoding="utf-8")
    (tar_root / "uninstall.sh").write_text(uninstall_sh, encoding="utf-8")
    (tar_root / "install.sh").chmod(0o755)
    (tar_root / "uninstall.sh").chmod(0o755)

    return deb_root, tar_root


def build_linux_installers(
    binary_path: Path,
    dist_dir: Path,
    version: str = APP_VERSION,
    arch: str = "x64",
) -> List[Path]:
    """Build `.deb` package (via `dpkg-deb`) and universal `.tar.gz` installer on Linux."""
    if not binary_path.is_file():
        raise FileNotFoundError(f"Linux binary not found: {binary_path}")

    dist_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    deb_arch = "amd64" if arch in ("x64", "amd64", "x86_64") else "arm64"

    with tempfile.TemporaryDirectory() as tmp:
        deb_root, tar_root = stage_linux_package_tree(binary_path, Path(tmp), version=version, arch=arch)

        if shutil.which("dpkg-deb"):
            deb_path = dist_dir / f"mudaepowertools_{version}_{deb_arch}.deb"
            if deb_path.exists():
                deb_path.unlink()
            subprocess.run(
                ["dpkg-deb", "--build", "--root-owner-group", str(deb_root), str(deb_path)],
                check=True,
            )
            outputs.append(deb_path)

        tar_path = dist_dir / f"MudaePowerTools-{version}-Linux-{arch}-installer.tar.gz"
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as tf:
            for item in sorted(tar_root.rglob("*")):
                arcname = f"MudaePowerTools-{version}-Linux-{arch}/{item.relative_to(tar_root)}"
                tf.add(item, arcname=arcname)
        outputs.append(tar_path)

    return outputs


def build_for_current_os(binary_path: Optional[Path] = None, dist_dir: Optional[Path] = None) -> List[Path]:
    root = Path(__file__).resolve().parent
    dist = dist_dir or (root / "dist")
    os_label, arch_label = detect_os_arch()
    ext = ".exe" if os_label == "Windows" else ""
    bin_path = binary_path or (dist / f"MudaePowerTools-{os_label}-{arch_label}{ext}")

    if os_label == "Windows":
        return [build_windows_msi(bin_path, dist, version=APP_VERSION, arch=arch_label)]
    if os_label == "macOS":
        return build_macos_installers(bin_path, dist, version=APP_VERSION, arch=arch_label)
    if os_label == "Linux":
        return build_linux_installers(bin_path, dist, version=APP_VERSION, arch=arch_label)
    raise RuntimeError(f"Unsupported OS for installer generation: {os_label}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build native OS installer for Mudae Power Tools Pro.")
    parser.add_argument("--binary", type=Path, default=None, help="Path to compiled standalone binary.")
    parser.add_argument("--dist-dir", type=Path, default=None, help="Destination dist directory.")
    opts = parser.parse_args()

    built = build_for_current_os(binary_path=opts.binary, dist_dir=opts.dist_dir)
    for item in built:
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"Built installer: {item} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
