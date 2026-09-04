#!/usr/bin/env python3
"""
Downloads:
  1. The Piper standalone binary for your platform (bundles espeak-ng,
     no compiled Python extension needed — avoids the piper-phonemize
     wheel-availability problems that plague `pip install piper-tts`
     on many platforms/Python versions).
  2. The default voice model (en_US-lessac-medium, ~60MB) used by
     configs/fast.yaml and configs/balanced.yaml.

Run once after installing requirements:
    python scripts/download_piper_voice.py
"""
from __future__ import annotations

import platform
import shutil
import stat
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PIPER_RELEASE_TAG = "2023.11.14-2"
PIPER_RELEASE_BASE = f"https://github.com/rhasspy/piper/releases/download/{PIPER_RELEASE_TAG}"

VOICE = "en_US-lessac-medium"
VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
)
VOICE_FILES = [f"{VOICE}.onnx", f"{VOICE}.onnx.json"]

BIN_DIR = Path("bin")
VOICE_DIR = Path("models/piper")


def _detect_asset() -> tuple[str, str]:
    """Returns (asset_filename, archive_kind) for this machine."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "piper_linux_x86_64.tar.gz", "tar"
        if machine in ("aarch64", "arm64"):
            return "piper_linux_aarch64.tar.gz", "tar"
        if machine.startswith("arm"):
            return "piper_linux_armv7l.tar.gz", "tar"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            # Known upstream issue: the arm64 tarball actually contains an
            # x86_64 binary. It still runs fine on Apple Silicon via
            # Rosetta 2 (already installed on most Macs by default).
            return "piper_macos_aarch64.tar.gz", "tar"
        return "piper_macos_x64.tar.gz", "tar"
    elif system == "Windows":
        return "piper_windows_amd64.zip", "zip"

    raise RuntimeError(
        f"Could not determine a Piper binary for system={system}, "
        f"machine={machine}. Download manually from "
        f"https://github.com/rhasspy/piper/releases/tag/{PIPER_RELEASE_TAG} "
        f"and place the extracted 'piper' executable in ./bin/"
    )


def download_binary() -> None:
    exe_name = "piper.exe" if platform.system() == "Windows" else "piper"
    exe_path = BIN_DIR / exe_name
    if exe_path.exists():
        print(f"[skip] {exe_path} already exists")
        return

    asset_name, archive_kind = _detect_asset()
    url = f"{PIPER_RELEASE_BASE}/{asset_name}"
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = BIN_DIR / asset_name

    print(f"[download] {url}")
    urlretrieve(url, archive_path)

    print(f"[extract] {archive_path}")
    extract_dir = BIN_DIR / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    if archive_kind == "tar":
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)
    else:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)

    # the archive contains a `piper/` folder with the executable + libs
    extracted_piper_dir = extract_dir / "piper"
    for item in extracted_piper_dir.iterdir():
        shutil.move(str(item), str(BIN_DIR / item.name))

    if platform.system() != "Windows":
        exe_path.chmod(exe_path.stat().st_mode | stat.S_IEXEC)

    shutil.rmtree(extract_dir)
    archive_path.unlink()
    print(f"[done] Piper binary at {exe_path}")


def download_voice() -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in VOICE_FILES:
        out_path = VOICE_DIR / fname
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue
        url = f"{VOICE_BASE_URL}/{fname}"
        print(f"[download] {url} -> {out_path}")
        urlretrieve(url, out_path)


def main() -> int:
    try:
        download_binary()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    download_voice()
    print("\nDone. Piper binary is in ./bin/, voice files are in models/piper/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
