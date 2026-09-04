#!/usr/bin/env python3
"""
Downloads the quantized local LLM used by configs/local_gpu.yaml:
Qwen2.5-3B-Instruct at Q4_K_M (~1.93GB), from Qwen's official GGUF repo.
Sized to comfortably fit a 4GB-VRAM GPU (e.g. GTX 1650) alongside
faster-whisper base.en.

Run once, only if you're using configs/local_gpu.yaml:
    python scripts/download_local_llm.py
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}"
OUT_DIR = Path("models/llm")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / FILENAME

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print(f"[download] {URL}")
    print("This is ~1.93GB, may take a few minutes depending on your connection.")
    urlretrieve(URL, out_path)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
