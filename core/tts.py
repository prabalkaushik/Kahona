"""
TTS wrapper around Piper's standalone binary (invoked via subprocess).

We deliberately shell out to the binary rather than using the `piper-tts`
pip package: that package depends on `piper-phonemize`, a compiled
extension with very inconsistent prebuilt-wheel availability across
Python versions and platforms. The standalone binary (bundles espeak-ng)
is far more reliable to get running on a fresh machine — see
scripts/download_piper_voice.py.

Called once per sentence chunk yielded by core.llm.StreamingLLM, so audio
for sentence 1 can start playing while sentence 2 is still being generated
by the LLM.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import TTSConfig

DEFAULT_BIN_DIR = Path("bin")


@dataclass
class SynthResult:
    audio: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int
    latency_s: float


class TTS:
    def __init__(self, config: TTSConfig, bin_dir: Path = DEFAULT_BIN_DIR):
        self.config = config
        exe_name = "piper.exe" if platform.system() == "Windows" else "piper"
        self.exe_path = bin_dir / exe_name

        voice_dir = Path(config.voice_dir)
        self.onnx_path = voice_dir / f"{config.voice}.onnx"
        self.json_path = voice_dir / f"{config.voice}.onnx.json"

        if not self.exe_path.exists():
            raise FileNotFoundError(
                f"Piper binary not found at {self.exe_path}. "
                f"Run `python scripts/download_piper_voice.py` first."
            )
        if not self.onnx_path.exists() or not self.json_path.exists():
            raise FileNotFoundError(
                f"Piper voice files not found in {voice_dir}. "
                f"Run `python scripts/download_piper_voice.py` first."
            )

        # actual sample rate comes from the voice's own config, not our YAML
        with open(self.json_path) as f:
            voice_meta = json.load(f)
        self.sample_rate = voice_meta.get("audio", {}).get(
            "sample_rate", config.sample_rate
        )

    def synthesize(self, text: str) -> SynthResult:
        t0 = time.monotonic()

        # --output-raw streams raw 16-bit PCM mono audio to stdout, at the
        # voice's native sample rate. No temp files, no WAV header parsing.
        result = subprocess.run(
            [
                str(self.exe_path),
                "--model", str(self.onnx_path),
                "--output-raw",
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Piper synthesis failed (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace')}"
            )

        audio_int16 = np.frombuffer(result.stdout, dtype=np.int16)
        audio = audio_int16.astype(np.float32) / 32768.0

        latency = time.monotonic() - t0
        return SynthResult(audio=audio, sample_rate=self.sample_rate, latency_s=latency)
