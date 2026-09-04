"""
ASR wrapper around faster-whisper (CTranslate2 backend — no torch, fast
on CPU with int8 quantization).

Phase 1 usage pattern: VAD hands us a complete speech segment (start to
end_of_turn), we transcribe it in one shot. This is simpler and more
accurate than incremental partial decoding, which we can add in Phase 2
once the baseline is solid and we have latency numbers to compare against.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from core.config import ASRConfig


@dataclass
class TranscriptionResult:
    text: str
    latency_s: float
    language: str


class ASR:
    def __init__(self, config: ASRConfig):
        self.config = config
        # device/compute_type come from config so the same code runs the
        # CPU-only tier (device=cpu, compute_type=int8) and the GPU tier
        # (device=cuda, compute_type=float16 or int8_float16) without
        # any code changes — just swap the YAML.
        self.model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """
        audio: float32 numpy array, mono, in [-1, 1], at `sample_rate` Hz.
        """
        t0 = time.monotonic()
        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,  # we already did VAD upstream
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        latency = time.monotonic() - t0
        return TranscriptionResult(text=text, latency_s=latency, language=info.language)
