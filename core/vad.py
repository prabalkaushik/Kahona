"""
Streaming Voice Activity Detection using Silero VAD (ONNX runtime, no torch).

Phase 1: exposes a simple state machine — feed it audio frames, it tells you
when speech starts and when it thinks the user is done talking (fixed
silence timeout). Phase 3 will replace `is_end_of_turn()` with a learned
classifier that also looks at partial-transcript completeness and prosody,
but the frame-feeding interface here stays the same so the rest of the
pipeline doesn't need to change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
DEFAULT_MODEL_PATH = Path("models/silero_vad.onnx")


def _ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[vad] downloading Silero VAD model to {path} ...")
        urlretrieve(SILERO_VAD_URL, path)
    return path


@dataclass
class VADEvent:
    kind: str  # "speech_start" | "speech_continue" | "end_of_turn"
    timestamp: float
    speech_prob: float


class StreamingVAD:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 32,
                 threshold: float = 0.5, silence_timeout_ms: int = 700,
                 min_speech_ms: int = 250, model_path: Path = DEFAULT_MODEL_PATH):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.threshold = threshold
        self.silence_timeout_s = silence_timeout_ms / 1000
        self.min_speech_s = min_speech_ms / 1000

        model_path = _ensure_model(model_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        # Silero VAD v5 keeps recurrent state between calls (2, 1, 128 float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(sample_rate, dtype=np.int64)

        self.in_speech = False
        self.speech_start_t: float | None = None
        self.last_speech_t: float | None = None

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.in_speech = False
        self.speech_start_t = None
        self.last_speech_t = None

    def _infer(self, frame: np.ndarray) -> float:
        """Run one frame through the ONNX model, return speech probability."""
        inp = frame.astype(np.float32)[None, :]
        ort_inputs = {"input": inp, "state": self._state, "sr": self._sr}
        out, new_state = self.session.run(None, ort_inputs)
        self._state = new_state
        return float(out.squeeze())

    def process_frame(self, frame: np.ndarray) -> VADEvent | None:
        """
        Feed one frame of audio (length == self.frame_samples, float32, [-1, 1]).
        Returns a VADEvent if something notable happened this frame, else None.
        """
        prob = self._infer(frame)
        now = time.monotonic()
        is_speech = prob >= self.threshold

        if is_speech:
            self.last_speech_t = now
            if not self.in_speech:
                self.in_speech = True
                self.speech_start_t = now
                return VADEvent("speech_start", now, prob)
            return VADEvent("speech_continue", now, prob)

        # not speech this frame
        if self.in_speech and self.last_speech_t is not None:
            silence_elapsed = now - self.last_speech_t
            speech_duration = (self.last_speech_t - self.speech_start_t
                                if self.speech_start_t else 0)
            if silence_elapsed >= self.silence_timeout_s:
                if speech_duration >= self.min_speech_s:
                    self.in_speech = False
                    event = VADEvent("end_of_turn", now, prob)
                    self.speech_start_t = None
                    self.last_speech_t = None
                    return event
                else:
                    # It was a short blip (noise/cough), reset and ignore
                    self.in_speech = False
                    self.speech_start_t = None
                    self.last_speech_t = None
                    # return None

        return None
