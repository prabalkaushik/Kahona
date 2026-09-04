"""
Microphone capture and speaker playback.

Phase 1: simple blocking-callback capture that feeds fixed-size frames to
the VAD, and a straightforward playback function for synthesized audio.
Phase 2 will make playback interruptible (barge-in support).
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from core.config import AudioIOConfig


class MicStream:
    """Pulls fixed-size float32 frames from the microphone via a queue."""

    def __init__(self, sample_rate: int, frame_samples: int, config: AudioIOConfig):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=config.channels,
            dtype="float32",
            blocksize=frame_samples,
            device=config.input_device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            # xruns etc. — log but don't crash the stream
            print(f"[audio_io] mic status: {status}")
        self._q.put(indata.copy().reshape(-1))

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.stop()
        self._stream.close()

    def frames(self):
        """Generator yielding one frame (numpy array) at a time, blocking."""
        while True:
            yield self._q.get()


def play_audio(audio: np.ndarray, sample_rate: int, config: AudioIOConfig) -> None:
    """Blocking playback of a float32 mono audio array."""
    sd.play(audio, samplerate=sample_rate, device=config.output_device)
    sd.wait()
