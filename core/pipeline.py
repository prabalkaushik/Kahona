"""
Phase 1 baseline pipeline: mic -> VAD -> ASR -> LLM (streamed) -> TTS
(streamed) -> speaker, with fixed silence-timeout turn-taking.

This is the regression target for every later phase. Once Phase 3's
turn-taking classifier exists, it plugs in by replacing StreamingVAD's
end_of_turn logic — the rest of this orchestration loop is designed to
stay the same.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from core.asr import ASR
from core.audio_io import MicStream, play_audio
from core.config import Settings
from core.llm import ConversationHistory, create_llm
from core.tts import TTS
from core.vad import StreamingVAD

logger = logging.getLogger("voice_agent.pipeline")


class VoiceAgentPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        logging.basicConfig(level=settings.logging.level)

        logger.info("Loading VAD...")
        self.vad = StreamingVAD(
            sample_rate=settings.vad.sample_rate,
            frame_ms=settings.vad.frame_ms,
            threshold=settings.vad.threshold,
            silence_timeout_ms=settings.vad.silence_timeout_ms,
            min_speech_ms=settings.vad.min_speech_ms,
        )

        logger.info("Loading ASR model (%s)...", settings.asr.model_size)
        self.asr = ASR(settings.asr)

        logger.info("Loading LLM backend (%s)...", settings.llm.backend)
        self.llm = create_llm(settings.llm)

        logger.info("Loading TTS voice (%s)...", settings.tts.voice)
        self.tts = TTS(settings.tts)

        self.history = ConversationHistory()

    def _handle_turn(self, speech_audio: np.ndarray) -> None:
        """Runs one full ASR -> LLM -> TTS cycle for a captured speech segment."""
        turn_t0 = time.monotonic()

        asr_result = self.asr.transcribe(speech_audio, self.settings.vad.sample_rate)
        if not asr_result.text:
            logger.info("(heard silence/noise, ignoring)")
            return

        logger.info("User: %s  [asr %.0fms]", asr_result.text, asr_result.latency_s * 1000)

        first_audio_t: float | None = None
        for sentence in self.llm.stream_sentences(self.history, asr_result.text):
            synth = self.tts.synthesize(sentence)
            if first_audio_t is None:
                first_audio_t = time.monotonic() - turn_t0
                logger.info("[time to first audio: %.0fms]", first_audio_t * 1000)
            logger.info("Agent: %s  [tts %.0fms]", sentence, synth.latency_s * 1000)
            play_audio(synth.audio, synth.sample_rate, self.settings.audio_io)

        total_t = time.monotonic() - turn_t0
        if self.settings.logging.log_latency_breakdown:
            logger.info("[turn complete in %.0fms total]", total_t * 1000)

    def run(self) -> None:
        logger.info("Listening... (Ctrl+C to stop)")
        speech_buffer: list[np.ndarray] = []

        with MicStream(
            sample_rate=self.settings.vad.sample_rate,
            frame_samples=self.vad.frame_samples,
            config=self.settings.audio_io,
        ) as mic:
            for frame in mic.frames():
                event = self.vad.process_frame(frame)

                if self.vad.in_speech:
                    speech_buffer.append(frame)

                if event is None:
                    continue

                if event.kind == "speech_start":
                    logger.info("Speech started...")
                elif event.kind == "end_of_turn":
                    logger.info("End of turn detected. Processing...")
                    if speech_buffer:
                        full_audio = np.concatenate(speech_buffer)
                        speech_buffer = []
                        self._handle_turn(full_audio)
                    logger.info("Listening...")
