"""
Loads a YAML config tier (configs/fast.yaml, configs/balanced.yaml, ...)
plus environment variables (.env) into a single typed settings object.

Usage:
    from core.config import load_settings
    settings = load_settings("configs/fast.yaml")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class VADConfig(BaseModel):
    model: str = "silero_vad"
    sample_rate: int = 16000
    frame_ms: int = 32
    threshold: float = 0.5
    silence_timeout_ms: int = 700
    min_speech_ms: int = 250


class ASRConfig(BaseModel):
    backend: str = "faster_whisper"
    model_size: str = "tiny.en"
    device: str = "cpu"           # "cpu" or "cuda"
    compute_type: str = "int8"     # cpu: "int8". cuda: "float16" or "int8_float16"
    beam_size: int = 1
    language: str = "en"


class LLMConfig(BaseModel):
    backend: str = "openai"           # "openai" or "local_gpu"
    model: str = "gpt-4o-mini"     # openai model name (ignored for local_gpu)
    max_tokens: int = 300
    temperature: float = 0.7
    stream: bool = True
    system_prompt: str = "You are a helpful voice assistant."
    api_key: Optional[str] = None  # populated from env, not YAML (openai only)

    # --- local_gpu backend only ---
    model_path: str = "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
    n_gpu_layers: int = -1          # -1 = offload all layers to GPU
    context_length: int = 4096


class TTSConfig(BaseModel):
    backend: str = "piper"
    voice: str = "en_US-lessac-medium"
    voice_dir: str = "models/piper"
    sentence_chunking: bool = True
    sample_rate: int = 22050


class AudioIOConfig(BaseModel):
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    channels: int = 1


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_latency_breakdown: bool = True


class Settings(BaseModel):
    vad: VADConfig = VADConfig()
    asr: ASRConfig = ASRConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    audio_io: AudioIOConfig = AudioIOConfig()
    logging: LoggingConfig = LoggingConfig()


def load_settings(config_path: str | Path = "configs/fast.yaml") -> Settings:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Run from the project root, or pass --config with a valid path."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    settings = Settings(**raw)

    # secrets come from env, never from YAML — only needed for the openai backend
    if settings.llm.backend == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_key_here":
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
                "key from https://platform.openai.com/ — or switch to "
                "configs/local_gpu.yaml to run fully offline instead."
            )
        settings.llm.api_key = openai_key

    return settings
