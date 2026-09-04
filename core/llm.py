"""
Streaming LLM layer with two interchangeable backends:

  - OpenAILLM:      cloud inference via OpenAI's API. Default. Best latency/
                   quality, but needs an API key and internet.
  - LocalGPULLM:  fully offline, runs a quantized GGUF model on your GPU
                   via llama-cpp-python (CUDA build). No API key, no
                   internet dependency — trades some quality/latency for
                   that independence. Sized for a 4GB-VRAM card (e.g.
                   GTX 1650) using Qwen2.5-3B-Instruct at Q4_K_M.

Both backends expose the same `stream_sentences()` generator interface,
so core/pipeline.py doesn't need to know or care which one is active —
that's decided entirely by configs/*.yaml (llm.backend: "openai" |
"local_gpu").

The key latency trick lives in the shared chunker: instead of waiting for
the full LLM response before synthesizing speech, we buffer streamed
tokens and yield complete sentences (or clause-level chunks on commas,
for long sentences) as soon as they're ready. The TTS stage can start
speaking sentence 1 while the LLM is still generating sentence 3.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Generator, Iterable, Protocol

from core.config import LLMConfig

# Split on sentence-ending punctuation or commas followed by a space.
# This yields chunks faster, so TTS starts speaking the first clause
# even earlier.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?,])\s+")


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ConversationHistory:
    turns: list[ChatTurn] = field(default_factory=list)
    max_turns: int = 20  # keep context bounded

    def add(self, role: str, content: str) -> None:
        self.turns.append(ChatTurn(role, content))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def to_messages(self, system_prompt: str) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": t.role, "content": t.content} for t in self.turns]
        return messages


def _chunk_into_sentences(
    token_deltas: Iterable[str], history: ConversationHistory
) -> Generator[str, None, None]:
    """
    Shared by both backends: consumes a stream of text deltas (however the
    backend produces them) and yields complete sentences as soon as they're
    available, so TTS can start on sentence 1 before the LLM finishes
    generating the rest.
    """
    buffer = ""
    full_response = ""

    for delta in token_deltas:
        if not delta:
            continue
        buffer += delta
        full_response += delta

        parts = _SENTENCE_END_RE.split(buffer)
        if len(parts) > 1:
            for complete_sentence in parts[:-1]:
                if complete_sentence.strip():
                    yield complete_sentence.strip()
            buffer = parts[-1]

    if buffer.strip():
        yield buffer.strip()

    history.add("assistant", full_response.strip())


class LLMBackend(Protocol):
    def stream_sentences(
        self, history: ConversationHistory, user_text: str
    ) -> Generator[str, None, None]: ...


class OpenAILLM:
    """Cloud inference via OpenAI. Default backend."""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI  # local import: keeps this optional at import time

        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    def stream_sentences(
        self, history: ConversationHistory, user_text: str
    ) -> Generator[str, None, None]:
        history.add("user", user_text)
        messages = history.to_messages(self.config.system_prompt)

        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True,
        )

        def deltas():
            for chunk in stream:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        yield from _chunk_into_sentences(deltas(), history)


class LocalGPULLM:
    """Fully offline inference via llama.cpp (CUDA build), for a quantized
    GGUF model sized to fit your GPU's VRAM. No API key, no internet.

    Requires llama-cpp-python built with CUDA support — see
    docs/local_gpu_setup.md for the install command, since the standard
    `pip install llama-cpp-python` builds CPU-only wheels.
    """

    def __init__(self, config: LLMConfig):
        from llama_cpp import Llama  # local import: optional dependency

        self.config = config
        try:
            self.llm = Llama(
                model_path=config.model_path,
                n_gpu_layers=config.n_gpu_layers,
                n_ctx=config.context_length,
                verbose=False,
            )
        except ValueError as e:
            raise FileNotFoundError(
                f"Could not load local LLM model at '{config.model_path}'. "
                f"Run `python scripts/download_local_llm.py` first. "
                f"Original error: {e}"
            ) from e

    def stream_sentences(
        self, history: ConversationHistory, user_text: str
    ) -> Generator[str, None, None]:
        history.add("user", user_text)
        messages = history.to_messages(self.config.system_prompt)

        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True,
        )

        def deltas():
            for chunk in stream:
                # llama-cpp-python mirrors the OpenAI delta shape
                yield chunk["choices"][0]["delta"].get("content")

        yield from _chunk_into_sentences(deltas(), history)


def create_llm(config: LLMConfig) -> LLMBackend:
    """Factory: picks the backend based on config.llm.backend, so
    core/pipeline.py stays agnostic to which one is active."""
    if config.backend == "openai":
        return OpenAILLM(config)
    if config.backend == "local_gpu":
        return LocalGPULLM(config)
    raise ValueError(
        f"Unknown llm.backend '{config.backend}'. Expected 'openai' or 'local_gpu'."
    )
