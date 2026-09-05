# Kahona — Ultra Low-Latency Production-Grade Speech-to-Speech Pipeline (Phase 1)

Welcome to the foundation of Kahona, a cutting-edge, production-grade speech-to-speech conversational agent. We are building a mesmerizingly fast, incredibly lightweight voice assistant designed to run seamlessly even on modest hardware without compromising on quality. 

Currently in **Phase 1 (Baseline)**, this pipeline establishes a robust, highly modular core. While you will already experience impressive speeds today, **the absolute "least latency" and true conversational fluidity will be achieved by Phase 4**, where I introduce advanced paralinguistic-aware prompting and full-duplex backchanneling.

## Why this architecture

On an CPU-only machine, running ASR + a capable LLM + TTS all
locally at once is not realistic — a usable local LLM alone needs several GB
even quantized, leaving nothing for the audio models or the OS.

So Phase 1 makes one deliberate split:

- **Local, CPU, ONNX/CTranslate2-based (no PyTorch)**: VAD, ASR, TTS.
  These are all small enough (tens to a few hundred MB) to run comfortably
  alongside each other.
- **Remote, via OpenAI API**: the LLM. OpenAI's inference is robust and
  can be very fast, compensating for the network hop.

This keeps your laptop's RAM budget free for later phases (the turn-taking
classifier in Phase 3 needs headroom).

## Phase 1 scope

This phase proves out the **core loop**: mic in → VAD → ASR → LLM (streamed)
→ TTS (streamed) → speaker out, using simple silence-based turn-taking.
This is your regression baseline — Phase 2 (latency engineering) and
Phase 3 (the semantic turn-taking classifier) will be measured against it.

No WebRTC/server yet — that's Phase 5 (packaging). Phase 1 runs locally via
your mic and speakers so you can iterate fast.

## Setup

### 1. System dependencies

```bash
# Linux
sudo apt-get install portaudio19-dev

# macOS
brew install portaudio
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get an OpenAI API key

Sign up at https://platform.openai.com/ and create an API key. Then:

```bash
cp .env.example .env
# edit .env and paste your key into OPENAI_API_KEY
```

### 4. Download the Piper TTS voice

```bash
python scripts/download_piper_voice.py
```

This grabs a small (~60MB) English voice model into `models/piper/`.
The Silero VAD ONNX model and faster-whisper `tiny.en` weights download
automatically on first run (cached locally after that).

### 5. Run it

```bash
python run_local.py
```

Speak into your mic. When you stop talking (configurable silence threshold
in `configs/fast.yaml`), the agent transcribes, sends it to the LLM, and
streams the spoken response back — sentence by sentence, so you hear the
start of the reply before the LLM has finished generating.

## Config tiers

Model choices are config-driven (`configs/*.yaml`), not hardcoded:

- `configs/fast.yaml` — CPU-only, smallest models, lowest latency. This is
  what `run_local.py` uses unless you pass `--config`. Works on any
  machine, no GPU required.
- `configs/balanced.yaml` — CPU-only, slightly larger ASR model
  (`base.en`) for better accuracy if you have more headroom.
- `configs/gpu.yaml` — for machines with a CUDA GPU (tested target:
  GTX 1650, 4GB VRAM). ASR runs on GPU (faster + more accurate), LLM stays
  on OpenAI's cloud. Best latency/quality combo if you have a GPU and don't
  mind the cloud dependency. See `docs/local_gpu_setup.md` for setup.
- `configs/local_gpu.yaml` — fully offline: ASR *and* LLM both run on your
  GPU (Qwen2.5-3B-Instruct, quantized to fit 4GB VRAM). No API key, no
  internet needed at runtime. Trades some LLM quality/latency for zero
  cloud dependency. See `docs/local_gpu_setup.md` — this one needs a bit
  more setup (CUDA + a GPU-enabled `llama-cpp-python` build).

Swap models by editing the YAML, not the code.

## Project layout

```
voice-agent/
├── core/
│   ├── vad.py           # Silero VAD (ONNX), streaming speech/silence detection
│   ├── asr.py            # faster-whisper streaming transcription
│   ├── llm.py             # OpenAI streaming client + sentence-boundary chunker
│   ├── tts.py              # Piper streaming synthesis
│   ├── audio_io.py          # mic capture / speaker playback (sounddevice)
│   └── pipeline.py           # orchestrates VAD -> ASR -> LLM -> TTS
├── configs/                    # model/latency config tiers
├── scripts/
│   └── download_piper_voice.py
├── run_local.py                  # Phase 1 entry point (mic -> speaker loop)
├── benchmarks/                     # latency measurement scripts (Phase 2)
└── requirements.txt
```

## What's next (roadmap)

- **Phase 2**: barge-in / interruption handling, concurrent partial-ASR
  decoding, latency benchmarking harness with published numbers per config.
- **Phase 3**: semantic + prosodic turn-taking classifier — the flagship
  feature. Replaces the silence-timeout in `core/vad.py` with a model that
  predicts "is the user actually done" from partial transcript completeness
  + pitch/energy trends, not just silence duration.
- **Phase 4**: backchannel injection (full-duplex "mm-hmm" acks) and
  paralinguistic-aware LLM prompting.
- **Phase 5**: WebRTC server, Docker image, pip package, client SDKs —
  making this usable by others, not just you.

## Known limitations of this phase (by design)

- Turn-taking is silence-based (fixed timeout) — this is what Phase 3 fixes.
- No barge-in: if you talk while the agent is speaking, it won't stop
  (Phase 2).
- Single-user, local mic only — no server/remote clients yet.
