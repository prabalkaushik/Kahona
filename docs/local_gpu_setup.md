# GPU Setup (configs/gpu.yaml and configs/local_gpu.yaml)

Tailored for your machine: Windows, GTX 1650 (4GB VRAM). Two things need
GPU support — `faster-whisper` (ASR) and, only if you're using
`configs/local_gpu.yaml`, `llama-cpp-python` (local LLM). This is the
fiddliest part of the whole project, mostly because of driver/toolkit
version-matching, not because of anything in our code — so read the notes
below before you file a bug against yourself at 1am.

## 1. NVIDIA driver + CUDA toolkit

Check your driver first:
```powershell
nvidia-smi
```
This shows your driver version and the max CUDA version it supports. If
this command doesn't work at all, install the latest GeForce driver from
nvidia.com first — a GTX 1650 fully supports CUDA 12.x with any
reasonably recent driver.

`ctranslate2` (which `faster-whisper` uses) currently requires **CUDA 12
+ cuDNN 9**. You do not necessarily need to install the full CUDA
Toolkit — the pip-installable NVIDIA runtime packages are enough and are
far less hassle on Windows:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If `faster-whisper` still can't find `cublas64_12.dll` or similar at
runtime (a known Windows-specific issue), find where pip installed those
packages (typically `<venv>\Lib\site-packages\nvidia\cublas\bin` and
`...\nvidia\cudnn\bin`) and add both folders to your `PATH` for the
session:

```powershell
$env:PATH += ";<venv>\Lib\site-packages\nvidia\cublas\bin;<venv>\Lib\site-packages\nvidia\cudnn\bin"
```

Add this to your shell profile (or a `.ps1` activation script) if you
don't want to repeat it every session.

## 2. faster-whisper on GPU (configs/gpu.yaml, configs/local_gpu.yaml)

# If you're only using configs/fast.yaml, configs/balanced.yaml, or
# configs/gpu.yaml (OpenAI LLM), you don't need this at all.
No code or dependency changes beyond step 1 — `core/asr.py` already reads
`device`/`compute_type` from the YAML. Just run with:

```powershell
python run_local.py --config configs/gpu.yaml
```

If it falls back to CPU silently or throws a DLL error, it's almost
always the PATH issue in step 1.

## 3. llama-cpp-python on GPU (configs/local_gpu.yaml only)

Skip this entirely if you're sticking with `configs/gpu.yaml` (OpenAI
handles the LLM there — no local LLM, no build headaches).

**The honest state of things:** upstream `llama-cpp-python` paused
publishing prebuilt Windows CUDA wheels in mid-2025 after a batch shipped
that crashed at runtime. As of now you have two realistic options:

**Option A — try the prebuilt wheel index first (fastest if it works):**
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```
Match `cu121`/`cu122`/etc. to your installed CUDA version. If this
installs cleanly and `from llama_cpp import Llama` doesn't error, you're
done. If inference silently runs on CPU (check `nvidia-smi` while it's
generating — GPU usage should spike), the wheel didn't actually pick up
CUDA; move to Option B.

**Option B — build from source (slower to set up, more reliable):**

# OpenAI API key (see .env.example).
Requirements: Visual Studio Build Tools (C++ workload) and the CUDA
Toolkit (not just the pip runtime packages — the compiler needs the full
toolkit headers this time) installed and on PATH.

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir
```
This compiles locally and takes several minutes, but avoids the
prebuilt-wheel reliability problems entirely.

**If both options are too much friction:** `llama-cpp-python` still works
fine on CPU with zero extra setup (just `pip install llama-cpp-python`,
no CMAKE_ARGS). A 3B Q4_K_M model on CPU will be slow for a "low-latency"
demo, but it's a legitimate fallback while you get CUDA sorted, and it
proves the pipeline logic works before you fight driver issues.

## 4. Download the model files

```powershell
python scripts/download_piper_voice.py       # if not already done
python scripts/download_local_llm.py          # only for configs/local_gpu.yaml
```

## 5. Run it

```powershell
python run_local.py --config configs/gpu.yaml         # GPU ASR + OpenAI LLM
python run_local.py --config configs/local_gpu.yaml     # fully offline
```

## VRAM budget check (4GB card)

| Component | Approx. VRAM |
|---|---|
| Whisper `base.en` (float16) | ~150-300MB |
| Qwen2.5-3B-Instruct Q4_K_M (local_gpu only) | ~2.0-2.3GB |
| CUDA context/runtime overhead | ~200-400MB |
| **Total (local_gpu.yaml)** | **~2.5-3GB** — comfortable headroom on 4GB |
| **Total (gpu.yaml, no local LLM)** | **~400-700MB** — plenty of room |

If you ever want to size up (e.g. Whisper `small.en` or a 7B local
model), `configs/gpu.yaml`'s hybrid approach (GPU ASR + cloud LLM) keeps
the most VRAM free, since the LLM isn't competing for it at all.
