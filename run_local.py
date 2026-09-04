#!/usr/bin/env python3
"""
Phase 1 entry point: run the voice agent locally using your mic and speakers.

Usage:
    python run_local.py                       # uses configs/fast.yaml
    python run_local.py --config configs/balanced.yaml
"""
from __future__ import annotations

import argparse
import sys

from core.config import load_settings
from core.pipeline import VoiceAgentPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the voice agent locally.")
    parser.add_argument(
        "--config", default="configs/fast.yaml",
        help="Path to a config YAML (default: configs/fast.yaml)",
    )
    args = parser.parse_args()

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Setup error: {e}", file=sys.stderr)
        return 1

    pipeline = VoiceAgentPipeline(settings)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
