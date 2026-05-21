#!/usr/bin/env python3
"""
Download Llama-3-8B-Instruct from ModelScope (no auth required).

Usage:
    pip install modelscope
    python scripts/download_model_ms.py

Target: ./models/Llama-3-8B-Instruct/
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_model_path, MODEL_NAME, ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="LLM-Research/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from modelscope.hub.snapshot_download import snapshot_download

    ensure_dirs()
    local_path = args.output or get_model_path(MODEL_NAME)

    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        print(f"Model already exists at {local_path}.")
        return

    os.makedirs(local_path, exist_ok=True)

    print(f"Downloading {args.model} via ModelScope...")
    print(f"Target: {local_path}")

    downloaded = snapshot_download(model_id=args.model, revision="master")

    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        print(f"Copying from {downloaded} to {local_path}...")
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        shutil.copytree(downloaded, local_path, dirs_exist_ok=True)

    print(f"Model ready at: {local_path}")

    if os.path.exists(os.path.join(local_path, "config.json")):
        print("Verification: config.json found ✓")
    else:
        print("WARNING: config.json not found. Contents:")
        for f in sorted(os.listdir(local_path))[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
