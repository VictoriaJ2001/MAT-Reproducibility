#!/usr/bin/env python3
"""
Download Llama-3-8B-Instruct from HuggingFace mirror to local models/ directory.

Usage (on server):
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/download_model.py

The model will be saved to: ./models/Llama-3-8B-Instruct/
"""

import argparse
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_NAME, get_model_path, ensure_dirs


def download_from_hub(model_id, local_dir):
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id} ...")
    print(f"Target: {local_dir}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")

    tmp_dir = local_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    snapshot_download(
        repo_id=model_id,
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.pth", "*.bin", "*flax*", "*tf*", "*onnx*"],
    )

    os.rename(tmp_dir, local_dir)
    print(f"Model saved to: {local_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="HF model ID (default: config MODEL_NAME mapping)")
    parser.add_argument("--output", type=str, default=None,
                        help="Local target dir (default: config get_model_path())")
    args = parser.parse_args()

    ensure_dirs()
    model_id = args.model or "meta-llama/Meta-Llama-3-8B-Instruct"
    local_path = args.output or get_model_path(MODEL_NAME)

    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"Model already exists at {local_path}. Delete it first to re-download.")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    download_from_hub(model_id, local_path)

    config_json = os.path.join(local_path, "config.json")
    if os.path.exists(config_json):
        print("Verification: config.json found ✓")
    else:
        print("WARNING: config.json not found. Download may be incomplete.")


if __name__ == "__main__":
    main()
