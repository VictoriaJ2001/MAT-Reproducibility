#!/usr/bin/env python3
"""
Download and prepare datasets for MAT-Steer reproducibility.

Downloads truthfulqa, toxigen, and bbq from HuggingFace Datasets
and saves them as JSON files in data/ in the format expected by the codebase.

Usage:
    python scripts/prepare_data.py              # all 3 datasets
    python scripts/prepare_data.py --dataset truthfulqa  # single dataset
    python scripts/prepare_data.py --output /path/to/data  # custom output dir
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import DATA_DIR, ensure_dirs


def prepare_truthfulqa(output_dir):
    from datasets import load_dataset

    print("Downloading TruthfulQA (HuggingFace: truthful_qa/multiple_choice)...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    data = []
    for item in dataset:
        data.append({
            "question": item["question"],
            "mc2_targets": {
                "choices": item["mc2_targets"]["choices"],
                "labels": item["mc2_targets"]["labels"],
            },
        })

    path = os.path.join(output_dir, "truthfulqa.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} questions → {path}")


def download_truthfulqa_csv(project_root):
    import urllib.request

    csv_dir = os.path.join(project_root, "TruthfulQA")
    csv_path = os.path.join(csv_dir, "TruthfulQA.csv")
    if os.path.exists(csv_path):
        print("  TruthfulQA.csv already exists, skipping.")
        return

    os.makedirs(csv_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    print(f"  Downloading TruthfulQA.csv from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    print(f"  Saved → {csv_path}")


def prepare_toxigen(output_dir):
    from datasets import load_dataset

    print("Downloading ToxiGen (HuggingFace: toxigen)...")
    dataset = load_dataset("skg/toxigen-data", "annotated", split="train")

    data = []
    for item in dataset:
        text = item.get("text", item.get("prompt", ""))
        label = int(item.get("toxicity_human", 0))
        if text:
            data.append({"text": text, "label": label})

    path = os.path.join(output_dir, "toxigen.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} samples → {path}")


def prepare_bbq(output_dir):
    import json
    import urllib.request

    print("Downloading BBQ from BIG-bench GitHub...")
    url = "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/bbq_lite_json/task.json"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        raw = json.loads(resp.read().decode())

    data = []
    count = 0
    for item in raw.get("examples", []):
        context = item.get("context_condition", "")
        prefix = item.get("question", "")
        target_scores = item.get("target_scores", {})
        choice_keys = list(target_scores.keys())

        if len(choice_keys) < 2 or not prefix:
            continue

        ans0 = choice_keys[0] if len(choice_keys) > 0 else ""
        ans1 = choice_keys[1] if len(choice_keys) > 1 else ""
        ans2 = choice_keys[2] if len(choice_keys) > 2 else ""

        # label = index of correct answer (target_scores value == 1)
        correct_idx = 0
        for i, k in enumerate(choice_keys):
            if target_scores.get(k, 0) == 1:
                correct_idx = i
                break

        full_question = f"{context} {prefix}".strip() if context else prefix
        data.append({
            "context": context,
            "question": full_question,
            "ans0": ans0,
            "ans1": ans1,
            "ans2": ans2,
            "label": correct_idx,
        })
        count += 1

    path = os.path.join(output_dir, "bbq.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} samples → {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MAT-Steer datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["truthfulqa", "toxigen", "bbq"],
                        help="Prepare a single dataset (default: all three)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: project/data/)")
    args = parser.parse_args()

    output_dir = args.output or DATA_DIR
    ensure_dirs()
    os.makedirs(output_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_truthfulqa_csv(project_root)

    PREPARERS = {
        "truthfulqa": prepare_truthfulqa,
        "toxigen": prepare_toxigen,
        "bbq": prepare_bbq,
    }

    if args.dataset:
        PREPARERS[args.dataset](output_dir)
    else:
        for name, fn in PREPARERS.items():
            print(f"\n{'='*50}")
            fn(output_dir)

    print(f"\nAll datasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
