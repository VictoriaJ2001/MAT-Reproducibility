#!/usr/bin/env python3
"""
Evaluate MAT-Steer checkpoint on TruthfulQA.

Evaluates both:
  1. Baseline (no intervention)
  2. MAT-Steer intervention at the specified layer

Requires:
  - TruthfulQA package: pip install git+https://github.com/sylinrl/TruthfulQA
  - OPENAI_API_KEY env var for GPT-judge / GPT-info metrics
"""

import argparse
import os
import sys
import json
import warnings

sys.path.append('..')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv

from interveners import MATIntervener, create_mat_pyvene_config
from utils import alt_tqa_evaluate, ENGINE_MAP
from config import (
    get_model_path, MODEL_NAME, INTERVENTION_LAYER,
    EVAL_MULTIPLIER, EVAL_INSTRUCTION_PROMPT, EVAL_DIR, ensure_dirs
)


def load_model_and_tokenizer(model_name, model_path=None):
    path = model_path or get_model_path(model_name)
    print(f"Loading model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_mat_intervener(checkpoint_path, multiplier=EVAL_MULTIPLIER):
    print(f"Loading MAT checkpoint: {checkpoint_path}")
    return MATIntervener.load_from_checkpoint(
        checkpoint_path,
        multiplier=multiplier,
        layer_norm_preserve=True,
    )


def run_evaluation(model, tokenizer, model_key, layer, mat_intervener=None,
                   instruction_prompt=EVAL_INSTRUCTION_PROMPT,
                   output_dir=None, tag=""):
    config_dict = None
    if mat_intervener is not None:
        config_dict = create_mat_pyvene_config([layer], mat_intervener)

    metric_names = ['mc']
    judge_name = os.environ.get('OPENAI_JUDGE_NAME', None)
    info_name = os.environ.get('OPENAI_INFO_NAME', None)

    if os.environ.get('OPENAI_API_KEY'):
        metric_names += ['judge', 'info']
    else:
        warnings.warn("OPENAI_API_KEY not set. Skipping GPT-judge and GPT-info metrics.")

    os.makedirs(output_dir, exist_ok=True)
    answer_dir = os.path.join(output_dir, 'answer_dump')
    summary_dir = os.path.join(output_dir, 'summary_dump')
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    input_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'TruthfulQA', 'TruthfulQA.csv'
    )

    filename = f"{model_key}_L{layer}_{tag}" if tag else f"{model_key}_L{layer}"
    output_path = os.path.join(answer_dir, f"{filename}.csv")
    summary_path = os.path.join(summary_dir, f"{filename}.csv")

    if mat_intervener is not None:
        model_to_eval = pv.IntervenableModel(config_dict, model)
    else:
        model_to_eval = model

    print(f"Running evaluation (tag={tag or 'default'})...")
    results = alt_tqa_evaluate(
        models={model_key: model_to_eval},
        metric_names=metric_names,
        input_path=input_csv,
        output_path=output_path,
        summary_path=summary_path,
        device="cuda",
        instruction_prompt=instruction_prompt,
        judge_name=judge_name,
        info_name=info_name,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAT-Steer checkpoint on TruthfulQA")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to MAT-Steer .pt checkpoint")
    parser.add_argument("--layer", type=int, default=INTERVENTION_LAYER)
    parser.add_argument("--instruction_prompt", type=str, default=EVAL_INSTRUCTION_PROMPT,
                        choices=["default", "informative"])
    parser.add_argument("--multiplier", type=float, default=EVAL_MULTIPLIER)
    parser.add_argument("--baseline", action="store_true", default=True,
                        help="Also compute baseline (no intervention)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()

    if args.output_dir is None:
        args.output_dir = EVAL_DIR

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path)
    mat_intervener = load_mat_intervener(args.checkpoint, args.multiplier)

    all_results = {}

    if args.baseline:
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION (no intervention)")
        print("=" * 60)
        baseline_results = run_evaluation(
            model, tokenizer, args.model_name, args.layer,
            mat_intervener=None,
            instruction_prompt=args.instruction_prompt,
            output_dir=args.output_dir,
            tag="baseline",
        )
        all_results['baseline'] = baseline_results.to_dict() if hasattr(baseline_results, 'to_dict') else str(baseline_results)

    print("\n" + "=" * 60)
    print("MAT-STEER EVALUATION")
    print("=" * 60)
    mat_results = run_evaluation(
        model, tokenizer, args.model_name, args.layer,
        mat_intervener=mat_intervener,
        instruction_prompt=args.instruction_prompt,
        output_dir=args.output_dir,
        tag="mat_steer",
    )
    all_results['mat_steer'] = mat_results.to_dict() if hasattr(mat_results, 'to_dict') else str(mat_results)

    results_path = os.path.join(args.output_dir, f"{args.model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'metadata': {
                'model_name': args.model_name,
                'checkpoint': args.checkpoint,
                'layer': args.layer,
                'multiplier': args.multiplier,
                'instruction_prompt': args.instruction_prompt,
            },
            'results': all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tag, res in [('BASELINE', baseline_results) if args.baseline else (None, None),
                     ('MAT-STEER', mat_results)]:
        if tag is None:
            continue
        print(f"\n{tag}:")
        if hasattr(res, 'to_numpy'):
            arr = res.to_numpy()[0].astype(float)
            labels = ['True*Info', 'True', 'Info', 'MC1', 'MC2', 'CE Loss', 'KL wrt Orig']
            for lbl, val in zip(labels, arr):
                print(f"  {lbl}: {val:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
