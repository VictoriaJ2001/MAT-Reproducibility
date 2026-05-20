"""
Centralized configuration for MAT-Steer reproducibility.

All model paths, dataset names, hyperparameters, and directory paths
are defined here so that individual scripts can import them consistently.
"""

import os

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Maps short model names → local path or HuggingFace model ID.
# On the server, the model is downloaded via ModelScope to:
#     ./models/Llama-3-8B-Instruct/
#   → absolute path: {PROJECT_ROOT}/models/Llama-3-8B-Instruct/
#
# For HuggingFace IDs, set HF_ENDPOINT=https://hf-mirror.com on the server
# to use the mirror.
MODEL_NAME_TO_PATH = {
    # --- Local paths (preferred, no network needed) ---
    "llama3_8B_instruct": os.path.join(PROJECT_ROOT, "models", "Llama-3-8B-Instruct"),

    # --- HuggingFace IDs (fallback, requires network) ---
    "llama3_8B": "meta-llama/Meta-Llama-3-8B",
    "llama3_8B_instruct_hf": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1_8B": "meta-llama/Llama-3.1-8B",
    "llama3.1_8B_chat": "mathewhe/Llama-3.1-8B-Chat",
    "llama3_70B": "meta-llama/Meta-Llama-3-70B",
    "llama3_70B_instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama2_chat_7B": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_chat_13B": "meta-llama/Llama-2-13b-chat-hf",
    "llama2_chat_70B": "meta-llama/Llama-2-70b-chat-hf",
    "llama_7B": "huggyllama/llama-7b",
    "alpaca_7B": "circulus/alpaca-7b",
    "vicuna_7B": "AlekseyKorshuk/vicuna-7b",
    "qwen2.5_7B": "Qwen/Qwen2.5-7B",
}

# The model used for MAT-Steer reproduction
MODEL_NAME = "llama3_8B_instruct"

# ---------------------------------------------------------------------------
# Intervention layer (paper uses Layer 14)
# ---------------------------------------------------------------------------
INTERVENTION_LAYER = 14

# ---------------------------------------------------------------------------
# Datasets for 3-attribute steering
# ---------------------------------------------------------------------------
DATASETS = ["truthfulqa", "toxigen", "bbq"]
NUM_ATTRIBUTES = len(DATASETS)

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "validation", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "validation", "evaluation_results")

# ---------------------------------------------------------------------------
# Training hyperparameters (from paper / README)
# ---------------------------------------------------------------------------
TRAIN_HYPERPARAMS = {
    "batch_size": 96,
    "epochs": 100,
    "lr": 0.001,
    "sigma": 2.0,
    "lambda_mmd": 1.0,
    "lambda_sparse": 0.9,
    "lambda_ortho": 0.1,
    "lambda_pos": 0.9,
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_MULTIPLIER = 1.0         # Intervention strength
EVAL_INSTRUCTION_PROMPT = "default"  # "default" or "informative"


def get_model_path(model_name: str = None) -> str:
    """
    Resolve the model name to an actual path/ID.

    Returns the local path if it exists, otherwise the HF ID.
    """
    name = model_name or MODEL_NAME
    path = MODEL_NAME_TO_PATH.get(name, name)
    # If it's a local path and exists, use it; otherwise return as-is (HF ID)
    if os.path.isdir(path):
        return path
    return path  # could be a HuggingFace model ID


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, FEATURES_DIR, CHECKPOINT_DIR, EVAL_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {MODEL_NAME} → {get_model_path()}")
    print(f"Datasets: {DATASETS}")
    print(f"Intervention layer: {INTERVENTION_LAYER}")
    print(f"Directories:\n  data:       {DATA_DIR}\n  features:   {FEATURES_DIR}\n  checkpoints: {CHECKPOINT_DIR}\n  eval:       {EVAL_DIR}")
