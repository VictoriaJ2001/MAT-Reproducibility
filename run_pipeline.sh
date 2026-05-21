#!/bin/bash
# ===========================================================================
# MAT-Steer Reproducibility Pipeline
# Run on server: bash run_pipeline.sh
# ===========================================================================
set -euo pipefail

# --- Environment ---
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# Change to project root (assumes script is at project root)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "MAT-Steer Pipeline"
echo "Project: $PROJECT_ROOT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# --- Directories ---
mkdir -p data features validation/checkpoints validation/evaluation_results

# ============================================================
# Step 0: Prepare Datasets (skip if data/*.json already exists)
# ============================================================
if [ -f data/truthfulqa.json ] && [ -f data/toxigen.json ] && [ -f data/bbq.json ]; then
    echo "[Step 0] Datasets already exist, skipping download."
else
    echo "[Step 0] Downloading datasets..."
    python scripts/prepare_data.py
fi

# ============================================================
# Step 0.5: Download Model (skip if already exists)
# ============================================================
MODEL_DIR="models/Llama-3-8B-Instruct"
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "[Step 0.5] Model already exists at $MODEL_DIR, skipping."
else
    echo "[Step 0.5] Downloading model from ModelScope..."
    python scripts/download_model_ms.py
fi

# ============================================================
# Step 1: Extract Activations
# ============================================================
echo ""
echo "[Step 1] Extracting activations at Layer 14..."

cd get_activations
for DATASET in truthfulqa toxigen bbq; do
    FEATURE_FILE="../features/llama3_8B_instruct_${DATASET}_labels.npy"
    if [ -f "$FEATURE_FILE" ]; then
        echo "  Activations for $DATASET already exist, skipping."
    else
        echo "  Extracting activations for $DATASET..."
        python get_activations.py \
            --model_name llama3_8B_instruct \
            --dataset_name "$DATASET" \
            --layer 14
    fi
done
cd ..

# ============================================================
# Step 2: Train MAT-Steer
# ============================================================
echo ""
echo "[Step 2] Training MAT-Steer model..."

CHECKPOINT="validation/checkpoints/llama3_8B_instruct_L14_mat_steer.pt"
if [ -f "$CHECKPOINT" ]; then
    echo "  Checkpoint already exists, skipping training."
else
    cd validation
    python steering.py \
        --model_name llama3_8B_instruct \
        --layer 14 \
        --save_path "../$CHECKPOINT" \
        --batch_size 96 \
        --epochs 100 \
        --lr 0.001 \
        --sigma 2.0 \
        --lambda_mmd 1.0 \
        --lambda_sparse 0.9 \
        --lambda_ortho 0.1 \
        --lambda_pos 0.9
    cd ..
fi

# ============================================================
# Step 3: Evaluate
# ============================================================
echo ""
echo "[Step 3] Evaluating on TruthfulQA..."

cd validation
python run_mat_eval.py \
    --model_name llama3_8B_instruct \
    --checkpoint "../$CHECKPOINT" \
    --layer 14 \
    --instruction_prompt default \
    --baseline
cd ..

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation results: validation/evaluation_results/"
echo "============================================================"
