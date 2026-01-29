#!/bin/bash

# This script runs two training sessions with different reasoner depths.
# The differences are primarily in the model_path (where the depth is defined).

# --- RUN 1 CONFIG ---
MODEL_D1="/llm-reco-ssd-share/baohonghui/ckpt/converted"
OUTPUT_D1="/llm-reco-ssd-share/baohonghui/torchrec/SumRec/merge/ckpt/reasoner_epoch6"

# --- RUN 2 CONFIG ---
MODEL_D2="/llm-reco-ssd-share/baohonghui/ckpt/converted-loop4"
OUTPUT_D2="/llm-reco-ssd-share/baohonghui/torchrec/SumRec/merge/ckpt/reasoner_loop4_epoch6"

# 1. Execute First Training Run
echo "Starting Run 1 (Standard/Low Depth)..."
bash honghui_train_8b_mpi.sh "$MODEL_D1" "$OUTPUT_D1"

echo "Run 1 Finished. Waiting 60 seconds for cluster resource cleanup (NCCL/GPU memory)..."
sleep 60

echo "Starting Run 2 (Increased Depth)..."

# 2. Execute Second Training Run
bash honghui_train_8b_mpi.sh "$MODEL_D2" "$OUTPUT_D2"

echo "Dual Training Complete."
