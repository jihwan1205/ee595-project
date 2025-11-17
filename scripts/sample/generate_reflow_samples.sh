#!/bin/bash
#SBATCH --job-name=fm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:a6000:1
#SBATCH -w node5

# Generate Reflow samples script

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs
source ~/.bashrc
conda activate /tmp/$USER/.conda/envs/ee595

# CKPT_PATH="./runs/fm_20251112_212008/checkpoint_iter_170000.pt"
# SAVE_DIR="./data/fm_reflow"
CKPT_PATH="./runs/meanflow_20251112_214732/checkpoint_iter_250000.pt"
SAVE_DIR="./data/meanflow_reflow"

python generate_reflow_data.py \
    --ckpt_path $CKPT_PATH \
    --save_dir $SAVE_DIR \
    --num_pairs 100000 \
    --batch_size 512 \
    --num_inference_steps 16