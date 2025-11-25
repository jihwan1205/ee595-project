#!/bin/bash
#SBATCH --job-name=meanflow_sample
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:a6000:1
#SBATCH -w node6

# Generate Reflow samples script

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs
source ~/.bashrc
conda activate /tmp/$USER/.conda/envs/ee595

CKPT_PATH="/home/jihwanshin/EE595_project/runs/meanflow_20251118_140918/checkpoint_iter_150000.pt"
SAVE_DIR="./data/meanflow_reflow"

python generate_reflow_data.py \
    --ckpt_path $CKPT_PATH \
    --save_dir $SAVE_DIR \
    --num_pairs 200000 \
    --batch_size 512 \
    --num_inference_steps 32