#!/bin/bash
#SBATCH --job-name=reflow
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:a6000:4
#SBATCH -w node5


set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs
# source ~/.bashrc
# conda activate /tmp/$USER/.conda/envs/ee595

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_meanflow_reflow_${TIMESTAMP}.log"

# Training configuration
NUM_ITERATIONS=100000
BATCH_SIZE=64
MODEL_TYPE="MeanFlow"
WANDB_PROJECT="image_generation"
WANDB_ENTITY="few-step-video-generation"
# WANDB_EXP_NAME="meanflow_reflow_fm_${TIMESTAMP}"
# REFLOW_DATASET_PATH="./data/fm_reflow"
# RESUME_CKPT="runs/fm_20251112_212008/checkpoint_iter_170000.pt"
WANDB_EXP_NAME="meanflow_reflow_meanflow_${TIMESTAMP}"
REFLOW_DATASET_PATH="./data/meanflow_reflow"
RESUME_CKPT="runs/meanflow_20251112_214732/checkpoint_iter_250000.pt"
SAVE_DIR="runs/${WANDB_EXP_NAME}"
LR=1e-4
NUM_TRAIN_TIMESTEPS=1000



# Additional training settings
LOG_INTERVAL=100
SAVE_INTERVAL=10000
VAL_INTERVAL=1000
SAMPLE_INTERVAL=5000

export PYTHONUNBUFFERED=1

echo "Starting Reflow Training"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Configuration:"
echo "  - Model Type: $MODEL_TYPE"
echo "  - Iterations: $NUM_ITERATIONS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LR"
echo "  - Save Directory: $SAVE_DIR"
echo "  - WandB Project: $WANDB_PROJECT"
echo ""

cmd="torchrun --standalone --nproc_per_node=gpu train.py \
    --use_ddp \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --num_iterations $NUM_ITERATIONS \
    --lr $LR \
    --device cuda \
    --num_train_timesteps $NUM_TRAIN_TIMESTEPS \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --val_interval $VAL_INTERVAL \
    --sample_interval $SAMPLE_INTERVAL \
    --save_dir $SAVE_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --wandb_exp_name $WANDB_EXP_NAME \
    --use_wandb \
    --use_lr_scheduler \
    --warmup_steps 2000 \
    --min_lr_ratio 0.05 \
    --beta_start 1e-4 \
    --beta_end 0.02 \
    --eval_fid \
    --fid_nfe_list 1 2 4 \
    --use_reflow \
    --reflow_dataset_path $REFLOW_DATASET_PATH \
    --resume_ckpt $RESUME_CKPT \
    --fid_batch_size 512"
    



echo "Command: $cmd"
echo ""
echo "============================================"
echo "Training started at $(date)"
echo "============================================"

# Execute training and redirect both stdout and stderr to log file
# Also display output on terminal using tee
eval $cmd 2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================"
echo "Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
