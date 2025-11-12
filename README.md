# EE595 Project: Image Generation with Diffusion and Flow Models

This repository implements three generative models for image generation:
- **DDPM** (Denoising Diffusion Probabilistic Model): A diffusion-based model that learns to denoise images
- **FlowMatching**: A flow-based model using Conditional Flow Matching (CFM) that learns velocity fields
- **MeanFlow**: A mean flow model with adaptive weighted loss

All models use a U-Net backbone architecture and are trained on CelebA datasets (64x64, 128x128, or 256x256 resolution). The project includes training scripts, FID evaluation, and sampling utilities.

## Environment Setup on slurm compute node

```shell
git clone git@github.com:jihwan1205/ee595-project.git
cd ee595-project
conda create --prefix /tmp/$USER/.conda/envs/ee595-env python=3.10 -y
conda activate ee595-env
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Running Training Scripts

The project includes SLURM batch scripts for training each model on compute nodes. All scripts are located in `scripts/train/`.

### Training DDPM

```shell
sbatch scripts/train/sbatch_ddpm.sh
```

This script trains a DDPM model with:
- 1M iterations
- Batch size: 32
- Learning rate: 1e-4
- Logs saved to `logs/train_ddpm_<timestamp>.log`
- Checkpoints saved to `runs/ddpm_<timestamp>/`

### Training FlowMatching

```shell
sbatch scripts/train/sbatch_fm.sh
```

This script trains a FlowMatching model with:
- 1M iterations
- Batch size: 32
- Learning rate: 1e-4
- Uses DDP (Distributed Data Parallel) with `torchrun`
- Logs saved to `logs/train_fm_<timestamp>.log`
- Checkpoints saved to `runs/fm_<timestamp>/`

### Training MeanFlow

```shell
sbatch scripts/train/sbatch_meanflow.sh
```

This script trains a MeanFlow model with:
- 1M iterations
- Batch size: 16
- Learning rate: 1e-4
- Uses DDP (Distributed Data Parallel) with `torchrun`
- Logs saved to `logs/train_meanflow_<timestamp>.log`
- Checkpoints saved to `runs/meanflow_<timestamp>/`

### Notes

- All scripts require a SLURM cluster with GPU access (A6000 GPU specified)
- Scripts automatically create log directories and timestamped experiment folders
- Training progress is logged to both files and WandB (if configured)
- FID evaluation is performed during training at specified intervals

