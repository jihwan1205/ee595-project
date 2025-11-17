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


## Stage 1 — Flow Matching Pre-training
We begin by training a 50M-parameter U-Net using standard **flow matching**:

- Forward process: linear interpolation  
- Target velocity: `v = x0 - x1`  
- Timesteps sampled from LogNormal distribution  
- Optimizer: AdamW  
- Dataset: CelebA 64×64  
- Goal: learn a strong flow field capable of sampling with 32–40 NFEs

After training, the FM model produces clean images with moderately long ODE trajectories.

```shell
sbatch scripts/train/sbatch_fm.sh
```

---


## Stage 2 — Synthetic Pair Generation
Using the pretrained FM model, we generate **100,000 (noise, image)** endpoint pairs:

- Noise sampled from `N(0, I)`
- Samples generated with **NFE = 32**
- Saved pairs:  
  - `x1`: initial noise  
  - `x0`: FM-generated image  

These pairs approximate optimal transport endpoints and serve as training data for MeanFlow.
```shell
sbatch scripts/samples/generate_reflow_samples.sh
```


---

## ⚡ Stage 3 — MeanFlow Training
The MeanFlow model is trained on the generated pairs to learn a **straightened, near-optimal transport map**:

- Inputs: `(x1, x0)`  
- Objective: MeanFlow velocity estimation  
- Effect: straightens the FM flow, improving few-step sampling  
- Result: a highly efficient sampler that performs well even at **1–4 NFEs**

```shell
sbatch scripts/train/sbatch_meanflow.sh
```

---

