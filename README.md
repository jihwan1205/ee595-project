# EE595 Project: Image Generation with Diffusion and Flow Models

## 📌 Overview
This project investigates few-step generative modeling on the **CelebA 64×64** dataset using a unified 50M-parameter U-Net architecture.  
We implement and compare three generative frameworks:

- **DDPM** (Denoising Diffusion Probabilistic Model): a classical diffusion model trained to progressively denoise images.  
- **FlowMatching**: a flow-based model using Conditional Flow Matching (CFM) to learn continuous velocity fields.  
- **MeanFlow**: a mean-flow model trained with an adaptive weighted loss to enable efficient few-step sampling.

To further improve low-NFE performance, we additionally explore **Rectified Flow**, applying it both to FlowMatching-pretrained models and MeanFlow-pretrained models for comparison.

Finally, we implement a two-stage pipeline that uses FlowMatching to generate synthetic (noise → image) transport pairs, which are then used to train a MeanFlow model.  
This produces a high-quality, few-step generative model capable of sampling images with as few as **1–4 NFEs**.

### Environment Setup

```bash
# Clone the repository
git clone git@github.com:jihwan1205/ee595-project.git
cd ee595-project

# Create Conda environment in /tmp (recommended for SLURM clusters)
conda create --prefix /tmp/$USER/.conda/envs/ee595-env python=3.10 -y
conda activate /tmp/$USER/.conda/envs/ee595-env

# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
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

## Stage 3 — MeanFlow Training
The MeanFlow model is trained on the generated pairs to learn a **straightened, near-optimal transport map**:

- Inputs: `(x1, x0)`  
- Objective: MeanFlow velocity estimation  
- Effect: straightens the FM flow, improving few-step sampling  
- Result: a highly efficient sampler that performs well even at **1–4 NFEs**

```shell
sbatch scripts/train/sbatch_meanflow.sh
```

---

