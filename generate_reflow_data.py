#!/usr/bin/env python3
"""
Generate Reflow dataset by sampling from a trained model
Creates (x0, x1) pairs where x0 is noise and x1 is generated sample
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils import load_model, seed_everything
from custom_model import create_custom_model


def generate_reflow_pairs(model, num_pairs, batch_size, num_inference_steps, device, save_dir):
    """
    Generate (x0, x1) pairs for reflow training.
    
    Args:
        model: Trained generative model
        num_pairs: Number of pairs to generate
        batch_size: Batch size for generation
        num_inference_steps: Number of steps for sampling
        device: Device to run on
        save_dir: Directory to save pairs
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    num_batches = int(np.ceil(num_pairs / batch_size))
    
    print(f"Generating {num_pairs} reflow pairs...")
    print(f"Saving to: {save_dir}")
    if torch.cuda.is_available():
        print(f"Using device: {device}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    pair_idx = 0
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating pairs"):
            current_batch_size = min(batch_size, num_pairs - pair_idx)
            
            # Get image resolution from model or args
            # Assuming 64x64 for now, adjust as needed
            image_resolution = 64  # Change this based on your dataset
            shape = (current_batch_size, 3, image_resolution, image_resolution)
            
            # Sample from model to get x1 (generated images)
            # Ensure sampling happens on the correct device
            if torch.cuda.is_available() and batch_idx == 0:
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1024**2
            
            x1 = model.sample(shape, num_inference_timesteps=num_inference_steps, verbose=False)
            
            # # Verify x1 is on the correct device
            # if x1.device != device:
            #     print(f"Warning: x1 is on {x1.device}, moving to {device}")
            #     x1 = x1.to(device)
            
            # x0 is the noise (prior distribution) - create on same device as x1
            x0 = torch.randn_like(x1)
            
            # # Debug info for first batch
            # if torch.cuda.is_available() and batch_idx == 0:
            #     mem_after = torch.cuda.memory_allocated() / 1024**2
            #     mem_peak = torch.cuda.max_memory_allocated() / 1024**2
            #     print(f"\nFirst batch GPU memory:")
            #     print(f"  Before sampling: {mem_before:.2f} MB")
            #     print(f"  After sampling: {mem_after:.2f} MB")
            #     print(f"  Peak memory: {mem_peak:.2f} MB")
            #     print(f"  x1 device: {x1.device}, x0 device: {x0.device}")
            
            # Save each pair
            for i in range(current_batch_size):
                pair = {
                    'x0': x0[i].cpu(),  # Noise (starting point)
                    'x1': x1[i].cpu()   # Generated sample (end point)
                }
                torch.save(pair, save_dir / f"pair_{pair_idx:06d}.pt")
                pair_idx += 1
    
    print(f"✓ Generated {pair_idx} pairs and saved to {save_dir}")


def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    print(f"Loading model from: {args.ckpt_path}")
    model = load_model(args.ckpt_path, create_custom_model, str(device), args.config_path)
    
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    
    # Verify model is on GPU
    model_device = next(model.parameters()).device
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Model device: {model_device}")
    if torch.cuda.is_available():
        print(f"✓ GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate reflow pairs
    generate_reflow_pairs(
        model=model,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        device=device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Reflow dataset")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config (default: model_config.json in checkpoint dir)")
    parser.add_argument("--save_dir", type=str, default="./data/reflow",
                       help="Directory to save reflow pairs")
    parser.add_argument("--num_pairs", type=int, default=50000,
                       help="Number of (x0, x1) pairs to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                       help="Number of inference steps for sampling")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    main(args)