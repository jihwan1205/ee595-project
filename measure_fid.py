#!/usr/bin/env python3
"""
FID measurement script using clean-fid
Compute FID scores between generated samples and reference dataset
"""

import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm

from cleanfid import fid
from dataset import create_data_module
from src.utils import tensor_to_pil_image, print


def prepare_reference_images(data_root, reference_dir, num_samples=None, dataset_name="celeba64"):
    """
    Prepare reference images from validation dataset.
    
    Args:
        data_root: Root directory of the dataset
        reference_dir: Directory to save reference images
        num_samples: Number of samples to save (None = all validation images)
        dataset_name: Name of the dataset ('celebahq256', 'celeba128', or 'celeba64')
        
    Returns:
        Path to reference directory
    """
    # Create reference directory
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if reference images already exist
    existing_images = list(reference_dir.glob("*.png"))
    if existing_images:
        # Count existing images
        existing_count = len(existing_images)
        
        # If we have images and either:
        # 1. num_samples is None (save all) - assume existing images are complete
        # 2. num_samples is specified and we have at least that many
        if num_samples is None:
            print(f"Reference images already exist in {reference_dir} ({existing_count} images)")
            print(f"✓ Skipping reference image preparation")
            return str(reference_dir)
        elif existing_count >= num_samples:
            print(f"Reference images already exist in {reference_dir} ({existing_count} images, need {num_samples})")
            print(f"✓ Skipping reference image preparation")
            return str(reference_dir)
        else:
            print(f"Reference directory exists but has only {existing_count} images (expected {num_samples})")
            print(f"Re-saving reference images...")
    
    print(f"Preparing reference images from validation set ({dataset_name})...")
    
    # Load dataset
    data_module = create_data_module(
        dataset_name=dataset_name,
        root=data_root,
        batch_size=32,
        num_workers=4
    )
    val_loader = data_module.val_dataloader()
    
    # Save validation images
    count = 0
    for batch in tqdm(val_loader, desc="Saving reference images"):
        if num_samples and count >= num_samples:
            break
        
        # Convert to PIL images
        pil_images = tensor_to_pil_image(batch)
        
        for img in pil_images:
            if num_samples and count >= num_samples:
                break
            img.save(reference_dir / f"{count:04d}.png")
            count += 1
    
    print(f"✓ Saved {count} reference images to {reference_dir}")
    return str(reference_dir)


def compute_fid(generated_dir, reference_dir, device="cuda", batch_size=64):
    """
    Compute FID score using clean-fid.
    
    Args:
        generated_dir: Directory containing generated images
        reference_dir: Directory containing reference images
        device: Device to use for computation
        batch_size: Batch size for FID computation
    
    Returns:
        FID score
    """
    print(f"\nComputing FID...")
    print(f"  Generated: {generated_dir}")
    print(f"  Reference: {reference_dir}")
    
    fid_score = fid.compute_fid(
        generated_dir, 
        reference_dir, 
        device=device,
        batch_size=batch_size,
        num_workers=4
    )
    
    return fid_score


def main():
    parser = argparse.ArgumentParser(description="Measure FID scores using clean-fid")
    
    # Input directories
    parser.add_argument("--generated_dir", type=str, required=True,
                       help="Directory containing generated image files")
    parser.add_argument("--reference_dir", type=str, default=None,
                       help="Directory containing reference images (will be created from validation set if not exists). Defaults to dataset-specific path.")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="celeba64", choices=['celebahq256', 'celebahq', 'celebahq-256', 'celeba128', 'celeba-128', 'celeba64', 'celeba-64'],
                       help="Dataset to use for FID evaluation")
    
    # Reference preparation (only used if reference_dir doesn't exist)
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory of the dataset (for creating reference images if needed). Defaults to dataset-specific path.")
    
    # Computation settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for FID computation")
    
    # Output
    parser.add_argument("--output_path", type=str, default="./fid_results.json",
                       help="Path to save FID results as JSON")
    
    args = parser.parse_args()
    
    # Set default paths based on dataset
    if args.data_root is None:
        if args.dataset.lower() in ['celebahq256', 'celebahq', 'celebahq-256']:
            args.data_root = "./data/datasets/celebahq256"
        elif args.dataset.lower() in ['celeba128', 'celeba-128']:
            args.data_root = "./data/datasets/celeba128"
        elif args.dataset.lower() in ['celeba64', 'celeba-64']:
            args.data_root = "./data/datasets/celeba64"
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.reference_dir is None:
        if args.dataset.lower() in ['celebahq256', 'celebahq', 'celebahq-256']:
            args.reference_dir = "./data/celebahq256_256x256/val"
        elif args.dataset.lower() in ['celeba128', 'celeba-128']:
            args.reference_dir = "./data/celeba128_128x128/val"
        elif args.dataset.lower() in ['celeba64', 'celeba-64']:
            args.reference_dir = "./data/celeba64_64x64/val"
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Check generated directory exists
    if not os.path.exists(args.generated_dir):
        raise ValueError(f"Generated directory not found: {args.generated_dir}")
    
    # Prepare reference directory if it doesn't exist
    if not os.path.exists(args.reference_dir):
        print(f"Reference directory not found: {args.reference_dir}")
        print("Creating reference images from validation set...")
        args.reference_dir = prepare_reference_images(
            data_root=args.data_root,
            reference_dir=args.reference_dir,
            num_samples=None,
            dataset_name=args.dataset
        )
    
    print(f"\n{'='*60}")
    print(f"Computing FID")
    print(f"{'='*60}")
    print(f"Generated: {args.generated_dir}")
    print(f"Reference: {args.reference_dir}")
    
    # Compute FID
    fid_score = compute_fid(
        args.generated_dir,
        args.reference_dir,
        args.device,
        args.batch_size
    )
    
    print(f"\n{'='*60}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"{'='*60}")
    
    # Save results to JSON
    results = {
        "fid_score": float(fid_score),
        "generated_dir": args.generated_dir,
        "reference_dir": args.reference_dir
    }
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
