#!/usr/bin/env python3
"""
Skeleton training script for generative models
Students need to implement their own model classes and modify this script accordingly.
"""

import argparse
import types
import torch
import torch.optim as optim
import wandb
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
import torch.distributed as dist


from dataset import create_data_module, get_data_iterator
from src.utils import get_current_time, seed_everything, ddp_setup, print, is_main
from custom.trainer.base_trainer import BaseTrainer
from custom_model import create_custom_model
from custom.dataset.reflow_dataset import ReflowDataset
from src.sg_ratio_scheduler import SGRatioScheduler
from src.ema import EMA
from measure_fid import prepare_reference_images


def load_dataset(args):
    # Load dataset
    print("Loading dataset...")
    
    if args.use_reflow:
        try:
            reflow_dataset = ReflowDataset(root=args.reflow_dataset_path)
            train_sampler, val_sampler = None, None
            if args.use_ddp:
                world_size = dist.get_world_size()
                assert args.batch_size % world_size == 0, "Batch size must be divisible by number of GPUs"
                args.batch_size = args.batch_size // world_size  # Adjust batch size per GPU
                args.fid_batch_size = args.fid_batch_size // world_size  # Adjust FID batch size per GPU
                train_sampler = DistributedSampler(reflow_dataset, shuffle=True)
                val_sampler = DistributedSampler(reflow_dataset)
            
            train_dataloader = DataLoader(
                reflow_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=4,
                drop_last=True
            )
            val_dataloader = DataLoader(
                reflow_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=4,
                drop_last=False
            )
            
            train_iterator = get_data_iterator(train_dataloader)
            valid_iterator = get_data_iterator(val_dataloader)
        except Exception as e:
            print(f"Error loading Reflow dataset: {e}")
            return None, None
        return train_iterator, valid_iterator, train_sampler
    
    if args.use_ddp:
        world_size = dist.get_world_size()
        assert args.batch_size % world_size == 0, "Batch size must be divisible by number of GPUs"
        args.batch_size = args.batch_size // world_size  # Adjust batch size per GPU
        args.fid_batch_size = args.fid_batch_size // world_size  # Adjust FID batch size per GPU
        
        def train_dataloader_ddp(self):
            return DataLoader(
                self.train_ds, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                sampler=DistributedSampler(self.train_ds, shuffle=True),
                drop_last=True,
            )

        def val_dataloader_ddp(self):
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                sampler=DistributedSampler(self.val_ds),
                drop_last=False,
            )

        try:
            data_module = create_data_module(
                dataset_name=args.dataset,
                root=args.data_root,
                batch_size=args.batch_size,
                num_workers=4,
                val_split=args.val_split,
                seed=args.seed
            )
            
            data_module.train_dataloader_ddp = types.MethodType(train_dataloader_ddp, data_module)
            data_module.val_dataloader_ddp = types.MethodType(val_dataloader_ddp, data_module)

            train_loader = data_module.train_dataloader_ddp()
            train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler) else None
            train_iterator = get_data_iterator(train_loader)
            valid_loader = data_module.val_dataloader_ddp()
            valid_iterator = get_data_iterator(valid_loader)
            
            print(f"Total iterations: {args.num_iterations}")
            print(f"Dataset resolution: {data_module.image_resolution}x{data_module.image_resolution}")
            # Store image resolution in args for use in trainer
            args.image_resolution = data_module.image_resolution
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None
    else:
        try:
            data_module = create_data_module(
                dataset_name=args.dataset,
                root=args.data_root,
                batch_size=args.batch_size,
                num_workers=4,
                val_split=args.val_split,
                seed=args.seed
            )
            
            train_loader = data_module.train_dataloader()
            train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler) else None
            train_iterator = get_data_iterator(train_loader)
            valid_loader = data_module.val_dataloader()
            valid_iterator = get_data_iterator(valid_loader)
            
            print(f"Total iterations: {args.num_iterations}")
            print(f"Dataset resolution: {data_module.image_resolution}x{data_module.image_resolution}")
            # Store image resolution in args for use in trainer
            args.image_resolution = data_module.image_resolution
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None
    
    return train_iterator, valid_iterator, train_sampler
    
def main(args):
    # setup DDP
    if args.use_ddp:
        ddp_setup()
        
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
    
    if args.val_reference_dir is None:
        if args.dataset.lower() in ['celebahq256', 'celebahq', 'celebahq-256']:
            args.val_reference_dir = "./data/celebahq256_256x256/val"
        elif args.dataset.lower() in ['celeba128', 'celeba-128']:
            args.val_reference_dir = "./data/celeba128_128x128/val"
        elif args.dataset.lower() in ['celeba64', 'celeba-64']:
            args.val_reference_dir = "./data/celeba64_64x64/val"
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.eval_fid:
        prepare_reference_images(
            data_root=args.data_root,
            reference_dir=args.val_reference_dir,
            num_samples=None,
            dataset_name=args.dataset
        )

    # Set seed for reproducibility
    seed_everything(args.seed)
    print(f"Seed set to: {args.seed}")
    
    # Add timestamp if save directory is not specified
    if args.save_dir == "./results":
        args.save_dir = f"./results/{get_current_time()}"
    
    if args.use_wandb and is_main():
        if args.wandb_exp_name is None:
            args.wandb_exp_name = f"{get_current_time()}"
            
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=vars(args),
            name=args.wandb_exp_name
        )
    
    args.save_dir = Path(args.save_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")

    # Prepare kwargs for create_custom_model from args
    model_kwargs = {}
    # Pass all custom arguments except training-specific ones
    # Network hyperparameters (ch, ch_mult, attn, num_res_blocks, dropout) are FIXED
    # Students can add their own custom arguments for scheduler/model configuration
    excluded_keys = ['device', 'batch_size', 'num_iterations', 
                     'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
    for key, value in args.__dict__.items():
        if key not in excluded_keys and value is not None:
            model_kwargs[key] = value

    try:
        model = create_custom_model(**model_kwargs)
        print(f"Model created: {type(model).__name__}")
        print(f"Scheduler: {type(model.scheduler).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.network.parameters()):,}")
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("Please implement the CustomScheduler and CustomGenerativeModel classes in custom_model.py")
        return
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Prepare model configuration for reproducibility
    # This will be saved together with each checkpoint
    model_config = {
        'model_type': type(model).__name__,
        'scheduler_type': type(model.scheduler).__name__,
        **model_kwargs  # Include all custom model arguments
    }
    
    # Load dataset
    result = load_dataset(args)
    if result is None or result[0] is None:
        print("Failed to load dataset")
        return
    train_iterator, valid_iterator, train_sampler = result
    
    # Create optimizer with model-specific settings
    if args.model_type in ('MeanFlow'):
        # MeanFlow and AlphaFlow use different Adam betas
        optimizer = optim.Adam(
            model.network.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.95)
        )
        print(f"Using Adam optimizer with betas=(0.9, 0.95) for MeanFlow and AlphaFlow")
    else:
        # Default Adam settings for other models
        optimizer = optim.Adam(model.network.parameters(), lr=args.lr)
        print(f"Using Adam optimizer with default betas=(0.9, 0.999)")
    
    
    # Setup EMA for MeanFlow
    ema = None
    if args.model_type == 'MeanFlow' and args.use_ema:
        ema = EMA(model, decay=args.ema_decay, device=dist.get_rank() if args.use_ddp else device)
        print(f"Using EMA with decay={args.ema_decay}")
    
    
    # Learning rate scheduler: Cosine annealing with linear warmup
    if args.use_lr_scheduler:
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            # Linear warmup: linearly increase from 0 to 1.0
            if current_step < args.warmup_steps:
                return float(current_step + 1) / float(args.warmup_steps)
            # Cosine annealing: smoothly decrease from 1.0 to min_lr_ratio
            progress = float(current_step - args.warmup_steps) / float(max(1, args.num_iterations - args.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine_decay
        
        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"Using LR Scheduler: Cosine Annealing with Warmup")
        print(f"  - Warmup steps: {args.warmup_steps}")
        print(f"  - Min LR ratio: {args.min_lr_ratio}")
    else:
        lr_scheduler = None
        print("Not using LR Scheduler (constant learning rate)")
    
    
    if args.meanflow_use_sg_scheduler:
        sg_ratio_scheduler = SGRatioScheduler(
            model=model,
            sg_ratio_warmup_iters=args.meanflow_sg_ratio_warmup_iters
        )
        print(f"Using Stop-Gradient Ratio Scheduler for MeanFlow")
        print(f"  - Warmup iters: {args.meanflow_sg_ratio_warmup_iters}")
    else:
        sg_ratio_scheduler = None
        print("Not using Stop-Gradient Ratio Scheduler")
    
    
    # Load checkpoint if resuming (expects EMA checkpoint with full training state)
    resume_iteration = None
    
    if args.resume_ema_ckpt:
        print(f"Resuming training from EMA checkpoint: {args.resume_ema_ckpt}")
        ckpt = torch.load(args.resume_ema_ckpt, map_location='cpu')
        
        # Load model state
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            print("  ✓ Model state loaded")
        else:
            raise ValueError("Checkpoint must contain 'state_dict' for model")
        
        # Load optimizer state if available
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("  ✓ Optimizer state loaded")
        
        # Load lr_scheduler state if available
        if 'lr_scheduler' in ckpt and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                print("  ✓ LR scheduler state loaded")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load LR scheduler state: {e}")
        
        # Load EMA state (required for EMA checkpoints)
        if 'ema' in ckpt:
            if ema is None:
                print("Warning: EMA checkpoint specified but EMA is not enabled. Creating EMA with default decay.")
                ema = EMA(model, decay=args.ema_decay, device=dist.get_rank() if args.use_ddp else device)
            ema.load_state_dict(ckpt['ema'])
            print("  ✓ EMA state loaded")
        else:
            raise ValueError("EMA checkpoint must contain 'ema' state dict")
        
        # Get iteration number if available
        if 'iteration' in ckpt:
            resume_iteration = ckpt['iteration']
            print(f"  ✓ Resuming from iteration {resume_iteration}")
        
        print("EMA checkpoint loaded successfully.")
    
    
    trainer = BaseTrainer(
        args=args,
        model=model,
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        sg_ratio_scheduler=sg_ratio_scheduler,
        ema=ema,
        model_config=model_config,
        train_sampler=train_sampler,
    )
    
    # Set resume iteration if available
    if resume_iteration is not None:
        trainer.iteration = resume_iteration
    
    trainer.train()
    dist.destroy_process_group()
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train generative model")
    
    # General training arguments
    parser.add_argument("--model_type", type=str, choices=['DDPM', 'DDIM', 'FlowMatching', 'MeanFlow'], help="Type of generative model to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_iterations", type=int, default=100_000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Maximum gradient norm for clipping (0 for no clipping)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on")
    parser.add_argument("--use_ddp", action="store_true", help="Use Distributed Data Parallel (DDP) for multi-GPU training")
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging")
    parser.add_argument("--save_interval", type=int, default=10000, help="Interval for saving checkpoints and samples")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--sample_interval", type=int, default=2000, help="Interval for generating samples during training")
    parser.add_argument("--sample_nfe_list", type=int, nargs='+', default=[1, 2, 4], help="Number of steps for sampling during training")
    parser.add_argument("--sample_batch_size", type=int, default=16, help="Batch size for sampling during training")
    parser.add_argument("--resume_ema_ckpt", type=str, default=None, help="Path to EMA checkpoint to resume training from (must contain full training state: model, optimizer, lr_scheduler, iteration, EMA)")
    
    # Validation arguments
    parser.add_argument("--val_interval", type=int, default=1000, help="Interval for validation")
    parser.add_argument("--val_max_batches", type=int, default=100, help="Max batches for validation (to speed up)")
    parser.add_argument('--skip_initial_evaluation', '-sie', action='store_true', help="Skip initial evaluation before training")
    parser.add_argument("--eval_fid", action='store_true', help="Evaluate FID score during validation")
    parser.add_argument("--dataset", type=str, default="celeba64", choices=['celebahq256', 'celebahq', 'celebahq-256', 'celeba128', 'celeba-128', 'celeba64', 'celeba-64'],
                       help="Dataset to use for training")
    parser.add_argument("--val_reference_dir", type=str, default=None, help="Directory to save/load reference images for FID evaluation. If not exists, will be created and populated with validation images. Defaults to dataset-specific path.")
    parser.add_argument("--fid_nfe_list", type=int, nargs='+', default=[1, 2, 4], help="Number of steps for FID evaluation during validation")
    parser.add_argument("--fid_batch_size", type=int, default=1024, help="Batch size for FID evaluation")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Root directory of the dataset (for creating reference images if needed). Defaults to dataset-specific path.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (only used for CelebA-HQ-256)") 

    # wandb arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_entity", type=str, default="diffusion_challenge", help="Wandb entity (team) name")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project", help="Wandb project name")
    parser.add_argument("--wandb_exp_name", type=str, default=None, help="Wandb experiment name")
    
    # Learning rate scheduler arguments
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use learning rate scheduler (Cosine annealing with warmup)")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05, help="Minimum learning rate as ratio of initial LR")
    
    # EMA arguments
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average (EMA) for model parameters")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")

    # DO NOT MODIFY THE PROVIDED NETWORK HYPERPARAMETERS 
    # (ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1)
    parser.add_argument("--use_additional_condition", action="store_true", help="Use additional condition embedding in U-Net (e.g., step size for Shortcut Models or end timestep for Consistency Trajectory Models)")
    
    # diffusion arguments
    parser.add_argument("--num_train_timesteps", '-ntt', type=int, default=1000, help="Number of training timesteps for scheduler")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting beta value for linear schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta value for linear schedule")
    # parser.add_argument("--diffusion_objective", type=str, default="pred_noise", choices=['pred_noise', 'pred_x0', 'pred_v'], help="Diffusion objective for training")
    # parser.add_argument("--beta_schedule", type=str, default="linear", choices=['linear'], help="Beta schedule type")
    
    # meanflow arguments
    parser.add_argument("--meanflow_scale_timestep", action="store_true", help="Scale timestep for MeanFlow")
    parser.add_argument("--meanflow_loss_eps", type=float, default=1e-3, help="Epsilon for MeanFlow loss")
    parser.add_argument("--meanflow_loss_p", type=float, default=1.0, help="P parameter for MeanFlow loss")
    parser.add_argument("--meanflow_loss_fn", type=str, default='adaptive_loss', choices=['mse_loss', 'adaptive_loss'], help="Loss function for MeanFlow")
    parser.add_argument("--meanflow_sg_ratio", type=float, default=0.0, help="sg_ratio = 1: full backprop; sg_ratio = 0: full stop-gradient")
    parser.add_argument("--meanflow_use_sg_scheduler", action="store_true", help="Use stop-gradient ratio scheduler for MeanFlow")
    parser.add_argument("--meanflow_sg_ratio_warmup_iters", type=int, default=7000, help="Number of iterations to warmup stop-gradient ratio from 0.0 to 1.0 for MeanFlow")
    
    # Reflow arguments
    parser.add_argument("--use_reflow", action="store_true", help="Use Reflow dataset and training procedure")
    parser.add_argument("--reflow_dataset_path", type=str, default="data/fm_reflow", help="Path to the Reflow dataset")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)