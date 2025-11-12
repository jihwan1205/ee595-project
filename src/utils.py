import random
import torch
import json
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from datetime import datetime
import os
from cleanfid import fid
import torch.distributed as dist
import builtins

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now
    
def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Expand timestep tensor to match the spatial dimensions of x.
    
    Args:
        t: Timestep tensor of shape (batch_size,)
        x: Data tensor of shape (batch_size, channels, height, width)
        
    Returns:
        Expanded timestep tensor
    """
    for _ in range(x.ndim - 1):
        t = t.unsqueeze(-1)
    return t

def tensor_to_pil_image(tensor: torch.Tensor):
    """
    Convert tensor to PIL image.
    
    Args:
        tensor: Tensor of shape (batch_size, channels, height, width) or (channels, height, width)
    Returns:
        List of PIL images or single PIL image if a batch size is 1.
    """
    # Handle 3D tensors (single image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    single_image = tensor.shape[0] == 1

    # denormalize from [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5)
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL
    tensor = tensor.clone().detach().cpu().permute(0,2,3,1).numpy()
    images = (tensor * 255).round().astype("uint8")
    images = [Image.fromarray(image) for image in images]
    if single_image:
        return images[0]
    return images


def save_model(model, checkpoint_path: str, model_config: Dict[str, Any], config_path: str = None):
    """
    Save model checkpoint and configuration JSON file together.
    
    Args:
        model: The generative model to save
        checkpoint_path: Path to save the checkpoint (.pt file)
        model_config: Dictionary containing model configuration
        config_path: Path to save the config JSON file (default: same directory as checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    
    # Save model config JSON
    if config_path is None:
        config_path = checkpoint_path.parent / 'model_config.json'
    else:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)

def save_ema(ema_state_dict, checkpoint_path: str):
    """
    Save EMA state dict to checkpoint.
    
    Args:
        ema_state_dict: EMA state dict to save
        checkpoint_path: Path to save the EMA checkpoint (.pt file)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(ema_state_dict, checkpoint_path)


def load_model(checkpoint_path: str, create_model_fn, device: str = "cpu", config_path: str = None):
    """
    Load model from checkpoint and configuration JSON file.
    
    Args:
        checkpoint_path: Path to the checkpoint (.pt file)
        create_model_fn: Function to create the model (e.g., create_custom_model)
        device: Device to load the model on
        config_path: Path to the config JSON file (default: same directory as checkpoint)
        
    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Determine config path
    if config_path is None:
        config_path = checkpoint_path.parent / 'model_config.json'
    else:
        config_path = Path(config_path)
    
    # Load model config
    model_kwargs = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        for key, value in saved_config.items():
            model_kwargs[key] = value
    
    # Create model architecture
    model = create_model_fn(device=device, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def transform_sample_for_fid(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 3:
        x = x.unsqueeze(0)

    # denormalize from [-1, 1] to [0, 1]
    x = (x * 0.5 + 0.5)

    # Clamp values to [0, 1] range
    x = torch.clamp(x, 0, 1)

    # Convert to PIL
    x = x.clone().detach().cpu()
    images = (x * 255).round().to(torch.uint8)
    return images

def compute_image_folder_statistics(folder_path: str, feat_model, device):
    if os.path.exists(os.path.join(folder_path, 'stats.npz')):
        stats = np.load(os.path.join(folder_path, 'stats.npz'))
        mu, sigma = stats['mu'], stats['sigma']
        return mu, sigma
    
    fbname2 = os.path.basename(folder_path)
    np_feats = fid.get_folder_features(
        folder_path,
        feat_model,
        num_workers=4,
        batch_size=64,
        device=device,
        mode='clean',
        description=f"FID {fbname2} : ",
        verbose=False,
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    
    # save mu and sigma as numpy
    np.savez(os.path.join(folder_path, 'stats.npz'), mu=mu, sigma=sigma)

    return mu, sigma

def compute_gen_image_statistics(images: torch.Tensor, feat_model, num_gen, batch_size, use_ddp, device, z_dim=3*64*64):
    images = transform_sample_for_fid(images)
    gen = lambda z: images

    np_feats = fid.get_model_features(
        gen,
        feat_model,
        mode='clean',
        z_dim=z_dim,
        num_gen=num_gen,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    
    if use_ddp:
        gathered_feats = None
        if is_main():
            gathered_feats = [torch.zeros_like(torch.tensor(np_feats), device=device) for _ in range(dist.get_world_size())]
        dist.gather(torch.tensor(np_feats, device=device), gather_list=gathered_feats, dst=0)
        if is_main():
            np_feats = torch.cat(gathered_feats, dim=0).cpu().numpy()
        del gathered_feats
        
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)

    return mu, sigma

def ddp_setup():
    dist.init_process_group("nccl")

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def print(*args, **kwargs):
    if is_main():
        builtins.print(*args, **kwargs)