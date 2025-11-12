import torch
from src.base_model import BaseScheduler


class MeanFlowScheduler(BaseScheduler):
    """
    MeanFlow Scheduler
    
    Implements MeanFlow which predicts the integrated flow over [r, t].
    Instead of predicting instantaneous velocity, it predicts the mean flow
    between two time points.
    
    Forward process: z = (1 - t) * data + t * noise
    Target: u_tgt = (noise - data) - (t - r) * du/dt
    """

    def __init__(self, 
        num_train_timesteps: int = 1000,
        **kwargs
    ):
        super().__init__(num_train_timesteps, **kwargs)

    def sample_timesteps(self, batch_size: int, device: torch.device, mean: float = -0.4, std: float = 1.0, diff_ratio: float = 0.25):
        """
        Sample random timesteps t and r for training using lognormal distribution.
        
        Sampling strategy:
        1. Sample t and r independently from lognorm(-0.4, 1.0)
        2. Swap if necessary to ensure t >= r
        3. With 75% probability, set r = t; 25% probability, keep r != t
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place timesteps on
            
        Returns:
            Tuple of (t, r) tensors of shape (batch_size,)
        """
        # Sample from lognormal distribution: lognorm(-0.4, 1.0)
        # lognormal = exp(normal(mean, std))
        # Sample t and r independently
        
        normal_t = torch.randn(batch_size, device=device) * std + mean
        normal_r = torch.randn(batch_size, device=device) * std + mean
        
        t = torch.sigmoid(normal_t)
        r = torch.sigmoid(normal_r)

        # Clip to [0, 1] range (in case of extreme values)
        t = torch.clamp(t, 0.0, 1.0)
        r = torch.clamp(r, 0.0, 1.0)
        
        # Swap to ensure t >= r
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        
        # 75% of samples: set r = t
        # 25% of samples: keep r != t
        equal_mask = torch.rand(batch_size, device=device) < 1 - diff_ratio
        r = torch.where(equal_mask, t, r)
        
        return t, r
    
    def forward_process(self, data, noise, t):
        """
        Apply the forward process using linear interpolation.
        
        For MeanFlow: z = (1 - t) * data + t * noise
        (same as Flow Matching)
        
        Args:
            data: Clean data tensor (batch_size, channels, height, width)
            noise: Noise tensor (batch_size, channels, height, width) 
            t: Timestep tensor (batch_size,) with values in [0, 1]
            
        Returns:
            Interpolated data tensor (batch_size, channels, height, width)
        """
        # Reshape t for broadcasting
        t = t.view(-1, 1, 1, 1)
        
        # Linear interpolation: z = (1-t)*data + t*noise
        z = (1.0 - t) * data + t * noise
        
        return z
    
    def reverse_process_step(self, xt, pred_mean_flow, t, t_next):
        """
        Perform one step of the reverse process using predicted mean flow.
        
        For MeanFlow, the predicted output is the integrated flow from current
        time to target time, so we can directly apply it.
        
        Args:
            xt: Current data (batch_size, channels, height, width)
            pred_mean_flow: Predicted mean flow (batch_size, channels, height, width)
            t: Current timestep (batch_size,)
            t_next: Next timestep (batch_size,)
            
        Returns:
            Updated data tensor at timestep t_next (batch_size, channels, height, width)
        """
        # For MeanFlow, we directly apply the predicted integrated flow
        # This is the key difference: pred_mean_flow already represents the displacement
        x_next = xt - pred_mean_flow
        
        return x_next
      
    def get_target(self, data, noise, t, r, dudt):
        """
        Get the target mean flow for model prediction.
        
        For MeanFlow:
        velocity v = noise - data
        target u_tgt = v - (t - r) * du/dt
        
        Args:
            data: Clean data tensor
            noise: Noise tensor  
            t: Timestep tensor
            r: Reference timestep tensor
            dudt: Time derivative of u (computed via JVP)
            
        Returns:
            Target mean flow tensor
        """
        # Basic velocity (same as Flow Matching)
        v = noise - data
        
        # Correction term based on integration interval
        t_diff = (t - r).view(-1, 1, 1, 1)
        u_tgt = v - t_diff * dudt
        
        return u_tgt