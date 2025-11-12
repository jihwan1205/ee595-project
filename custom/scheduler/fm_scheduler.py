import torch
from src.base_model import BaseScheduler


class FlowMatchingScheduler(BaseScheduler):
    """
    Flow Matching Scheduler
    
    Implements Conditional Flow Matching (CFM) following optimal transport paths.
    Uses linear interpolation between noise (x0) and data (x1) with constant velocity.
    
    Forward process: x_t = (1 - t) * noise + t * data
    Velocity field: v_t = data - noise (constant along the path)
    """

    def __init__(self, 
        num_train_timesteps: int = 1000,
        sigma_min: float = 0.0,  # Minimum noise level at t=1
        **kwargs
    ):
        self.sigma_min = sigma_min
        super().__init__(num_train_timesteps, **kwargs)
    
    def _setup(self, **kwargs):
        """
        Setup scheduler-specific parameters for Flow Matching.
        Flow Matching doesn't require complex schedules like DDPM.
        """
        # Create a time grid for training
        # Time goes from 0 (pure noise) to 1 (pure data)
        timesteps = torch.linspace(0, 1, self.num_train_timesteps)
        self.register_buffer("timesteps", timesteps)

    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        For Flow Matching, we sample uniformly from [0, 1].
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place timesteps on
            
        Returns:
            Tensor of shape (batch_size,) with timestep values in [0, 1]
        """
        # Sample continuous time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)
        return t
    
    def forward_process(self, data, noise, t):
        """
        Apply the forward process using linear interpolation.
        
        For Flow Matching: x_t = (1 - t) * x_0 + t * x_1
        where x_0 is noise and x_1 is data.
        
        Args:
            data: Clean data tensor (batch_size, channels, height, width)
            noise: Noise tensor (batch_size, channels, height, width) 
            t: Timestep tensor (batch_size,) with values in [0, 1]
            
        Returns:
            Interpolated data tensor (batch_size, channels, height, width)
        """
        # Reshape t for broadcasting
        t = t.view(-1, 1, 1, 1)
        
        # Linear interpolation from noise to data
        x_t = (1.0 - t) * noise + t * data
        
        return x_t
    
    def reverse_process_step(self, xt, pred_velocity, t, t_next):
        """
        Perform one step of the reverse process using Euler integration.
        
        For Flow Matching, we integrate the velocity field backwards:
        x_{t-dt} = x_t - v_t * dt
        
        Args:
            xt: Current data (batch_size, channels, height, width)
            pred_velocity: Predicted velocity field (batch_size, channels, height, width)
            t: Current timestep (batch_size,)
            t_next: Next timestep (batch_size,)
            
        Returns:
            Updated data tensor at timestep t_next (batch_size, channels, height, width)
        """
        # Compute dt (step size)
        # For reverse process, we move from t to t_next where t_next < t
        dt = (t - t_next).view(-1, 1, 1, 1)
        
        # Euler integration: x_{t-dt} = x_t - v_t * dt
        x_next = xt - pred_velocity * dt
        
        return x_next
      
    def get_target(self, data, noise, t):
        """
        Get the target velocity field for model prediction.
        
        For Flow Matching with linear interpolation (optimal transport):
        The velocity field is constant: v_t = x_1 - x_0 = data - noise
        
        Args:
            data: Clean data tensor
            noise: Noise tensor  
            t: Timestep tensor
            
        Returns:
            Target velocity field tensor
        """
        # The optimal transport velocity is constant along the path
        velocity = data - noise
        return velocity

