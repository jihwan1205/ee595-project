import torch
from src.base_model import BaseGenerativeModel
import torch.nn.functional as F


class FlowMatchingModel(BaseGenerativeModel):
    """
    Flow Matching Model
    
    Implements Conditional Flow Matching (CFM) for generative modeling.
    The model learns to predict velocity fields along optimal transport paths
    from noise to data distributions.
    
    Key differences from DDPM:
    - Predicts velocity field instead of noise
    - Uses continuous time t ∈ [0, 1] instead of discrete timesteps
    - Forward process is deterministic linear interpolation
    - Simpler reverse process using Euler integration
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
    
    def _setup(self, **kwargs):
        """Setup model-specific parameters. Override in subclasses."""
        # Flow Matching typically doesn't need additional parameters
        # but can be extended with velocity weighting, etc.
        pass
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute the training loss for Flow Matching.
        
        The model learns to predict the velocity field v_t that moves
        from noise distribution to data distribution.
        
        Args:
            data: Data samples (clean images)
            noise: Prior samples (Gaussian noise)
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Loss tensor (MSE between predicted and target velocity)
        """
        device = self.device
        batch_size = data.size(0)

        # Sample random timesteps uniformly from [0, 1]
        t = self.scheduler.sample_timesteps(batch_size, device)
        
        # Apply forward process: interpolate between noise and data
        x_t = self.scheduler.forward_process(data, noise, t)
        
        # Predict velocity field at time t
        pred_velocity = self.predict(x_t, t)
        
        # Get target velocity (data - noise for optimal transport)
        target_velocity = self.scheduler.get_target(data, noise, t)

        # Compute MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss
    
    def predict(self, xt, t, **kwargs):
        """
        Predict the velocity field given data and timestep.
        
        Args:
            xt: Current data tensor (interpolated between noise and data)
            t: Timestep tensor (values in [0, 1])
            **kwargs: Additional arguments
            
        Returns:
            Predicted velocity field tensor
        """
        # For Flow Matching, the network takes continuous time t ∈ [0, 1]
        # We need to convert it to the format expected by the network
        # Most networks expect timesteps scaled to [0, num_train_timesteps-1]
        t_scaled = t * (self.scheduler.num_train_timesteps - 1)
        
        pred_velocity = self.network(xt, t_scaled)
        return pred_velocity
    
    @torch.no_grad()
    def sample(self, shape, num_inference_timesteps=20, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples using Flow Matching.
        
        Starting from pure noise, we integrate the learned velocity field
        to arrive at the data distribution.
        
        Args:
            shape: Shape of the samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of integration steps
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments
        
        Returns:
            Generated samples or trajectory
        """
        device = self.device
        
        # Start from pure noise at t=0
        x = torch.randn(shape, device=device)
        traj = [x.clone()] if return_traj else None

        # Create timesteps from 0 (noise) to 1 (data)
        # We reverse the integration: start at t=0, end at t=1
        timesteps = torch.linspace(0, 1, num_inference_timesteps + 1, device=device)
        
        for i in range(len(timesteps) - 1):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Repeat timesteps for batch dimension
            t_batch = t_current.unsqueeze(0).repeat(shape[0])
            t_next_batch = t_next.unsqueeze(0).repeat(shape[0])

            # Predict velocity field at current time
            pred_velocity = self.predict(x, t_batch)
            
            # Integrate forward: x_{t+dt} = x_t + v_t * dt
            dt = (t_next_batch - t_batch).view(-1, 1, 1, 1)
            x = x + pred_velocity * dt

            if return_traj:
                traj.append(x.clone())

            if verbose :
                print(f"Sampling step {i}/{num_inference_timesteps}")

        return traj if return_traj else x

