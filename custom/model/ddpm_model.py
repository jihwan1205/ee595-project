import torch
from src.base_model import BaseGenerativeModel
import torch.nn.functional as F


class DDPMModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    Students need to implement this class by inheriting from BaseGenerativeModel.
    This class wraps the network and scheduler to provide training and sampling interfaces.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
    
    def _setup(self, **kwargs):
        # TODO: Initialize your model-specific parameters (e.g., EMA, loss weights)
        """Setup model-specific parameters. Override in subclasses."""
        pass
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute the training loss.
        
        Args:
            data: Data samples (clean images)
            noise: Prior samples (noise for diffusion, x0 for flow)
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Loss tensor
        """
        device = self.device
        batch_size = data.size(0)

        t = self.scheduler.sample_timesteps(batch_size, device)
        x_t = self.scheduler.forward_process(data, noise, t)
        pred = self.predict(x_t, t)
        target = self.scheduler.get_target(data, noise, t)

        loss = F.mse_loss(pred, target)
        return loss
    
    def predict(self, xt, t, **kwargs):
        """
        Make a prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data tensor
            t: Timestep tensor
            **kwargs: Additional arguments (e.g., aux_timestep for some advanced models)
            
        Returns:
            Model prediction tensor
        """
        pred = self.network(xt, t)
        return pred
    
    @torch.no_grad()
    def sample(self, shape, num_inference_timesteps=20, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples from the model.
        
        Args:
            shape: Shape of the samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of inference steps
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments for specific model types
        
        Returns:
            Generated samples or trajectory
        """
        device = self.device
        x = torch.randn(shape, device=device)
        traj = [x.clone()] if return_traj else None

        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_timesteps, dtype=torch.long, device=device)
        for i, t in enumerate(timesteps):
            t = t.long().unsqueeze(0).repeat(shape[0])
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=device)
            t_next = t_next.long().unsqueeze(0).repeat(shape[0])

            pred_noise = self.predict(x, t)
            x = self.scheduler.reverse_process_step(x, pred_noise, t, t_next)

            if return_traj:
                traj.append(x.clone())

            if verbose and (i % (num_inference_timesteps // 5) == 0):
                print(f"Sampling step {i}/{num_inference_timesteps}")

        return traj if return_traj else x
