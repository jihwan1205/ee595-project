import torch
from src.base_model import BaseGenerativeModel
import torch.nn.functional as F



def adaptive_weighted_loss(u, u_tgt, loss_eps, loss_p):
    loss = (u - u_tgt) ** 2
    loss = loss.sum(dim=(1, 2, 3))

    adp_wt = (loss.detach() + loss_eps) ** loss_p
    loss = loss / adp_wt
    loss = loss.mean()
    return loss

class MeanFlowModel(BaseGenerativeModel):
    """
    MeanFlow Model
    
    Implements MeanFlow which predicts the integrated velocity (mean flow)
    over a time interval [r, t] instead of instantaneous velocity.
    
    Key differences from Flow Matching:
    - Predicts mean flow over interval [r, t] instead of velocity at t
    - Uses JVP (Jacobian-Vector Product) to compute du/dt
    - Network takes additional parameter r (start of integration interval)
    - Enables 1-step generation: x = noise - u(noise, r=0, t=1)
    """

    def __init__(self, network, scheduler, meanflow_loss_fn='adaptive_loss', meanflow_scale_timestep=False, meanflow_loss_eps=1e-3, meanflow_loss_p=1.0, meanflow_sg_ratio=0.0, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        self.scale_timestep = meanflow_scale_timestep
        self.loss_eps = meanflow_loss_eps
        self.loss_p = meanflow_loss_p
        self.sg_ratio = meanflow_sg_ratio
        
        if meanflow_loss_fn == 'mse_loss':
            self.loss_fn = F.mse_loss
        elif meanflow_loss_fn == 'adaptive_loss':
            self.loss_fn = lambda u, u_tgt: adaptive_weighted_loss(u, u_tgt, self.loss_eps, self.loss_p)
        else:
            raise ValueError(f"Unsupported meanflow_loss_fn: {meanflow_loss_fn}")

    def _setup(self, **kwargs):
        """Setup model-specific parameters."""
        pass
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute the training loss for MeanFlow.
        
        The model learns to predict the mean flow u over interval [r, t]
        using JVP to compute the time derivative.
        
        Args:
            data: Data samples (clean images)
            noise: Prior samples (Gaussian noise)
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor (MSE between predicted and target mean flow)
        """
        device = self.device
        batch_size = data.size(0)

        # Sample random timesteps t and r uniformly from [0, 1]
        t, r = self.scheduler.sample_timesteps(batch_size, device)
        
        # Forward process: z = (1-t)*data + t*noise
        z = self.scheduler.forward_process(data, noise, t)
        
        # Velocity vector: v = noise - data
        v = noise - data
        
        # Compute u and du/dt using JVP
        # u = network(z, r, t)
        # du/dt is computed by taking derivative w.r.t. t with tangent vector 1
        u, dudt = self._compute_u_and_dudt(z, r, t, v)
        
        # Get target: u_tgt = v - (t - r) * du/dt
        u_tgt = self.scheduler.get_target(data, noise, t, r, dudt)
        
        # Stop gradient on u_tgt (as in the pseudo code)
        u_tgt = (1 - self.sg_ratio) * u_tgt.detach() + self.sg_ratio * u_tgt
        # Compute adaptive weighted MSE loss
        loss = self.loss_fn(u, u_tgt)
        
        return loss
    
    def _compute_u_and_dudt(self, z, r, t, v):
        """
        Compute u = network(z, r, t) and du/dt using JVP.
        
        Following pseudo code: u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
        
        Args:
            z: Interpolated data
            r: Reference time
            t: Current time
            v: Velocity vector (for JVP tangent, not used for dudt)
            
        Returns:
            Tuple of (u, du/dt)
        """
        # Compute u and du/dt using JVP
        # Following pseudo code: tangents = (v, 0, 1)
        # v for z, 0 for r, 1 for t
        drdt = torch.zeros_like(r)
        dtdt = torch.ones_like(t)
        
        # jvp returns: (output, jvp_output)
        # jvp_output = du/dz * v + du/dr * 0 + du/dt * 1 = du/dz * v + du/dt
        # This matches the pseudo code: jvp(fn, (z, r, t), (v, 0, 1))
        u, dudt = torch.func.jvp(
            func=lambda z_input, r_input, t_input: self.predict(z_input, r_input, t_input),
            primals=(z, r, t),
            tangents=(v, drdt, dtdt),
        )
        
        return u, dudt
    
    def predict(self, zt, r, t, **kwargs):
        """
        Predict the mean flow given data and timesteps.
        
        Args:
            zt: Current data tensor
            r: Reference timestep tensor (start of interval)
            t: Current timestep tensor (end of interval)
            **kwargs: Additional arguments
            
        Returns:
            Predicted mean flow tensor
        """
        # Scale timesteps to [0, num_train_timesteps-1] for network
        if self.scale_timestep:
            t = t * (self.scheduler.num_train_timesteps - 1)
            r = r * (self.scheduler.num_train_timesteps - 1)

        # Network needs to accept both r and t
        # We use the condition parameter for r (reference time)
        pred_mean_flow = self.network(zt, t, condition = t - r)
        
        return pred_mean_flow
    
    @torch.no_grad()
    def sample(self, shape, num_inference_timesteps=1, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples using MeanFlow.
        
        MeanFlow enables few-step generation. In the extreme case (1-step):
        x = noise - u(noise, r=0, t=1)
        
        Args:
            shape: Shape of the samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of integration steps (can be 1!)
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments
        
        Returns:
            Generated samples or trajectory
        """
        device = self.device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        traj = [x.clone()] if return_traj else None
        
        timesteps = torch.linspace(1.0, 0.0, steps=num_inference_timesteps + 1, device=device)
        batch_size = shape[0]
        
        for i in range(num_inference_timesteps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]

            # t_current > t_next
            t_batch = t_current.unsqueeze(0).repeat(batch_size)
            r_batch = t_next.unsqueeze(0).repeat(batch_size)
            
            u = self.predict(x, r_batch, t_batch) * (t_current - t_next)
            
            # Update: x = x - u (since we're going from noise to data)
            x = x - u
            
            if return_traj:
                traj.append(x.clone())
            
            if verbose:
                print(f"Sampling step {i+1}/{num_inference_timesteps}")

        return traj if return_traj else x

