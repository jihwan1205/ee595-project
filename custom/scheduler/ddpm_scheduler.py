import torch
from src.base_model import BaseScheduler


class DDPMScheduler(BaseScheduler):
    """
    Custom Scheduler Skeleton
    
    TODO: Students need to implement this class in their own file.
    Required methods:
    1. sample_timesteps: Sample random timesteps for training
    2. forward_process: Apply forward process to transform data
    3. reverse_process_step: Perform one step of the reverse process
    4. get_target: Get target for model prediction
    """

    def __init__(self, 
        num_train_timesteps: int = 1000, 
        beta_start: float = 1e-4, 
        beta_end: float = 0.02,
        diffusion_objective: str = 'pred_noise',
        beta_schedule: str = 'linear',
        **kwargs
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.diffusion_objective = diffusion_objective
        self.beta_schedule = beta_schedule
        super().__init__(num_train_timesteps, **kwargs)
    
    def _setup(self, **kwargs):
        """
        Setup scheduler-specific parameters. Override in subclasses.
        e.g., betas, alphas, alphas_cumprod, etc.
        """
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))


    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place timesteps on
            
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        return timesteps
    
    def forward_process(self, data, noise, t):
        """
        Apply the forward process to transform clean data to noisy data.
        
        Args:
            data: Clean data tensor (batch_size, channels, height, width)
            noise: Noise tensor (batch_size, channels, height, width) 
            t: Timestep tensor (batch_size,)
            
        Returns:
            Noisy data tensor (batch_size, channels, height, width)
        """
        mu = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * data
        sigma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = mu + sigma * noise
        return x_t
    
    def reverse_process_step(self, xt, pred, t, t_next):
        """
        Perform one step of the reverse (denoising) process.
        
        Args:
            xt: Current noisy data (batch_size, channels, height, width)
            pred: Model prediction (batch_size, channels, height, width)
            t: Current timestep (batch_size,)
            t_next: Next timestep (batch_size,)
            
        Returns:
            Updated data tensor at timestep t_next (batch_size, channels, height, width)
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = (1.0 / torch.sqrt(alpha_t))
        
        mean = sqrt_recip_alpha * (xt - (beta_t / sqrt_one_minus_alphas_cumprod) * pred)
        sigma = torch.sqrt(self.posterior_variance[t])
        sigma = torch.where(t_next < 0 , torch.zeros_like(sigma), sigma).view(-1, 1, 1, 1) # No noise when t_next is smaller than 0
        z = torch.randn_like(xt)   
        
        x_prev = mean + sigma * z
        return x_prev
      
    def get_target(self, data, noise, t):
        """
        Get the target for model prediction used in L2 loss computation.
        
        e.g., for diffusion models: target = noise (for noise prediction) or x0 (for x0 prediction)
              for flow models: target = velocity field
        
        Args:
            data: Clean data tensor
            noise: Noise tensor  
            t: Timestep tensor
            
        Returns:
            Target tensor for model prediction
        """
        return noise