import torch
from ddpm_scheduler import DDPM_Scheduler


class DDIM_Scheduler(DDPM_Scheduler):
     def __init__(self, 
        num_train_timesteps: int = 1000, 
        beta_start: float = 1e-4, 
        beta_end: float = 0.02,
        diffusion_objective: str = 'pred_noise',
        beta_schedule: str = 'linear',
        **kwargs
    ):
        super().__init__(num_train_timesteps, beta_start, beta_end, diffusion_objective, beta_schedule, **kwargs)
    