import torch
from src.base_model import BaseGenerativeModel

class SGRatioScheduler:
    """
    Scheduler for managing the stop-gradient ratio during training.
    This allows dynamic adjustment of the stop-gradient ratio over training iterations.
    """
    def __init__(self, model: BaseGenerativeModel, sg_ratio_warmup_iters: int):
        """
        Initialize the SGRatioScheduler.
        
        Args:
            sg_ratio_warmup_iters: Number of iterations for warmup phase
            sg_ratio_start: Initial stop-gradient ratio
            sg_ratio_end: Final stop-gradient ratio
            total_iters: Total number of training iterations
        """
        self.model = model
        self.sg_ratio_warmup_iters = sg_ratio_warmup_iters
        self.current_iter = 0
        self.current_sg_ratio = 0.0
    
    def _set_current_sg_ratio(self):
        self.current_sg_ratio = min(1.0, self.current_iter / self.sg_ratio_warmup_iters)
    
    def initialize(self):
        self._set_current_sg_ratio()
        self.model.sg_ratio = self.current_sg_ratio

    def step(self):
        """
        Update the stop-gradient ratio based on the current iteration.
        """
        self.current_iter += 1
        self._set_current_sg_ratio()
        self.model.sg_ratio = self.current_sg_ratio
