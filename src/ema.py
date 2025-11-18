import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Maintains a moving average of model parameters that can be used
    for inference to improve stability and performance.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay rate (default: 0.9999)
            device: Device to store EMA parameters
        """
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create shadow parameters (EMA copy of model parameters)
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    @torch.no_grad()
    def update(self):
        """
        Update EMA parameters.
        
        Should be called after each optimizer step:
        shadow = decay * shadow + (1 - decay) * param
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    @torch.no_grad()
    def apply_shadow(self):
        """
        Apply EMA parameters to the model.
        Backs up current parameters first.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    @torch.no_grad()
    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state dict for saving."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict.
        
        Handles both old format ({'state_dict': shadow}) and new format ({'decay': ..., 'shadow': ...}).
        """
        # Handle old format where state_dict was saved as {'state_dict': shadow}
        if 'state_dict' in state_dict and 'shadow' not in state_dict:
            self.shadow = state_dict['state_dict']
        else:
            self.shadow = state_dict['shadow']
        
        # Update decay if present in state_dict
        if 'decay' in state_dict:
            self.decay = state_dict['decay']

