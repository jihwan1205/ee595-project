#!/usr/bin/env python3
"""
Template for implementing custom generative models
Students should create their own implementation by inheriting from the base classes.

This file provides skeleton code for implementing generative models.
Students need to implement the TODO sections in their own files.
"""

from src.network import UNet
from custom.scheduler.ddpm_scheduler import DDPMScheduler
from custom.model.ddpm_model import DDPMModel
from custom.scheduler.fm_scheduler import FlowMatchingScheduler
from custom.model.fm_model import FlowMatchingModel
from custom.scheduler.meanflow_scheduler import MeanFlowScheduler
from custom.model.meanflow_model import MeanFlowModel
# from custom.scheduler.modular_meanflow_scheduler import ModularMeanFlowScheduler
# from custom.model.modular_meanflow_model import ModularMeanFlowModel
# from custom.scheduler.alphaflow_scheduler import AlphaFlowScheduler
# from custom.model.alphaflow_model import AlphaFlowModel

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_custom_model(model_type='DDPM', **kwargs):
    """
    Example function to create a custom generative model.
    
    Students should modify this function to create their specific model.
    
    Args:
        device: Device to place model on
        model_type: Type of model to create ('DDPM', 'FlowMatching', or 'MeanFlow')
        **kwargs: Additional arguments that can be passed to network or scheduler
                  (e.g., num_train_timesteps, use_additional_condition for scalar conditions
                   like step size in Shortcut Models or end timestep in Consistency Trajectory Models, etc.)
    """
    
    # MeanFlow requires additional condition for r (reference time)
    if model_type == 'MeanFlow':
        kwargs['use_additional_condition'] = True
    
    # Create U-Net backbone with FIXED hyperparameters
    # DO NOT MODIFY THESE HYPERPARAMETERS
    network = UNet(
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_additional_condition=kwargs.get('use_additional_condition', False)
    )
    
    if model_type == 'DDPM':
        scheduler = DDPMScheduler(**kwargs)
        model = DDPMModel(network, scheduler, **kwargs)
    elif model_type == 'FlowMatching':
        scheduler = FlowMatchingScheduler(**kwargs)
        model = FlowMatchingModel(network, scheduler, **kwargs)
    elif model_type == 'MeanFlow':
        scheduler = MeanFlowScheduler(**kwargs)
        model = MeanFlowModel(network, scheduler, **kwargs)
    # elif model_type == 'ModularMeanFlow':
    #     scheduler = ModularMeanFlowScheduler(**kwargs)
    #     model = ModularMeanFlowModel(network, scheduler, **kwargs)
    # elif model_type == 'AlphaFlow':
    #     scheduler = AlphaFlowScheduler(**kwargs)
    #     model = AlphaFlowModel(network, scheduler, **kwargs)
    else: 
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return model