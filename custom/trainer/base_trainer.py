import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
import argparse
from cleanfid import fid
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import gc

from src.sg_ratio_scheduler import SGRatioScheduler
from src.utils import save_model, tensor_to_pil_image, compute_gen_image_statistics, compute_image_folder_statistics, is_main, print, save_ema_full_checkpoint


class BaseTrainer:
    def __init__(self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        train_iterator: torch.utils.data.DataLoader,
        valid_iterator: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model_config: dict,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        sg_ratio_scheduler: SGRatioScheduler = None,
        ema = None,
        train_sampler = None,
    ):
        self.args = args
        self.rank = dist.get_rank() if self.args.use_ddp else self.args.device
        self.model = DDP(model.to(self.rank), device_ids=[self.rank]) if self.args.use_ddp else model.to(self.rank)
        self.module = self.model.module if self.args.use_ddp else self.model
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema = ema
        self.sg_ratio_scheduler = sg_ratio_scheduler
        self.model_config = model_config
        self.train_sampler = train_sampler
        
        self.args.save_dir.mkdir(exist_ok=True, parents=True)
    
    def _save_training_config(self):
        config = {
            'num_iterations': self.args.num_iterations,
            'lr': self.args.lr,
            'log_interval': self.args.log_interval,
            'save_interval': self.args.save_interval,
            'device': str(self.args.device),
        }
        with open(self.args.save_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Training config saved to: {self.args.save_dir / 'training_config.json'}")
        
    def _do_log(self):
        self._draw_loss_curve()
    
    def _draw_loss_curve(self):
        # Save training loss curve
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.args.save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save training curve: {e}")
            
    def _log_wandb_train(self):
        log_dict = {}
        log_dict['train/train_loss'] = self.train_losses[-1]
        if self.lr_scheduler is not None:
            log_dict["train/learning_rate"] = self.optimizer.param_groups[0]['lr']
        # if self.args.model_type == 'AlphaFlow':
        #     log_dict["train/alpha"] = self.module.alpha
        wandb.log(log_dict, step=self.iteration)

    def _log_wandb_eval(self, log_dict: dict):
        wandb.log(log_dict, step=self.iteration)

    def _save_model(self, ckpt_name: str):
        checkpoint_path = self.args.save_dir / ckpt_name
        
        # Save main checkpoint WITHOUT EMA (only model, optimizer, lr_scheduler, iteration)
        save_model(
            self.module, 
            str(checkpoint_path), 
            self.model_config,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            iteration=self.iteration,
            ema_state_dict=None  # Don't include EMA in main checkpoint
        )
        
        # Save EMA checkpoint with full training state (for resuming)
        if self.args.use_ema and self.ema is not None:
            ema_state_dict = self.ema.state_dict()
            ema_ckpt_name = ckpt_name.replace('.pt', '_ema.pt')
            ema_checkpoint_path = self.args.save_dir / ema_ckpt_name
            # Save EMA checkpoint with all training state for resuming
            save_ema_full_checkpoint(
                model=self.module,
                ema_state_dict=ema_state_dict,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                iteration=self.iteration,
                checkpoint_path=str(ema_checkpoint_path)
            )

    def _make_sample_grid(self, samples: torch.Tensor, n_row: int = 4, n_col: int = 4) -> torch.Tensor:
        B, C, H, W = samples.shape
        assert B >= n_row * n_col, "Not enough samples to make the grid"
        line_width = 1
        samples = F.pad(samples, (0, line_width, 0, line_width), value=-0.5)

        H_pad, W_pad = H + line_width, W + line_width

        grid = samples.reshape(n_row, n_col, C, H_pad, W_pad).permute(2, 0, 3, 1, 4).reshape(C, n_row * H_pad, n_col * W_pad)
        grid = tensor_to_pil_image(grid)
        return grid

    @torch.no_grad()
    def _sample(self, sample_name: str = '', nfe: int = 1, batch_size: int = 4, do_save: bool = False, do_log: bool = False) -> torch.Tensor:
        self.model.eval()
        
        # Use EMA parameters for sampling if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        n_r, n_c = 4, 4
        # Get image resolution from args (set by data module)
        image_resolution = getattr(self.args, 'image_resolution', 64)
        C, H, W = 3, image_resolution, image_resolution
        shape = (batch_size, C, H, W)
        
        samples = self.module.sample(shape, num_inference_timesteps=nfe)
        pil_images = tensor_to_pil_image(samples)
        
        if do_log and self.args.use_wandb:
            samples_grid = self._make_sample_grid(samples, n_row=n_r, n_col=n_c)
            wandb.log({f"samples/nfe_{nfe}": wandb.Image(samples_grid)}, step=self.iteration)

        if do_save:
            for i, img in enumerate(pil_images):
                img_name = f"{sample_name + '_' if sample_name else ''}iter{self.iteration}_nfe{nfe}_sample_{i}.png"
                img.save(self.args.save_dir / img_name)
        
        # Restore original parameters
        if self.ema is not None:
            self.ema.restore()

        self.model.train()
        return samples
    
    @torch.no_grad()    
    def _evaluate_fid(self, nfe: int = 20) -> float:
        generated_images = self._sample(nfe=nfe, batch_size=self.args.fid_batch_size)
        if not hasattr(self, 'feat_model'):
            self.feat_model = fid.build_feature_extractor(mode='clean', device=self.rank, use_dataparallel=False)

        mu, sigma = compute_gen_image_statistics(generated_images, feat_model=self.feat_model, num_gen=self.args.fid_batch_size, batch_size=self.args.fid_batch_size, use_ddp=self.args.use_ddp, device=self.rank)
        mu2, sigma2 = compute_image_folder_statistics(folder_path=self.args.val_reference_dir, feat_model=self.feat_model, device=self.rank)
        
        del self.feat_model
        
        fid_score = fid.frechet_distance(mu, sigma, mu2, sigma2)
        return fid_score

    @torch.no_grad()
    def _eval_step(self, data: torch.Tensor) -> float:
        # noise = torch.randn_like(data)
        noise, data = self._get_noise_and_data(data)
        loss = self.module.compute_loss(data, noise)
        return loss

    def _move_to_device(self, data, device):
        """Move data to device, handling both tensor and dictionary formats."""
        if isinstance(data, dict):
            return {k: v.to(device) for k, v in data.items()}
        else:
            return data.to(device)
    
    def _get_noise_and_data(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.use_reflow:
            noise = data['x0']
            data = data['x1']
        else:
            noise = torch.randn_like(data)
        return noise, data
    
    @torch.no_grad()
    def _evaluate_loss(self) -> float:
        # Use EMA parameters for evaluation if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        total = 0.0
        count = 0
        max_batches = self.args.val_max_batches
        
        pbar = tqdm(self.valid_iterator, desc="Validating", leave=False, total=max_batches, disable=not is_main())
        for b_idx, data in enumerate(pbar):
            if max_batches is not None and b_idx >= max_batches: break
            
            data = self._move_to_device(data, self.rank)
            loss = self._eval_step(data)
            total += loss
            count += 1
            
            if count > 0:
                pbar.set_postfix({"Val Loss": f"{total/count:.4f}"})
        
        val_loss = total / max(1, count)
        if self.args.use_ddp:
            dist.reduce(val_loss, dst=0, op=dist.ReduceOp.AVG)
        
        # Restore original parameters
        if self.ema is not None:
            self.ema.restore()
        
        return val_loss.item()
    
    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        log_dict = {}
        
        eval_loss = self._evaluate_loss()
        log_dict["valid/val_loss"] = eval_loss
        
        # Print validation loss with flush
        print(f"\n[Iter {self.iteration}] Validation Loss: {eval_loss:.6f}", flush=True)
        
        if self.args.eval_fid:
            pbar = tqdm(self.args.fid_nfe_list, leave=False, total=len(self.args.fid_nfe_list), disable=not is_main())
            for nfe in pbar:
                pbar.set_description(f"Evaluating FID (nfe={nfe})")
                fid_score = self._evaluate_fid(nfe=nfe)
                log_dict[f"valid/FID/nfe={nfe}"] = fid_score
                # Print FID score with flush
                print(f"[Iter {self.iteration}] FID (nfe={nfe}): {fid_score:.4f}", flush=True)

                if fid_score < self.best_fid_per_nfe[nfe]:
                    self.best_fid_per_nfe[nfe] = fid_score
                    self._save_model(ckpt_name=f"best_nfe{nfe}.pt")
        
        if self.args.use_wandb and is_main():
            self._log_wandb_eval(log_dict)
        
        self.model.train()
        gc.collect()

    def _train_step(self, data: torch.Tensor) -> torch.Tensor:
        # noise = torch.randn_like(data)
        noise, data = self._get_noise_and_data(data)
        
        loss = self.module.compute_loss(data, noise)

        self.optimizer.zero_grad()
        loss.backward()
        if self.args.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        
        # Step learning rate scheduler if it exists
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.sg_ratio_scheduler is not None:
            self.sg_ratio_scheduler.step()
        
        # Update EMA if it exists
        if self.ema is not None:
            self.ema.update()
        
        return loss
    
    def train(self):
        print(f"Starting training for {self.args.num_iterations} iterations...")
        print(f"Model: {type(self.model).__name__}")
        print(f"Device: {self.args.device}")
        print(f"Save directory: {self.args.save_dir}")
        print(self.args)

        if is_main():
            self._save_training_config()
        self.model.train()

        if self.sg_ratio_scheduler is not None:
            self.sg_ratio_scheduler.initialize()
        
        self.train_losses = [] # Store training losses as class attribute
        # Initialize iteration if not already set (e.g., from resume)
        if not hasattr(self, 'iteration') or self.iteration is None:
            self.iteration = 0
        start_iteration = self.iteration + 1 if self.iteration > 0 else 1
        self.best_fid_per_nfe = {nfe: float('inf') for nfe in self.args.fid_nfe_list}
        
        if not self.args.skip_initial_evaluation and start_iteration == 1:
            self._evaluate()

        pbar = tqdm(range(start_iteration, self.args.num_iterations + 1), desc="Training", disable=not is_main())
        for iteration in pbar:
            self.iteration = iteration # Store current iteration as class attribute
            self.model.iteration = iteration
            
            # Update DistributedSampler epoch periodically to ensure proper data shuffling
            if self.train_sampler is not None and isinstance(self.train_sampler, DistributedSampler):
                # Calculate epoch based on iteration (assuming dataset size / batch_size iterations per epoch)
                # Update epoch every 1000 iterations to ensure proper shuffling
                if iteration % 1000 == 1:
                    epoch = (iteration - 1) // 1000
                    self.train_sampler.set_epoch(epoch)
            
            data = next(self.train_iterator)
            data = self._move_to_device(data, self.rank)
            loss = self._train_step(data)
            
            loss = loss.item()  
            self.train_losses.append(loss)
            avg_loss = sum(self.train_losses[-self.args.log_interval:]) / min(self.args.log_interval, len(self.train_losses))
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Avg Loss": f"{avg_loss:.4f}"})
            
            # only rank 0 does logging and saving
            if self.args.use_wandb and is_main():
                self._log_wandb_train()

            if iteration % self.args.val_interval == 0:
                self._evaluate()

            if iteration % self.args.sample_interval == 0 and is_main():
                for nfe in self.args.sample_nfe_list:
                    self._sample(nfe=nfe, batch_size=self.args.sample_batch_size, do_save=False, do_log=True)

            if iteration % self.args.save_interval == 0 and is_main():
                ckpt_name = f"checkpoint_iter_{iteration}.pt"
                self._save_model(ckpt_name)
        
        # Save final model
        if is_main():
            final_ckpt_name = "final_model.pt"
            self._save_model(final_ckpt_name)
        
        pbar.close()