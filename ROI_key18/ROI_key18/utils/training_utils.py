# utils/training_utils.py

import math
import random
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """
    Exponential Moving Average of model parameters.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class WarmupCosineSchedule:
    """
    Learning rate scheduler with warmup and cosine annealing.
    """
    def __init__(self, 
                 optimizer: Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_scale: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_scale = min_lr_scale
        self.step_count = 0
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / \
                      max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_scale + 0.5 * (1 - self.min_lr_scale) * \
                   (1 + math.cos(math.pi * progress))
        
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group['lr'] = base_lr * scale
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr_scale': self.min_lr_scale,
            'step_count': self.step_count,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr_scale = state_dict['min_lr_scale']
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def save_checkpoint(path: str, 
                   model: nn.Module,
                   optimizer: Optimizer,
                   scheduler: Optional[WarmupCosineSchedule],
                   scaler: Optional[torch.cuda.amp.GradScaler],
                   ema: Optional[EMA],
                   epoch: int,
                   global_step: int,
                   best_metric: float,
                   config: Any):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'best_metric': best_metric,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: str,
                   model: nn.Module,
                   optimizer: Optional[Optimizer] = None,
                   scheduler: Optional[WarmupCosineSchedule] = None,
                   scaler: Optional[torch.cuda.amp.GradScaler] = None,
                   ema: Optional[EMA] = None,
                   device: str = 'cuda') -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from: {path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("Scaler state loaded")
    
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        print("EMA state loaded")
    
    return checkpoint


def compute_metrics(predictions: np.ndarray, 
                   targets: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        predictions: [N] array
        targets: [N] array
    
    Returns:
        Dict with MAE, RMSE, Pearson r
    """
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    # Pearson correlation
    if len(predictions) > 1:
        r = float(np.corrcoef(predictions, targets)[0, 1])
    else:
        r = 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r': r
    }