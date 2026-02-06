"""
Miscellaneous utility functions.

Common utilities for:
- Random seed setting
- Device management
- Parameter counting
- Time formatting
- Learning rate extraction
"""
import logging
import random
from typing import Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Make cudnn deterministic (may reduce performance)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)
                If None, automatically select GPU if available

    Returns:
        torch.device object
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    return device

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count number of parameters in model.

    Paper Context:
    MAE-AST Base has ~86M parameters (encoder + decoder during pretraining).
    During fine-tuning, decoder is removed, leaving ~85M parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())

    return params

def format_params(num_params: int) -> str:
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def print_model_info(model: nn.Module):
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    logger.info("=" * 50)
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {format_params(total_params)} ({total_params:,})")
    logger.info(f"  Trainable parameters: {format_params(trainable_params)} ({trainable_params:,})")
    logger.info(f"  Non-trainable parameters: {format_params(total_params - trainable_params)}")
    logger.info("=" * 50)


def format_time(seconds: float) -> str:
    #Format seconds as HH:MM:SS.
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    # Get current learning rate from optimizer.
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

def save_config(config: dict, filepath: str):
    # save configuration to YAML file.
    import yaml
    from pathlib import Path

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> dict:
    # Load configuration from YAML file.
    import yaml
    from pathlib import Path

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {filepath}")
    return config


class Timer:
    """
    Simple timer context manager.

    Usage:
        with Timer("Training epoch"):
            # ... training code ...
    """

    def __init__(self, name: str = "Operation", logger_obj: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger_obj or logger
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} took {format_time(self.elapsed)}")