from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BaseConfig:
    """Base configuration for all experiments."""
    # Device and reproducibility
    device: str = 'cuda'
    seed: int = 42
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    
    # Dataset
    dataset: str = 'mnist'  # 'mnist', 'cifar10', 'cifar100'
    num_workers: int = 4
    
    # Logging and checkpointing
    experiment_name: str = 'baseline'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_every: int = 5  # Save checkpoint every N epochs
    log_every: int = 50  # Log metrics every N batches
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    
    # TensorBoard and W&B
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = 'dla-lab1'
    wandb_entity: Optional[str] = None


@dataclass
class MLPConfig(BaseConfig):
    """Configuration for MLP experiments."""
    # Model architecture
    input_size: int = 784  # 28*28 for MNIST
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    num_classes: int = 10
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: str = 'relu'  # 'relu', 'gelu', 'leaky_relu'
    
    # Residual connections
    use_residual: bool = False
    
    # Experiment type
    experiment_name: str = 'mlp_baseline'


@dataclass
class CNNConfig(BaseConfig):
    """Configuration for CNN experiments."""
    # Model architecture
    in_channels: int = 3  # 1 for MNIST, 3 for CIFAR
    base_channels: int = 64
    num_blocks: int = 3
    channels_list: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_classes: int = 10
    dropout: float = 0.2
    
    # Residual connections
    use_residual: bool = False
    
    # Block configuration
    conv_per_block: int = 2
    kernel_size: int = 3
    use_batch_norm: bool = True
    
    # Pooling
    use_adaptive_pooling: bool = True
    pool_size: int = 2
    
    # Experiment type
    experiment_name: str = 'cnn_baseline'


@dataclass
class DistillationConfig(CNNConfig):
    """Configuration for Knowledge Distillation experiments."""
    # Teacher model
    teacher_checkpoint: str = './checkpoints/teacher_best.pt'
    
    # Distillation parameters
    temperature: float = 3.0
    alpha: float = 0.5  # Weight for distillation loss (1-alpha for hard labels)
    
    # Student model (typically smaller than teacher)
    base_channels: int = 32
    channels_list: List[int] = field(default_factory=lambda: [32, 64, 128])
    
    experiment_name: str = 'distillation'


@dataclass
class FineTuneConfig(CNNConfig):
    """Configuration for Fine-tuning experiments."""
    # Pre-trained model
    pretrained_checkpoint: str = './checkpoints/cifar10_best.pt'
    
    # Fine-tuning strategy
    freeze_backbone: bool = True
    unfreeze_after_epoch: int = 5
    
    # Target dataset
    target_dataset: str = 'cifar100'
    num_classes: int = 100
    
    # Fine-tuning learning rate (typically lower)
    lr: float = 1e-4
    
    experiment_name: str = 'finetune'


def get_config(experiment_type: str, **kwargs) -> BaseConfig:
    """
    Factory function to get appropriate configuration.
    
    Args:
        experiment_type: Type of experiment ('mlp', 'cnn', 'distillation', 'finetune')
        **kwargs: Additional parameters to override defaults
    
    Returns:
        Configuration object
    """
    config_map = {
        'mlp': MLPConfig,
        'cnn': CNNConfig,
        'distillation': DistillationConfig,
        'finetune': FineTuneConfig,
    }
    
    if experiment_type not in config_map:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Choose from {list(config_map.keys())}")
    
    config_class = config_map[experiment_type]
    return config_class(**kwargs)
